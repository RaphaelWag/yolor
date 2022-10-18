# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou, wh_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    # TODO normalize coordinates
    device = targets.device
    # print(device)
    lcls, lbox, lobj, ldst, lrad_x, lrad_y, lang_x, lang_y = torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device), \
                                                             torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, tdst, trad_x, trad_y, tang_x, tang_y = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    MSEdst, MSErad_x, MSErad_y, MSEang_x, MSEang_y = nn.MSELoss().to(device), nn.MSELoss().to(device), \
                                                     nn.MSELoss().to(device), nn.MSELoss().to(device), \
                                                     nn.MSELoss().to(device)

    MSE_box_x, MSE_box_y = nn.MSELoss().to(device), nn.MSELoss().to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    # balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.25, 0.06]  # P3-5 or P3-6 new
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Box Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pwh = torch.ones(size=pwh.shape).to(device) * h['box_size']
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            # lbox += (1.0 - iou).mean()  # iou loss
            lbox = MSE_box_x(pbox.T[..., 0], tbox[i][..., 0]) + MSE_box_y(pbox.T[..., 1], tbox[i][..., 1])
            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 8:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 8:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # Single Regression
            pdst = ps[..., 5].sigmoid()
            prad_x = ps[..., 6].sigmoid()
            prad_y = ps[..., 7].sigmoid()
            pang_x = ps[..., 8].sigmoid()
            pang_y = ps[..., 9].sigmoid()
            if model.training:
                ldst += MSEdst(pdst, tdst[i])  # we can replace this loss function with rmse or anything we like
                lrad_x += MSErad_x(prad_x, trad_x[i])
                lrad_y += MSErad_y(prad_y, trad_y[i])
                lang_x += MSEang_x(pang_x, tang_x[i])
                lang_y += MSEang_y(pang_y, tang_y[i])

            else:
                ldst += MSEdst(pdst, tdst[i])
                lrad_x += MSErad_x(prad_x, trad_x[i])
                lrad_y += MSErad_y(prad_y, trad_y[i])
                lang_x += MSEang_x(pang_x, tang_x[i])
                lang_y += MSEang_y(pang_y, tang_y[i])

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
    lcls *= h['cls'] * s
    if model.training:
        ldst *= h['distance'] * s
        lrad_x *= h['radius'] * s
        lrad_y *= h['radius'] * s
        lang_x *= h['angle'] * s
        lang_y *= h['angle'] * s
    else:
        ldst *= s
        lrad_x *= s
        lrad_y *= s
        lang_x *= s
        lang_y *= s

    bs = tobj.shape[0]  # batch size
    loss = lbox + lobj + lcls + ldst + lrad_x + lrad_y + lang_x + lang_y
    g_rad = lrad_x + lrad_y
    g_ang = lang_x + lang_y
    return loss * bs, torch.cat((lbox, lobj, lcls, ldst, g_rad, g_ang, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h,dst)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch, tdst, trad_x, trad_y, tang_x, tang_y = [], [], [], [], [], [], [], [], []
    gain = torch.ones(12, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    v = model.hyp['v']
    v_sq = v ** 2

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain[0:10]
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 9].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        tdst.append(t[:, 6].float() + 0.5)  # distance
        # calculate angle coordinates on unit sphere
        gamma_2 = torch.remainder(t[:, 8], 180) * 0.034906585039  # 0.034906585039 = 2 pi/360 * 2
        tang_x.append(torch.cos(gamma_2) / 4 + 0.5)
        tang_y.append(torch.sin(gamma_2) / 4 + 0.5)

        # calculate radius coordinates on unit sphere
        radius_sign = torch.sign(t[:, 8] + 1e-6 - 180)
        r = t[:, 7] * radius_sign
        alpha_2 = torch.arccos(r / torch.sqrt(r ** 2 + v_sq)) * 2
        trad_x.append(torch.cos(alpha_2) / 4 + 0.5)
        trad_y.append(torch.sin(alpha_2) / 4 + 0.5)

    return tcls, tbox, indices, anch, tdst, trad_x, trad_y, tang_x, tang_y


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True,
                 BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()

        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale / 2.0) / self.bin_count
        end = max - (self.scale / 2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        # print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
            pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
            pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (
            pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)

        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins)  # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result
