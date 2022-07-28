import os

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import argparse
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression

from glob import glob

COLORS = np.random.uniform(0, 255, size=(80, 3))

parser = argparse.ArgumentParser(prog='eigencam_r.py')
parser.add_argument('--input', type=str, default='inference/images/IMG_6479.JPG', help='path to input image')
parser.add_argument('--weights', type=str, required=True, help='path to weights')
parser.add_argument('--save-path', type=str, default='inference/output/', help='path to save output')
opt = parser.parse_args()


def parse_detections(pred):
    boxes, colors, names = [], [], []
    for det in pred:
        for *xyxy, _, cls in det:
            xyxy = torch.tensor(xyxy).view(1, 4)
            xmin = int(xyxy[0, 0])
            ymin = int(xyxy[0, 1])
            xmax = int(xyxy[0, 2])
            ymax = int(xyxy[0, 3])
            category = int(cls)
            color = COLORS[category]

            boxes.append((xmin, ymin, xmax, ymax))
            colors.append(color)
            names.append(str(cls))
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


transform = transforms.ToTensor()
weights = opt.weights
device = 'cpu'
imgsz = 640
half = device != 'cpu'
basename = os.path.basename(opt.input)

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16
model.eval()
model.cpu()
target_layers = [model.model[-5], model.model[-4], model.model[-2], model.model[-6]]
print(target_layers)

cam = EigenCAM(model, target_layers, use_cuda=False)


img = np.array(Image.open(opt.input))
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()

tensor = transform(img).unsqueeze(0)
img = np.float32(img) / 255

# cam on whole image
grayscale_cam = -np.square(cam(tensor, eigen_smooth=True, aug_smooth=True)[0, :, :])
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image).save(os.path.join(opt.save_path, basename))

# cam constrains on bbox
pred = model(tensor)
pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.4)

boxes, colors, names = parse_detections(pred)
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
Image.fromarray(renormalized_cam_image).save(os.path.join(opt.save_path, basename.replace('.', '_2.')))
