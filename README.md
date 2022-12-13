# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

## Installation for Inference on Windows using Anaconda

```
conda env create -f environment.yaml
```

## Training

Single GPU training:

```
python train.py --batch-size 64 --img 640 640 --data data/minutiae.yaml \
    --cfg models/yolov7-tiny-silu.yaml --weights 'weights/yolov7-tiny.pt' --device 0 \
    --name yolov7-aiba2 --hyp hyp.minutiae.s.yaml --epochs 120 --single-cls
```


## Inference to save results into text files on CPU

```
python detect.py --img-size 640 --source val_images --weights weights/aiba2.pt --device cpu --name yolov7-aiba2 --save-txt --conf-thres 0.02 --iou-thres 0.4 --v 100 --save-conf --box-size 0.08
```

## Inference to save results into text files when CUDA is available

```
python detect.py --img-size 640 --source val_images --weights weights/aiba2.pt --device 0 --name yolov7-aiba2 --save-txt --conf-thres 0.02 --iou-thres 0.4 --v 100 --save-conf --box-size 0.08
```
