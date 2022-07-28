Usage of eigencam Script

```
python eigencam_r.py --input inference/images/IMG_6479.JPG --weights weights/dd_screws.pt --save-path inference/output/ --eigen-smooth --aug-smooth
```
The Script generates two output images in the --save-path directory. One showing the Eigencam heatmap for the entire image
and a second image showing the heatmap constrained for the detections where the heatmap is normalized individually for each box.

The options --eigen-smooth and --aug-smooth lead to better results but increase the runtime.