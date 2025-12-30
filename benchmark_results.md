# Benchmark Results

## Table 1: Billboard Object Detection (bounding box) accuracy

| Model                   | Stage   |   Images |   Instances |     P |     R | AP50      | AP50-95   |
|:------------------------|:--------|---------:|------------:|------:|------:|:----------|:----------|
| YOLOv8n                 | Test    |      211 |         509 | 0.722 | 0.643 | 0.723     | 0.473     |
| YOLOv8s                 | Test    |      211 |         509 | 0.787 | 0.612 | 0.731     | 0.479     |
| YOLOv8m                 | Test    |      211 |         509 | 0.818 | 0.6   | **0.736** | **0.48**  |
| YOLO-Worldv2(Zero-shot) | Test    |      211 |         509 | 0.33  | 0.69  | 0.598     | 0.408     |


## Table 2: Billboard Segmentation (precise mask) accuracy

| Model       | Stage   |   Images |   Instances |     P |     R | AP50      | AP50-95   |
|:------------|:--------|---------:|------------:|------:|------:|:----------|:----------|
| YOLOv8n-seg | Test    |       31 |          43 | 0.786 | 0.684 | **0.672** | **0.542** |
| YOLOv8s-seg | Test    |       31 |          43 | 0.819 | 0.512 | 0.586     | 0.442     |
| YOLOv8m-seg | Test    |       31 |          43 | 0.693 | 0.605 | 0.636     | 0.462     |
| YOLOv8+SAM  | Test    |       31 |          43 | 0.307 | 0.535 | 0.307     | 0.264     |
| Mask R-CNN  | Test    |       31 |          43 | 0.397 | 0.721 | 0.397     | 0.347     |


## Table 3: Tracking Algorithm Performance (YOLOv8m-det + SAM2)

Evaluation setup: YOLOv8m detection → SAM2 segmentation at keyframes (interval=30), trackers propagate masks between keyframes.

| Tracker                    | Mean IoU  | Median IoU | Success@0.5 | Success@0.7 | Failures   | FPS     |
|:---------------------------|:----------|:-----------|:------------|:------------|:-----------|:--------|
| PlanarKalmanTracker        | **0.915** | 0.939      | 99.7%       | 96.5%       | 1/379      | 283.7   |
| FeatureHomographyTracker   | 0.907     | 0.953      | 96.2%       | 92.7%       | 14/379     | 2.9     |
| HybridFlowTracker          | 0.906     | 0.963      | 97.6%       | 88.6%       | 9/379      | 35.7    |
| SAM2Tracker                | 0.859     | 0.843      | 96.2%       | 91.3%       | 14/379     | 8.6     |
| ECCHomographyTracker       | 0.684     | 0.651      | 79.8%       | 38.0%       | 76/379     | 1.9     |
| AdaptiveOpticalFlowTracker | 0.577     | 0.879      | 63.1%       | 59.6%       | 143/379    | 32.8    |