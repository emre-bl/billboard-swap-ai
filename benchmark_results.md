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
| YOLOv8+SAM  | Test    |       31 |          43 | 0.407 | 0.512 | 0.36      | 0.284     |
| Mask R-CNN  | Test    |       31 |          43 | 0.867 | 0.605 | 0.626     | 0.531     |