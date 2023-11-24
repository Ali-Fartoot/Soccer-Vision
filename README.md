# Team Prediction & Player Detection using YOLOv8
Overview
This repository contains a Python program utilizing YOLOv8 from Ultralytics to perform team prediction and player detection in a custom dataset. The program achieves an mAP of 38 at 50 IoU (Intersection over Union) and maintains an average FPS (Frames Per Second) range of 18-20.

![Scene](https://github.com/Ali-Fartout/Soccer-Vision/blob/master/Screenshot%202023-11-24%20135143.png)

# Features
* Team Prediction: Identifies and predicts teams based on specified criteria.
* Player Detection: Detects and localizes players within a given frame or set of frames.
* Custom Dataset: Trained and tested on a custom dataset for specific use cases.
* YOLOv8 Integration: Utilizes the YOLOv8 model architecture for efficient object detection.
* Performance Metrics: Achieves an mAP of 38 at 50 IoU on the custom dataset.
* FPS Monitoring: Sustains an average FPS range of 18-20 for real-time or batch processing.

# Requirements
Python 3.x
PyTorch
Ultralytics YOLOv8
Custom Dataset (Refer to dataset preparation section

# Usage
1. git clone https://github.com/Ali-Fartout/Soccer-Vision.git
2. run capture.py
