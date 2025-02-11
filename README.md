## Overview
This repository contains the implementation of the SORT (Simple Online and Realtime Tracking) algorithm, inspired by the paper "SIMPLE ONLINE AND REALTIME TRACKING" by Alex Bewley et al ([link to the paper](https://arxiv.org/abs/1602.00763)). The SORT algorithm is designed for efficient multiple object tracking in real-time applications, leveraging high-quality object detection and a straightforward tracking approach. 

## Key Features
- **Real-time Performance**: Achieves a tracking update rate of 260Hz, significantly faster than state-of-the-art trackers.
- **Simplicity**: Utilizes basic techniques like the Kalman filter and Hungarian algorithm for motion prediction and data association.
- **Detection Quality**: Incorporates advanced CNN-based detection methods for improved tracking accuracy.

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- OpenCV
- TensorFlow or PyTorch (depending on the detection model)

## Differences From The Original Implementation

* Added metrics dictionary to the main class and also a funcion for calculating known metrics of multiple object tracking
* Each trajectory also gets an id to help in the calculation of metrics related to ids
* Gave the model the ability to get a detector as input and perform detection in place of getting detected bounding boxes as input

## Usage
if a base detector exsits:
```
# define the detector class which has a forward function that gets an image as input and
# returns bounding boxes alon with their confidences
class Detector():
    def forward(image):
        return bounding_boxes

detector = Detector()
sort = SORT(detector, max_age, iou_threshold_detection, iou_threshold_track)
for frame in frames:
    sort.update(image, gt_dets)
```
otherwise:
```
sort = SORT(None, max_age, iou_threshold_detection, iou_threshold_track)
for frame in frames:
    sort.update(dets, gt_dets)
```

## Results
results on the det.txt and gt.txt without specifying detector and on the first 60 frames
```
Total time: 0.699s
FPS: 1197.658  
Metrics:       
    MOTA: 0.442
    MOTP: 0.837
    IDF1: 0.587
    HOTA: 0.565
    AssA: 0.704
    DetA: 0.454
```
