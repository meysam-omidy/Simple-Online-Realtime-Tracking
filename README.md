# About This Repository

This repository contains the implementation of Simple Online Realtime Tracking. It uses kalman filter as its core localizer of objects and calculates similarity of objects based on the intersection over union (iou) between their bounding boxes. The state of kalman filter contains x,y,s,r,x',y',s'. x,y are the location of bounding box center in 2d space, s is the area, r is the aspect ratio of the bounding box and the rest are their velocities except for r since its velocity is assumed to be constant. It perform association using hungarian algorithm . [link to the paper](https://arxiv.org/abs/1602.00763)

# Differences from main implementation

* Added metrics dictionary to the main class and also a funcion for calculating known metrics of multiple object tracking
* each trajectory also gets an id to help in calculation of metrics related to ids
* gave the model the ability to get a detector as input and perform detection in place of getting detected boxes as input

# use case

if a base detector exsits:
```
# define the detector class which has a forward function that gets an image as input and returns bounding boxes with confidences
class Detector():
    def forward(image):
      pass

detector = Detector()
sort = SORT(detector, max_age, iou_threshold_detection, iou_threshold_track)
```

