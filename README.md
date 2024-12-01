# About This Repository

This repository contains the implementation of Simple Online Realtime Tracking. It uses kalman filter as its core localizer of objects and calculates similarity of objects based on the intersection over union (iou) between their bounding boxes. The state of kalman filter contains x,y,s,r,x',y',s'. x,y are the location of bounding box center in 2d space, s is the area, r is the aspect ratio of the bounding box and the rest are their velocities except for r since its velocity is assumed to be constant. It perform association using hungarian algorithm . 

# Differences from main implementation

. Added metrics 
