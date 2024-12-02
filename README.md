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

## Overview
This repository contains the implementation of the SORT (Simple Online and Realtime Tracking) algorithm, inspired by the paper "SIMPLEONLINEANDREALTIMETRACKING" by Alex Bewley et al. The SORT algorithm is designed for efficient multiple object tracking in real-time applications, leveraging high-quality object detection and a straightforward tracking approach.

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
- Other libraries as specified in the `requirements.txt`

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/sort.git
cd sort
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To run the SORT tracker, use the following command:
```bash
python main.py --input <path_to_input_video> --output <path_to_output_video>
```
Replace `<path_to_input_video>` with the path to your input video file and `<path_to_output_video>` with the desired output file path.

## Evaluation
The tracking performance can be evaluated using the provided scripts. Make sure to have the MOT benchmark dataset available for testing.

## Contributing
Contributions to improve the implementation are welcome. Please fork the repository, make your changes, and submit a pull request.

## Citation
If you use this code in your research, please cite the original paper:
```
@article{bewley2016simple,
  title={SIMPLEONLINEANDREALTIMETRACKING},
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  journal={arXiv preprint arXiv:1602.00763},
  year={2016}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
