# Depth and Object Detection 

## Overview
This project aims to assist blind people by providing them with information about the depth and detected objects in their surroundings. We have developed a system that utilizes computer vision techniques to calculate depth and detect objects in real-time. The information is then converted from text to speech for easy understanding by blind individuals.
## Calibration

To ensure accurate depth calculations, we first calibrate the camera. This is done by capturing images of various patterns and using them to obtain the camera coefficients. These coefficients are later used in the depth calculation process. Use [my_phone_chess](https://github.com/abhisharma2408/StereoVision/tree/main/my_phone_chess) folder for calibration with the file [calibration.py](https://github.com/abhisharma2408/StereoVision/blob/main/calibration.py) 
## Depth Calculation
Using binocular geometry, we calculate the disparity matrix. This matrix allows us to determine the distance of objects from the camera. By analyzing the disparity values, we can estimate the depth of the scene using focal plength of the camera used.
## Object Detection
To detect objects, we utilize a deep neural network (DNN) module. We train the DNN using the COCO dataset, and [haarcascade_frontface_default](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) which contains a wide range of object categories. The trained model is then used to detect objects in real-time.
## Text-to-Speech Conversion
Once the depth and object information is obtained, we convert it from text to speech using pyttsx3 library This allows blind individuals to receive the information audibly, enabling them to navigate their surroundings more effectively.
## Usage
To use this system, follow the steps below:

1. Calibrate the camera by capturing images of various patterns.
2. Calculate the depth and detect objects in real-time using the provided software.
3. Convert the obtained information from text to speech for easy understanding.

## Acknowledgements
- Dataset: [COCO Dataset](https://cocodataset.org/)
## Authors

- [Amaan Zaidi](https://github.com/amaanz)
- [Abhishek Sharma](https://github.com/abhisharma2408)

