# Real-Time Mask Detection
[colab notebook](https://colab.research.google.com/github/Ali-Noghabi/face-mask-detection/blob/main/mask_detection.ipynb)

## Table of Contents

1. [Introduction](#introduction)
2. [Project Setup](#project-setup)
3. [Detect and Predict](#detect-and-predict)
4. [Running the Project](#running-the-project)
5. [Results](#results)
6. [References](#references)

## Introduction

This project involves real-time mask detection using a pre-trained neural network. It can detect faces in video streams or webcam feeds and classify whether the person is wearing a mask or not. The project leverages deep learning techniques and computer vision to provide accurate and real-time mask detection.

## Project Setup

To run this project, you need to set up your environment and install the required libraries.

### Libraries Used

- **TensorFlow**: For loading and running the mask detection model.
- **OpenCV**: For real-time video processing and face detection.
- **imutils**: For basic image processing functions.
- **NumPy**: For numerical operations.

### Installation

To install the necessary libraries, run:

```bash
pip install tensorflow opencv-python imutils numpy
```

### Directory Structure

- `face_detector/deploy.prototxt`: Configuration file for the face detection model.
- `face_detector/res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained weights for the face detection model.
- `mask_detector/mask_detector_model.keras`: Pre-trained mask detector model.
- `mask_detection.py`: Main script to run the mask detection.

## Detect and Predict

The core functionality of the project involves detecting faces in a video frame and predicting whether a mask is worn. This is achieved using the following functions:

### Face Detection

Faces are detected using a pre-trained Caffe model. The model configuration is loaded from `deploy.prototxt` and the weights are loaded from `res10_300x300_ssd_iter_140000.caffemodel`.

```python
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
```

### Mask Detection

The [mask detection model](mask_detector\README.md) is a pre-trained Keras model loaded from `mask_detector_model.keras`. The model predicts the presence of a mask in detected faces.

```python
maskNet = load_model("mask_detector/mask_detector_model.keras")
```

### Detection and Prediction Function

This function processes each video frame, detects faces, and predicts mask usage.

```python
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Implementation of face detection and mask prediction
```

## Running the Project

To run the project, you can process either a video file or use a live webcam feed.

### Processing a Video File

Ensure that the video file is in the same directory as the script or provide the full path. Use the following command to run the script:

```bash
python mask_detection.py --video video.mp4
```

### Using the Webcam

To use the webcam for real-time mask detection, simply run the script without any arguments:

```bash
python mask_detection.py
```

Press 'q' to exit the video display.

## Results

The project will display the video feed with bounding boxes around detected faces. Each box will have a label indicating whether a mask is detected or not, along with the confidence level.

Example output:

![Mask](with_mask.png)
![NoMask](without_mask.png)

## References

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [imutils](https://github.com/jrosebr1/imutils)
- [NumPy](https://numpy.org/)
