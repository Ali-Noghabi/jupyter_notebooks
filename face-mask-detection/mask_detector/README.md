# Mask Detector Model Training
[colab notebook](https://colab.research.google.com/github/Ali-Noghabi/face-mask-detection/blob/main/mask_detector/train_mask_detector.ipynb)
## Table of Contents

1. [Introduction](#introduction)
2. [Project Setup](#project-setup)
3. [Model Training](#model-training)
4. [Running the Project](#running-the-project)
5. [Results](#results)
6. [References](#references)

## Introduction

This project involves training a custom mask detector model using deep learning techniques. The trained model can then be used to detect masks in real-time video streams or webcam feeds. The project leverages TensorFlow for training the model and OpenCV for real-time video processing.

## Project Setup

To run this project, you need to set up your environment and install the required libraries.

### Libraries Used

- **TensorFlow**: For training the mask detection model.
- **OpenCV**: For real-time video processing and face detection.
- **imutils**: For basic image processing functions.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting training results.

### Installation

To install the necessary libraries, run:

```bash
pip install tensorflow opencv-python imutils numpy matplotlib
```

### Directory Structure

- `data/with_mask`: Directory containing images of people with masks.
- `data/without_mask`: Directory containing images of people without masks.
- `train_mask_detector.py`: Script to train the mask detector model.

[images source](https://github.com/prajnasb/observations/tree/master)

## Model Training

To train your own mask detector model, follow these steps:

1. **Prepare Dataset**: Collect and organize a dataset with images of faces with and without masks. Organize the dataset into two directories: `with_mask` and `without_mask`.

2. **Train Model**: Use the provided `train_mask_detector.py` script to train the model. This script uses a deep learning framework to train a mask detection model on your dataset.

### Training Script

The training script `train_mask_detector.py` performs the following steps:

- Loads and preprocesses the images from the dataset.
- Splits the data into training and testing sets.
- Defines a convolutional neural network (CNN) for mask detection.
- Trains the model using data augmentation.
- Saves the trained model.

#### Code Explanation

- **Data Loading and Preprocessing**: Images are loaded, converted to RGB, resized to 224x224 pixels, and stored in arrays.

```python
data = []
labels = []

for category in categories:
    path = os.path.join(dataset_path, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (224, 224))
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
```

- **Label Encoding and Data Augmentation**: Labels are encoded, and data augmentation is set up to improve model generalization.

```python
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = preprocess_input(data)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
```

- **Model Definition and Training**: A CNN is defined, compiled, and trained using the preprocessed data.

```python
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY),
    epochs=20
)

model.save("mask_detector_model.keras")
```

## Running the Project

### Training the Model

To train the mask detection model, run the following command:

```bash
python train_mask_detector.py
```

This command will train the model on the provided dataset and save the trained model as `mask_detector_model.keras`.

## Results

After training, the model will be evaluated on the test set, and the training history will be plotted. You can use these plots to understand the model's performance and adjust the training parameters accordingly.

[pre-trained model](https://drive.google.com/file/d/1mxx_IMhhHuqAIt3RWqpHqpFkjhMf3uzF/view?usp=sharing)
## References

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [imutils](https://github.com/jrosebr1/imutils)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
