# Federated Learning with MNIST Dataset

This project demonstrates the implementation of Federated Learning using the MNIST dataset in Google Colab. It simulates multiple clients (robots) that collaboratively train a global model without sharing their raw data.

[![Jupyter Nootbook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ali-Noghabi/FL-MNIST/blob/main/Federated_Learning_with_MNIST_Dataset.ipynb)
## Table of Contents

1. [Introduction](#introduction)
2. [Project Setup](#project-setup)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Federated Learning Steps](#federated-learning-steps)
5. [Running the Project](#running-the-project)
6. [Results](#results)
8. [References](#references)

## Introduction

**Federated Learning (FL)** is a machine learning paradigm where multiple devices (clients) collaboratively train a model without sharing their raw data. Instead, each device trains a local model on its data and shares only the model parameters with a central server, which aggregates these parameters to update a global model. This approach enhances data privacy and reduces the need for massive data transfers.

In this project, we use the MNIST dataset, which contains images of handwritten digits, to simulate federated learning across multiple clients. Each client trains a neural network locally on its subset of the data, and the central server aggregates the models.

## Project Setup

### Install Necessary Libraries

```bash
pip install tensorflow tensorflow-federated numpy matplotlib
```

### Import Libraries and Prepare Dataset

```python
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### Simulate Multiple Clients

```python
num_clients = 10
client_data = []

# Split the training data into `num_clients` parts
for i in range(num_clients):
    start_idx = i * len(x_train) // num_clients
    end_idx = (i + 1) * len(x_train) // num_clients
    client_data.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))

# Convert client data to TensorFlow datasets
client_datasets = [
    tf.data.Dataset.from_tensor_slices((x, y)).batch(32) for x, y in client_data
]
```

## Neural Network Architecture

### Detailed Explanation of Layers

1. **Flatten Layer**
   - **Purpose:** Reshapes the input data from a multi-dimensional array to a one-dimensional array.
   - **Input Shape:** 28x28 (an image of a digit in the MNIST dataset).
   - **Output Shape:** 784 (28x28 = 784).

2. **Dense Layer (128 units)**
   - **Purpose:** Fully connected layer with ReLU activation.
   - **Units:** 128 neurons.
   - **Activation Function:** ReLU.

3. **Dropout Layer (20%)**
   - **Purpose:** Regularization technique to prevent overfitting.
   - **Rate:** 0.2 (20% of the input units are set to zero during each update).

4. **Dense Layer (10 units)**
   - **Purpose:** Output layer for classification with softmax activation.
   - **Units:** 10 neurons (one for each digit class).
   - **Activation Function:** Softmax.

### Define the Neural Network Model

```python
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

## Federated Learning Steps

### Local Training on Each Client

Each client trains a local model using its own subset of data.

```python
def client_update(model, dataset, epochs=1):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=epochs)
    return model.get_weights()
```

### Aggregate the Models to Form a Global Model

The central server aggregates the weights received from all clients to update the global model.

```python
def aggregate_weights(weights):
    new_weights = []
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.mean(np.array(weights_list_tuple), axis=0)
        )
    return new_weights
```

### Federated Training Loop

The federated training loop involves multiple rounds of training and aggregation. Here is a step-by-step breakdown:

1. **Initialize the Global Model:**
   - Start by creating a global model and obtaining its initial weights.

2. **Iterate for a Number of Rounds:**
   - For each round, perform the following steps:
     - **Local Training:** Each client trains its local model using its subset of the data and the current global weights. The trained local model's weights are collected.
     - **Model Aggregation:** The collected weights from all clients are averaged to form the new global weights. This aggregation ensures that the knowledge learned by each client is incorporated into the global model.
     - **Update Global Model:** The global model's weights are updated with the newly aggregated weights.
     - **Compile Global Model:** Ensure the global model is compiled before evaluating its performance.
     - **Evaluate Global Model:** Test the global model on the test data to monitor its performance after each round.

```python
global_model = create_keras_model()
global_weights = global_model.get_weights()

num_rounds = 10

for round_num in range(num_rounds):
    local_weights = []

    for client_dataset in client_datasets:
        local_model = create_keras_model()
        local_model.set_weights(global_weights)
        local_weights.append(client_update(local_model, client_dataset))

    global_weights = aggregate_weights(local_weights)
    global_model.set_weights(global_weights)
    
    # Compile the global model before evaluation
    global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Evaluate the global model
    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f'Round {round_num+1}, Loss: {loss}, Accuracy: {accuracy}')
```

#### Explanation of Combining Models

- **Local Training:** Each client trains a model on its local data. This process allows each client to learn from its unique dataset, which might have different characteristics compared to others.
- **Collect Weights:** After training, each client sends its model's weights (parameters) to the central server.
- **Aggregation:** The central server aggregates these weights by averaging them. This step is crucial as it combines the knowledge gained from all clients.
  - **Why Averaging?** Averaging helps to generalize the global model. Each client's model contributes equally, which helps the global model to learn a broader representation of the data.
- **Update Global Model:** The global model's weights are updated with the aggregated weights, thus incorporating the knowledge from all clients.
- **Compile and Evaluate:** The global model is compiled and evaluated on the test dataset to monitor its performance.

### Plot Test Images with Predictions

```python
# Predict using the global model
predictions = global_model.predict(x_test)

# Plot some test images with their predicted labels
num_images_to_plot = 10
plt.figure(figsize=(15, 5))

for i in range(num_images_to_plot):
    plt.subplot(1, num_images_to_plot, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')

plt.show()
```

## Results

After training the federated learning model, you can visualize the results by plotting some test images along with their predicted and true labels. 

![Model Results](result-plot.png)

## References

1. [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
2. [TensorFlow Federated](https://www.tensorflow.org/federated)
3. [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
