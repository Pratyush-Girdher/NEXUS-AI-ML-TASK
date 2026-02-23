# CNN From Scratch – MNIST Digit Classification

## Problem Statement

The objective of this project is to implement an image classification system using a Convolutional Neural Network (CNN) completely from scratch using Python and NumPy.

High-level deep learning frameworks such as TensorFlow, Keras, and PyTorch were not used. The implementation focuses on understanding convolution operations, pooling mechanisms, activation functions, and forward and backward propagation.

The trained model is deployed using a Streamlit interface that allows users to upload an image and view the predicted digit.

---

## Dataset Description

This project uses the **MNIST Handwritten Digits Dataset**.

- Total Images: 70,000 grayscale images
- Image Size: 28 × 28 pixels
- Classes: Digits (0–9)

Dataset Source:

OpenML MNIST Dataset (loaded using Scikit-learn).

Each image is normalized between 0 and 1 before training.

---

## Model Architecture

The CNN architecture used:

Input Image (28 × 28 × 1)

↓

Convolution Layer
- Filter Size: 3 × 3
- Number of Filters: 8
- Output: 26 × 26 × 8

↓

ReLU Activation

↓

Max Pooling Layer
- Pool Size: 2 × 2
- Output: 13 × 13 × 8

↓

Flatten Layer
- 1352 Features

↓

Fully Connected Layer
- Output Nodes: 10

↓

Softmax Layer
- Probability distribution across digits (0–9)

---

## Implementation Details

The model was implemented using:

- Python
- NumPy
- Pandas
- Scikit-learn

### Components Implemented From Scratch

1. Manual Convolution Operation
2. ReLU Activation Function
3. Max Pooling Layer
4. Fully Connected Layer
5. Softmax Function
6. Cross Entropy Loss
7. Forward Propagation
8. Backpropagation (Fully Connected Layer)

No deep learning frameworks were used.

---

## Training Process

Training includes:

- Batch Processing
- Cross Entropy Loss Optimization
- Gradient Updates using Backpropagation

The convolution filters remain fixed while the fully connected layer learns classification boundaries.

---

## Evaluation Metrics

Model performance is evaluated using:

- Cross Entropy Loss
- Classification Accuracy

Evaluation is performed using unseen MNIST test samples.

Example Output:

Loss ≈ 2.2 → 1.5 (reducing during training)

Accuracy improves across epochs.

---

## Deployment (Streamlit Interface)

A Streamlit-based web interface is implemented.

Features:

- Upload image (.png / .jpg)
- Automatic grayscale conversion
- Image resizing to 28 × 28
- Model prediction display.

---

## Project Structure
