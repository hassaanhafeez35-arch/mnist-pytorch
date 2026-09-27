# MNIST Digit Classifier — PyTorch

A simple fully-connected neural network built with PyTorch to classify handwritten digits from the MNIST dataset.

## Overview

This project trains a basic feedforward neural network to recognize handwritten digits (0–9) from the classic MNIST dataset. It covers the full pipeline: loading data, building a model, training it, evaluating accuracy, and running inference on a single image.

## Model Architecture

A simple 3-layer fully connected network:
- Input: 784 (28x28 flattened image)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 64 neurons + ReLU
- Output: 10 neurons (one per digit class)

## Tools & Libraries

- Python
- PyTorch (`torch`, `torch.nn`, `torch.optim`)
- torchvision (for the MNIST dataset)
- Matplotlib (for visualizing predictions)

## Training Details

- Loss function: Cross-Entropy Loss
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 5
- Batch size: 64

## How to Run

```bash
pip install torch torchvision matplotlib
python mnist_project.py
```

The script will:
1. Download the MNIST dataset automatically (if not already present)
2. Train the model for 5 epochs, printing loss per epoch
3. Evaluate accuracy on the test set
4. Save the trained model as `mnist_model.pth`
5. Display a sample test image with the model's prediction vs. the actual label

## Results

The model achieves solid accuracy on the MNIST test set after just 5 epochs of training on a simple feedforward architecture (no convolutions).

## What I Learned

- How to build a basic neural network using `nn.Module` and `nn.Sequential`
- Setting up a full PyTorch training loop (forward pass, loss calculation, backpropagation, optimizer step)
- Evaluating a model using `torch.no_grad()` and computing accuracy
- Saving and loading model weights
- Running inference on individual samples and visualizing predictions with Matplotlib