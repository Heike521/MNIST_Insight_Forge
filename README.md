# 🧠 Neural Network for Digit Recognition

This project implements an artificial neural network (ANN) that recognizes handwritten digits. It uses the MNIST dataset and is built entirely with NumPy, PIL, and scikit-learn—without high-level frameworks like TensorFlow or PyTorch.
The goal is to gain a solid understanding of forward and backpropagation, weight updates, and visualization of neural network processes.

---

## 📦 Features

- Standalone neural network (feedforward network)
- Training over multiple epochs with sigmoid activation and MSE loss
- Manual backpropagation procedure
- Visualization of:
  - Loss and accuracy curves
  - Weight matrices
  - Confusion matrix (as table & heatmap)
  - Heatmap of misclassifications
  - Misclassified examples
- Complete saving of all results as .png and .csv
- Generation of a structured project report with training statistics

---

## 🧠 Model Architecture

| Layer            | Number of nodes  | Description                     |
|------------------|------------------|---------------------------------|
| Input layer      | 784              | 28 × 28 pixels of MNIST image   |
| Hidden layer     | 300              | Fully connected                 |
| Output layer     | 10               | Digits 0–9                      |

- Activation function: **Sigmoid**
- Loss function: **Mean Squared Error (MSE)**
- Optimization: **Stochastic Gradient Descent (SGD)**

---

## 🔧 Prerequisites

```bash
pip install numpy pillow torchvision scikit-learn

---

## 👩‍💻 Author & Project Status

**Author:** Heike Fasold  
**Project:** Entwicklung eines neuronalen Netzes zur Ziffernerkennung  
**Status as of:** 26. Juli 2025  

The aim of this project was to understand and practically implement the mathematical foundations of neural networks in Python—without using prebuilt deep learning frameworks.
