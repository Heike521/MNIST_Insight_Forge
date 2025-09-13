# torch-digits 🔢🔥

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**torch-digits** is a PyTorch-based project for classifying handwritten digits from the MNIST dataset.  
It demonstrates a clean training pipeline with reproducibility, evaluation, and visualization features.

---

## 📌 Features

- Feedforward Neural Network (2 hidden layers, BatchNorm, Dropout)
- Training, validation and test with accuracy evaluation
- Visualizations: Loss/Accuracy curves, Confusion Matrix, ROC/PR, misclassified images
- Metrics export as CSV, model checkpoints as `.pth`
- Logging & reproducible seeds
- CLI for hyperparameters (epochs, batch size, learning rate)

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<YOUR_USERNAME>/torch-digits.git
cd torch-digits
pip install -r requirements.txt

---

## 📂 Project Structure

torch-digits/
│
├── src/
│   └── NN_PyTorch.py      # main training script
│
├── models/                # saved models & checkpoints
├── tests/                 # unit tests
│   └── test_nn_pytorch.py
│
├── requirements.txt       # dependencies
├── README.md              # project documentation
├── LICENSE                # license file
└── .gitignore             # ignored files for git

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
