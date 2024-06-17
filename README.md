# numpy-NN-DL
Handwritten Digit Classification using a Neural Network
This project implements a simple neural network from scratch to classify handwritten digits from the popular MNIST dataset. The model is trained and tested using the Scikit-learn load_digits dataset, which is a smaller version of the MNIST dataset containing 8x8 pixel images.

Project Overview
The project includes the following key components:

Data Loading and Preprocessing
Neural Network Implementation
Training and Evaluation
Results Visualization
Data Loading and Preprocessing
The digits dataset is loaded using sklearn.datasets.load_digits(). The images are normalized by dividing by 16.0, and the labels are one-hot encoded using OneHotEncoder.

Neural Network Implementation
The neural network is implemented from scratch with the following features:

Architecture: Input layer, one hidden layer (256 neurons), and an output layer (10 neurons for digit classification).
Activation Functions: Sigmoid for the hidden layer and Softmax for the output layer.
Loss Function: Cross-entropy loss.
Optimization: Gradient descent.
Training and Evaluation
The dataset is split into training and test sets. The model is trained using forward and backward propagation over 1000 iterations with a learning rate of 0.1. The accuracy and confusion matrix are calculated for both the training and test sets.

Installation
To run this project, you need to have Python and the following libraries installed:

NumPy
Scikit-learn
Matplotlib
You can install these packages using pip:
