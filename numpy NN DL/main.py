import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Normalize the input data
X = X / 16.0

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Transpose for easier matrix operations
X_train = X_train.T
X_test = X_test.T
Y_train = y_train.T
Y_test = y_test.T

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1. / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Forward propagation
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A1, A2

# Cross-entropy loss
def compute_loss(A2, Y):
    m = Y.shape[1]
    log_probs = np.multiply(np.log(A2), Y)
    cost = -np.sum(log_probs) / m
    return np.squeeze(cost)

# Sigmoid derivative
def sigmoid_derivative(Z):
    s = 1 / (1 + np.exp(-Z))
    return s * (1 - s)

# Backward propagation
def backward_propagation(X, Y, W1, b1, W2, b2, A1, A2):
    m = X.shape[1]
    
    # Output layer gradients
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # Hidden layer gradients
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(np.dot(W1, X) + b1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Initialize the neural network parameters
input_size = X_train.shape[0]
hidden_size = 256  # Increased from 128
output_size = 10

W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

# Training loop
learning_rate = 0.1  # Increased from 0.01
num_iterations = 1000

for i in range(num_iterations):
    # Forward propagation
    A1, A2 = forward_propagation(X_train, W1, b1, W2, b2)
    
    # Compute loss
    cost = compute_loss(A2, Y_train)
    
    # Backward propagation
    dW1, db1, dW2, db2 = backward_propagation(X_train, Y_train, W1, b1, W2, b2, A1, A2)
    
    # Update parameters
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost}")

print("Training complete.")

# Function to make predictions
def predict(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Make predictions on the training set
predictions_train = predict(X_train, W1, b1, W2, b2)

# Convert one-hot encoded Y_train back to labels
true_labels_train = np.argmax(Y_train, axis=0)

# Compute accuracy
accuracy_train = accuracy_score(true_labels_train, predictions_train)
print(f"Training Accuracy: {accuracy_train * 100:.2f}%")

# Compute confusion matrix
conf_matrix_train = confusion_matrix(true_labels_train, predictions_train)
print("Confusion Matrix (Training Set):")
print(conf_matrix_train)

# Make predictions on the test set
predictions_test = predict(X_test, W1, b1, W2, b2)

# Convert one-hot encoded Y_test back to labels
true_labels_test = np.argmax(Y_test, axis=0)

# Compute accuracy on test set
accuracy_test = accuracy_score(true_labels_test, predictions_test)
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

# Compute confusion matrix for test set
conf_matrix_test = confusion_matrix(true_labels_test, predictions_test)
print("Confusion Matrix (Test Set):")
print(conf_matrix_test)
