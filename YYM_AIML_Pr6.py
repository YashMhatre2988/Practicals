# Perceptron using Stochastic Gradient Descent (SGD)

import numpy as np

# Training dataset for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target outputs
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

# Learning rate
lr = 0.1

# Number of epochs
epochs = 10

# Activation function
def activation(z):
    return 1 if z >= 0 else 0

# Training using SGD
for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}")

    for i in range(len(X)):

        # Linear combination
        z = np.dot(X[i], weights) + bias

        # Prediction
        y_pred = activation(z)

        # Error
        error = y[i] - y_pred

        # Weight update
        weights = weights + lr * error * X[i]

        # Bias update
        bias = bias + lr * error

        print(f"Input: {X[i]}, Predicted: {y_pred}, Error: {error}")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Testing
print("\nTesting Perceptron")

for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    prediction = activation(z)

    print(f"Input: {X[i]} -> Output: {prediction}")