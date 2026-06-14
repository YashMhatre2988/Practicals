import numpy as np

# ─────────────────────────────────────────────
# Input dataset for AND operation
# ─────────────────────────────────────────────
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target outputs
T = np.array([0, 0, 0, 1])

# ─────────────────────────────────────────────
# Initialize weights and bias
# ─────────────────────────────────────────────
w = np.zeros(2)
b = 0

# Learning rate
alpha = 0.1

# Number of epochs
epochs = 10

print("Initial Weights:", w)
print("Initial Bias:", b)

# ─────────────────────────────────────────────
# Training Process
# ─────────────────────────────────────────────
for epoch in range(epochs):

    print("\nEpoch", epoch + 1)

    for i in range(len(X)):

        x = X[i]
        target = T[i]

        # Net input
        yin = b + np.dot(w, x)

        # Error
        error = target - yin

        # Weight update
        w = w + alpha * error * x

        # Bias update
        b = b + alpha * error

        print(f"Input: {x}, Target: {target}, Output: {yin:.2f}")
        print("Updated Weights:", w)
        print("Updated Bias:", round(b, 2))

# ─────────────────────────────────────────────
# Testing Phase
# ─────────────────────────────────────────────
print("\nTesting Adaline for AND Operation")

for i in range(len(X)):

    yin = b + np.dot(w, X[i])

    # Activation function
    output = 1 if yin >= 0.5 else 0

    print(f"Input: {X[i]} -> Output: {output}")