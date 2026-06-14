# Perceptron for OR operation

# Training data
inputs = [(0,0), (0,1), (1,0), (1,1)]
labels = [0, 1, 1, 1]

# Initialize weights and bias 
w1, w2 = 0.0, 0.0
bias = 0.0
learning_rate = 0.1
epochs = 10

# Step activation function 
def step(x):
    return 1 if x >= 0 else 0

# Training loop
for _ in range(epochs):
    for (x1, x2), label in zip(inputs, labels): 
        linear_output = w1*x1 + w2*x2 + bias
        prediction = step(linear_output)
        # Update rule
        error = label - prediction
        w1 += learning_rate * error * x1 
        w2 += learning_rate * error * x2 
        bias += learning_rate * error

# Testing
print("Trained weights:", w1, w2) 
print("Bias:", bias)

for x1, x2 in inputs:
    output = step(w1*x1 + w2*x2 + bias)
    print(f"{x1} OR {x2} = {output}")
