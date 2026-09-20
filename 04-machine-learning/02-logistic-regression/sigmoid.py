import numpy as np

def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
  s = sigmoid(z)
  return s * (1 - s)


values = np.array([-5, -2, 0 , 2, 5], dtype=float)

# probabilities = sigmoid(values)

# print("Input:")
# print(values)

# print("\nSigmoid:")
# print(probabilities)

print("Sigmoid:")
print(sigmoid(values))

print("\nSigmoid derivative:")
print(sigmoid_derivative(values))