import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# dot_product = np.dot(a, b)

# print("Dot Product: ", dot_product)

A = np.array([
  [1, 2],
  [3, 4],
])

B = np.array([
  [5, 6],
  [7, 8],
])

C = A @ B

print("A:")
print(A)

print("B:")
print(B)

print("C:")
print(C)