import numpy as np

X = np.array([
    [100, 2],
    [250, 5],
    [300, 8],
    [150, 3]
])

# print("X:")
# print(X)

# print("Shape:", X.shape)
# print("Dimensions:", X.ndim)
# print("Size:", X.size)
# print("Data type:", X.dtype)

# print(X[0])
# print(X[0][1])

print(X[1:3])

print(X[1:3, 1:2])