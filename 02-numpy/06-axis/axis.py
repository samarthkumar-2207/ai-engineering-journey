import numpy as np

X = np.array([
    [100, 2],
    [250, 5],
    [300, 8],
    [150, 3],
])

print("X:")
print(X)

print("Column means:", np.mean(X, axis=0))
print("Row means:", np.mean(X, axis=1))

print("Column standard deviation:", np.std(X, axis=0))
print("Column minimum:", np.min(X, axis=0))
print("Column maximum:", np.max(X, axis=0))
