import numpy as np

# prices = np.array([100, 250, 300, 150])

# tax = 10

# final_prices = prices + tax

# print("Prices:", prices)
# print("Tax:", tax)
# print("Final prices:", final_prices)

X = np.array([
    [100, 2],
    [250, 5],
    [300, 8],
    [150, 3],
])

mean = np.array([200, 4.5])

X_centered = X - mean

print("Original X:")
print(X)

print("Mean:", mean)

print("Centered X:")
print(X_centered)