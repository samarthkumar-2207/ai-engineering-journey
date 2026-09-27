import numpy as np

from sklearn.preprocessing import StandardScaler


X = np.array([
    [22, 35000],
    [29, 52000],
    [35, 68000],
    [41, 82000],
])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original:")
print(X)

print("\nScaled:")
print(X_scaled)

print("\nMean:")
print(X_scaled.mean(axis=0))

print("\nStandard deviation:")
print(X_scaled.std(axis=0))