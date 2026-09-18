import numpy as np

from multiple_linear_regression import MultipleLinearRegression


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1

        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


X = np.array([
    [1, 1000],
    [2, 2000],
    [3, 3000],
    [4, 4000],
    [5, 5000]
], dtype=float)

y = np.array([
    10,
    20,
    30,
    40,
    50
], dtype=float)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original X:")
print(X)

print("\nScaled X:")
print(X_scaled)

print("\nMean:")
print(X_scaled.mean(axis=0))

print("\nStd:")
print(X_scaled.std(axis=0))

model = MultipleLinearRegression(
    learning_rate=0.01,
    epochs=5000
)

model.fit(X_scaled, y)

print("\nWeights:")
print(model.w)

print("\nBias:")
print(model.b)

print("\nPredictions:")
print(model.predict(X_scaled))

print("\nFinal MSE:")
print(model.loss_history[-1])