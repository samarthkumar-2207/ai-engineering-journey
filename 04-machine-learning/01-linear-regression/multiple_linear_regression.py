import numpy as np


class MultipleLinearRegression:

    def __init__(self, learning_rate=0.0001, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = 0.0

        self.loss_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize one weight per feature
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):

            # 1. Predictions
            y_pred = self.predict(X)

            # 2. Loss
            loss = self.mse(y, y_pred)
            self.loss_history.append(loss)

            # 3. Error
            error = y_pred - y

            # 4. Gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # 5. Update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ], dtype=float)

    y = np.array([
        10,
        15,
        20,
        25,
        30
    ], dtype=float)

    model = MultipleLinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    print("Weights:", model.w)
    print("Bias:", model.b)

    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("Final MSE:", model.loss_history[-1])