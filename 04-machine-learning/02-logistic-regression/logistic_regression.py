import numpy as np


class LogisticRegression:
  def __init__(self, learning_rate = 0.01, epochs = 1000):
    self.learning_rate = learning_rate
    self.epochs = epochs

    self.w = None
    self.b = 0

    self.loss_history = []

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def predict_proba(self, X):
    z = X @ self.w + self.b
    return self.sigmoid(z)

  def log_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

  def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):

            # Forward pass
            y_pred = self.predict_proba(X)

            # Calculate loss
            loss = self.log_loss(y, y_pred)
            self.loss_history.append(loss)

            # Calculate gradients
            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Gradient descent
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

  def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)