import numpy as np

class LinearRegression:
   def __init__(self, learning_rate = 0.01, epochs = 5000):
      self.learning_rate = learning_rate
      self.epochs = epochs

      self.w = 0.0
      self.b = 0.0

      self.loss_history = []

   def predict(self, X):
      return self.w * X + self.b

   def mse(self, y_true, y_pred):
      return np.mean((y_true - y_pred) ** 2)

   def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):

            # 1. Make predictions
            y_pred = self.predict(X)

            # 2. Calculate loss
            loss = self.mse(y, y_pred)
            self.loss_history.append(loss)

            # 3. Calculate gradients
            error = y_pred - y

            dw = (2 / n) * np.sum(X * error)
            db = (2 / n) * np.sum(error)

            # 4. Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self
   
if __name__ == "__main__":
    
  from linear_regression import LinearRegression

  X = np.array([1, 2, 3, 4, 5], dtype=float)
  y = np.array([35, 45, 55, 65, 75], dtype=float)


  learning_rates = [0.001, 0.01, 0.1]

  for lr in learning_rates:

      model = LinearRegression(
          learning_rate=lr,
          epochs=1000
      )

      model.fit(X, y)

      print(f"\nLearning rate: {lr}")
      print(f"Weight: {model.w}")
      print(f"Bias: {model.b}")
      print(f"Final MSE: {model.loss_history[-1]}")