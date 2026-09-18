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

def train_test_split(X, y, test_size=0.2, seed=42):

    np.random.seed(seed)

    indices = np.random.permutation(len(X))

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    return 1 - (ss_res / ss_tot)


X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [16, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [20, 21],
], dtype=float)

y = np.array([
    10.2, 14.8, 20.5, 24.7, 30.1,
    34.9, 40.3, 44.8, 50.2, 54.6,
    60.4, 64.9, 70.1, 75.3, 79.8,
    84.7, 90.2, 95.1, 99.6, 105.2
], dtype=float)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    seed=42
)


model = MultipleLinearRegression(
    learning_rate=0.01,
    epochs=5000
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Actual:")
print(y_test)

print("\nPredicted:")
print(y_pred)

print("\nEvaluation:")
print("MSE :", mse(y_test, y_pred))
print("MAE :", mae(y_test, y_pred))
print("RMSE:", rmse(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))