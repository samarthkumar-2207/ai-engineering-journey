import numpy as np


def split_features_target(data):
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def shuffle_data(X, y, seed=42):
    np.random.seed(seed)

    indices = np.random.permutation(len(X))

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled

def train_test_split(X, y, test_size):
    split_index = int((1 - test_size) * len(X))

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

# StandardScaler from scratch

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # if std becomes 0 due to constant value of feature
        self.std[self.std == 0] = 1

        return self

    def transform(self, X):
        return (X - self.mean) / self.std 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)