import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [3, 3],
    [4, 3],
    [4, 4],
    [5, 4],
    [5, 5],
    [6, 5],
])

y = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy)