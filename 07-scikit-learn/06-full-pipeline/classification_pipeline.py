import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = np.array([
    [22, 35000, "Delhi", "Basic"],
    [29, 52000, "Mumbai", "Premium"],
    [35, 68000, "Pune", "Premium"],
    [41, 82000, "Delhi", "Basic"],
    [25, 40000, "Mumbai", "Basic"],
    [32, 60000, "Pune", "Premium"],
    [45, 90000, "Delhi", "Premium"],
    [27, 45000, "Pune", "Basic"],
], dtype=object)

y = np.array([
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1
])


X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  stratify=y
)

preprocessor = ColumnTransformer(
  transformers=[
    ("num", StandardScaler(), [0,1]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [2,3]),
  ]
)

pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("model", LogisticRegression()),
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Probabilities:", y_prob)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))