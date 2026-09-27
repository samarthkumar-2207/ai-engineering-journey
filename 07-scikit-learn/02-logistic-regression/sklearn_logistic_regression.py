import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# Features
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


# Target
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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Create model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Probabilities
y_prob = model.predict_proba(X_test)[:, 1]


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


print("Predictions:", y_pred)
print("Actual:", y_test)

print("\nProbabilities:", y_prob)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)