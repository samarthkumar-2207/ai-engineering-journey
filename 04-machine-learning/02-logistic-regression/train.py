import numpy as np

from logistic_regression import LogisticRegression


X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
], dtype=float)


# Shuffle the dataset
np.random.seed(42)

indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]


# Train/Test split
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# Train model
model = LogisticRegression(
    learning_rate=0.1,
    epochs=5000
)

model.fit(X_train, y_train)

for threshold in [0.3, 0.5, 0.7, 0.9]:
    predictions = model.predict(X_test, threshold=threshold)

    print(f"\nThreshold: {threshold}")
    print("Predictions:", predictions)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


print("X Test:")
print(X_test.flatten())

print("\nActual:")
print(y_test)



print("\nProbabilities:")
print(y_prob)

print("\nPredicted:")
print(y_pred)



from classification_metrics import (
    accuracy,
    precision,
    recall,
    f1_score
)

for threshold in [0.3, 0.5, 0.7, 0.9]:

    predictions = model.predict(
        X_test,
        threshold=threshold
    )

    print(f"\nThreshold: {threshold}")
    print("Predictions:", predictions)

    print("Accuracy :", accuracy(y_test, predictions))
    print("Precision:", precision(y_test, predictions))
    print("Recall   :", recall(y_test, predictions))
    print("F1 Score :", f1_score(y_test, predictions))