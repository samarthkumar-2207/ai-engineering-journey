import numpy as np


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tn, fp, fn, tp


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)



y_true = np.array([1, 0, 1])
y_pred = np.array([1, 1, 1])


tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nMetrics:")
print("Accuracy :", accuracy(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall   :", recall(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))