import numpy as np


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tn, fp, fn, tp


def precision_recall_curve_manual(y_true, y_prob):

    thresholds = np.linspace(0, 1, 101)

    precisions = []
    recalls = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        )

        if tp + fp == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        precisions.append(precision)
        recalls.append(recall)

    return (
        np.array(precisions),
        np.array(recalls),
        thresholds
    )

def auc_trapezoidal(x, y):
    order = np.argsort(x)

    x_sorted = x[order]
    y_sorted = y[order]

    return np.trapezoid(y_sorted, x_sorted)



y_true = np.array([1, 0, 1])

y_prob = np.array([
    0.9999291,
    0.57108183,
    0.99928121
])


precision, recall, thresholds = precision_recall_curve_manual(
    y_true,
    y_prob
)


for i in range(0, 101, 10):

    print(
        f"Threshold: {thresholds[i]:.1f} "
        f"Precision: {precision[i]:.2f} "
        f"Recall: {recall[i]:.2f}"
    )


auc = auc_trapezoidal(
    recall,
    precision
)

print("\nPR-AUC:", auc)