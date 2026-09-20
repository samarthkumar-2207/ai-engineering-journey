import numpy as np


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tn, fp, fn, tp


def roc_curve_manual(y_true, y_prob):
    thresholds = np.linspace(0, 1, 101)

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        )

        if fp + tn == 0:
            false_positive_rate = 0.0
        else:
            false_positive_rate = fp / (fp + tn)

        if tp + fn == 0:
            true_positive_rate = 0.0
        else:
            true_positive_rate = tp / (tp + fn)

        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)

    return np.array(fpr), np.array(tpr), thresholds

def auc_trapezoidal(fpr, tpr):
    order = np.argsort(fpr)

    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    return np.trapezoid(tpr_sorted, fpr_sorted)


y_true = np.array([1, 0, 1])

y_prob = np.array([
    0.9999291,
    0.57108183,
    0.99928121
])

fpr, tpr, thresholds = roc_curve_manual(
    y_true,
    y_prob
)

for i in range(0, 101, 10):
    print(
        f"Threshold: {thresholds[i]:.1f} "
        f"FPR: {fpr[i]:.2f} "
        f"TPR: {tpr[i]:.2f}"
    )

auc = auc_trapezoidal(fpr, tpr)

print("\nAUC:", auc)