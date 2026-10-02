from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "titanic_logistic_pipeline.joblib"
)

# ==================================================
# 2. LOAD TEST DATA
# ==================================================

test_data = pd.read_csv(TEST_DATA_PATH)

X_test = test_data.drop(columns=["survived"])
y_test = test_data["survived"]

# ==================================================
# 3. LOAD TRAINED PIPELINE
# ==================================================

model = joblib.load(MODEL_PATH)

# ==================================================
# 4. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==================================================
# 5. CALCULATE METRICS
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

print("Model Evaluation")
print("================")

print("\nTest Samples:", len(X_test))

print("\nAccuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))