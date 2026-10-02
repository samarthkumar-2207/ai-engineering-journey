from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_titanic
from features import TitanicFeatureEngineer


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "titanic_logistic_pipeline.joblib"
)


# ==================================================
# 2. LOAD DATA
# ==================================================

X, y = load_titanic()


# ==================================================
# 3. REPRODUCE THE EXISTING TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 4. FEATURE DEFINITIONS
# ==================================================

NUMERICAL_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "is_alone",
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
]


# ==================================================
# 5. LOAD SAVED LOGISTIC REGRESSION PIPELINE
# ==================================================

logistic_model = joblib.load(MODEL_PATH)


# ==================================================
# 6. LOGISTIC REGRESSION PREDICTIONS
# ==================================================

logistic_pred = logistic_model.predict(X_test)

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]


# ==================================================
# 7. MAJORITY-CLASS BASELINE
# ==================================================

majority_class = y_train.mode()[0]

baseline_pred = [majority_class] * len(y_test)


# ==================================================
# 8. RANDOM FOREST PREPROCESSING
# ==================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


random_forest_preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        NUMERICAL_FEATURES
    ),
    (
        "cat",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    ),
])


# ==================================================
# 9. RANDOM FOREST PIPELINE
# ==================================================

# ==================================================
# 9. RANDOM FOREST PIPELINE
# ==================================================

random_forest_pipeline = Pipeline([
    (
        "feature_engineering",
        TitanicFeatureEngineer()
    ),
    (
        "preprocessor",
        random_forest_preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ),
])


# ==================================================
# 10. TRAIN RANDOM FOREST
# ==================================================

random_forest_pipeline.fit(
    X_train,
    y_train
)


# ==================================================
# 11. RANDOM FOREST PREDICTIONS
# ==================================================

rf_pred = random_forest_pipeline.predict(X_test)

rf_prob = random_forest_pipeline.predict_proba(X_test)[:, 1]


# ==================================================
# 12. EVALUATION FUNCTION
# ==================================================

def evaluate_model(y_true, y_pred, y_prob=None):
    results = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    if y_prob is not None:
        results["ROC-AUC"] = roc_auc_score(
            y_true,
            y_prob
        )

    return results


# ==================================================
# 13. EVALUATE MAJORITY BASELINE
# ==================================================

baseline_results = evaluate_model(
    y_test,
    baseline_pred
)


# ==================================================
# 14. EVALUATE LOGISTIC REGRESSION
# ==================================================

logistic_results = evaluate_model(
    y_test,
    logistic_pred,
    logistic_prob
)


# ==================================================
# 15. EVALUATE RANDOM FOREST
# ==================================================

rf_results = evaluate_model(
    y_test,
    rf_pred,
    rf_prob
)


# ==================================================
# 16. MODEL COMPARISON
# ==================================================

results = pd.DataFrame(
    [
        baseline_results,
        logistic_results,
        rf_results,
    ],
    index=[
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest",
    ]
)


# ==================================================
# 17. DISPLAY RESULTS
# ==================================================

print("\nModel Comparison")
print("================")

print(results)