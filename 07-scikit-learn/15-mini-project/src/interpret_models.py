from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
# 5. LOAD LOGISTIC REGRESSION PIPELINE
# ==================================================

logistic_model = joblib.load(MODEL_PATH)


# ==================================================
# 6. LOGISTIC REGRESSION COEFFICIENTS
# ==================================================

logistic_preprocessor = (
    logistic_model
    .named_steps["preprocessor"]
)

logistic_estimator = (
    logistic_model
    .named_steps["model"]
)

logistic_feature_names = (
    logistic_preprocessor
    .get_feature_names_out()
)

logistic_coefficients = (
    logistic_estimator
    .coef_[0]
)

logistic_interpretation = pd.DataFrame({
    "Feature": logistic_feature_names,
    "Coefficient": logistic_coefficients,
})

logistic_interpretation["Absolute_Coefficient"] = (
    logistic_interpretation["Coefficient"]
    .abs()
)

logistic_interpretation = (
    logistic_interpretation
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
)


# ==================================================
# 7. RANDOM FOREST PIPELINE
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

random_forest_pipeline = Pipeline([
    ("feature_engineering", TitanicFeatureEngineer()),
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
# 8. TRAIN RANDOM FOREST
# ==================================================

random_forest_pipeline.fit(
    X_train,
    y_train
)


# ==================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

rf_preprocessor = (
    random_forest_pipeline
    .named_steps["preprocessor"]
)

rf_estimator = (
    random_forest_pipeline
    .named_steps["model"]
)

rf_feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)

rf_importances = (
    rf_estimator
    .feature_importances_
)

rf_interpretation = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importances,
})

rf_interpretation = (
    rf_interpretation
    .sort_values(
        "Importance",
        ascending=False
    )
)


# ==================================================
# 10. DISPLAY RANDOM FOREST IMPORTANCE
# ==================================================

print("\nRandom Forest Feature Importance")
print("================================")

print(
    rf_interpretation.to_string(
        index=False
    )
)


# ==================================================
# 11. DISPLAY LOGISTIC REGRESSION COEFFICIENTS
# ==================================================

print("\nLogistic Regression Coefficients")
print("=================================")

print(
    logistic_interpretation[
        [
            "Feature",
            "Coefficient",
            "Absolute_Coefficient",
        ]
    ].to_string(index=False)
)