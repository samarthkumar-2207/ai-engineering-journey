from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_titanic
from features import TitanicFeatureEngineer


# ==================================================
# 1. LOAD DATA
# ==================================================

X, y = load_titanic()


# ==================================================
# 2. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 4. SAVE TEST SET
# ==================================================

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

test_data = X_test.copy()
test_data["survived"] = y_test

test_data_path = PROCESSED_DATA_DIR / "test.csv"

test_data.to_csv(
    test_data_path,
    index=False
)


# ==================================================
# 5. DEFINE FEATURE TYPES
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
# 6. NUMERICAL PREPROCESSING
# ==================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==================================================
# 7. CATEGORICAL PREPROCESSING
# ==================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==================================================
# 8. COMBINE PREPROCESSING
# ==================================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])


# ==================================================
# 9. COMPLETE ML PIPELINE
# ==================================================

model_pipeline = Pipeline([
    ("feature_engineering", TitanicFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# ==================================================
# 10. TRAIN MODEL
# ==================================================

model_pipeline.fit(
    X_train,
    y_train
)


# ==================================================
# 11. SAVE MODEL
# ==================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_path = MODEL_DIR / "titanic_logistic_pipeline.joblib"

joblib.dump(
    model_pipeline,
    model_path
)


# ==================================================
# 12. TRAINING INFORMATION
# ==================================================

print("Training completed successfully.")

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTest set saved to:")
print(test_data_path)

print("\nModel:")
print(model_pipeline)

print("\nModel saved to:")
print(model_path)