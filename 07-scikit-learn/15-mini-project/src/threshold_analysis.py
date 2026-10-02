from pathlib import Path
import json

import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from data import load_titanic


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT.parent
    / "models"
    / "titanic_logistic_pipeline.joblib"
)

TEST_PATH = (
    PROJECT_ROOT.parent
    / "data"
    / "processed"
    / "test.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT.parent
    / "data"
    / "processed"
    / "threshold_analysis.csv"
)

THRESHOLD_PATH = (
    PROJECT_ROOT.parent
    / "models"
    / "titanic_threshold.json"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

DEFAULT_THRESHOLD = 0.50


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a given threshold."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def main():

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    X, y = load_titanic()

    # Reproduce the official train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 2. Create validation split
    # --------------------------------------------------

    X_train_sub, X_validation, y_train_sub, y_validation = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )
    )

    print("Threshold Analysis")
    print("==================")
    print(f"Original training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Untouched test samples: {len(X_test)}")
    print()

    # --------------------------------------------------
    # 3. Load saved production pipeline
    # --------------------------------------------------

    production_model = load(MODEL_PATH)

    # Clone the production pipeline so that validation
    # training does not modify the saved model.
    validation_model = clone(production_model)

    # --------------------------------------------------
    # 4. Train validation model
    # --------------------------------------------------

    validation_model.fit(
        X_train_sub,
        y_train_sub,
    )

    # --------------------------------------------------
    # 5. Evaluate thresholds on validation set
    # --------------------------------------------------

    validation_probabilities = validation_model.predict_proba(
        X_validation
    )[:, 1]

    validation_results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )

        validation_results.append(metrics)

    validation_df = pd.DataFrame(validation_results)

    # Select the threshold with the highest validation F1.
    best_row = validation_df.loc[
        validation_df["F1"].idxmax()
    ]

    validation_selected_threshold = float(
        best_row["Threshold"]
    )

    # --------------------------------------------------
    # 6. Evaluate thresholds on untouched test set
    # --------------------------------------------------

    test_df = pd.read_csv(TEST_PATH)

    y_test_persisted = test_df["survived"].astype(int)

    X_test_persisted = test_df.drop(
        columns=["survived"]
    )

    # Use the already-trained production model.
    test_probabilities = production_model.predict_proba(
        X_test_persisted
    )[:, 1]

    default_metrics = calculate_metrics(
        y_test_persisted,
        test_probabilities,
        DEFAULT_THRESHOLD,
    )

    validation_selected_metrics = calculate_metrics(
        y_test_persisted,
        test_probabilities,
        validation_selected_threshold,
    )

    # --------------------------------------------------
    # 7. Save threshold experiment results
    # --------------------------------------------------

    validation_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    threshold_info = {
        "production_threshold": DEFAULT_THRESHOLD,
        "validation_selected_threshold": validation_selected_threshold,
        "selection_metric": "F1",
        "validation_samples": len(X_validation),
        "test_samples": len(X_test_persisted),
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_info,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # 8. Display validation results
    # --------------------------------------------------

    print("Validation Threshold Results")
    print("============================")

    print(
        validation_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            },
        )
    )

    print()
    print("Validation-Selected Threshold")
    print("=============================")
    print(
        f"Threshold: {validation_selected_threshold:.2f}"
    )
    print(
        f"Validation F1: {best_row['F1']:.4f}"
    )

    # --------------------------------------------------
    # 9. Display final test comparison
    # --------------------------------------------------

    print()
    print("Final Test Evaluation")
    print("=====================")

    comparison = pd.DataFrame(
        [
            {
                "Threshold": DEFAULT_THRESHOLD,
                "Accuracy": default_metrics["Accuracy"],
                "Precision": default_metrics["Precision"],
                "Recall": default_metrics["Recall"],
                "F1": default_metrics["F1"],
            },
            {
                "Threshold": validation_selected_threshold,
                "Accuracy": validation_selected_metrics["Accuracy"],
                "Precision": validation_selected_metrics["Precision"],
                "Recall": validation_selected_metrics["Recall"],
                "F1": validation_selected_metrics["F1"],
            },
        ]
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            },
        )
    )

    print()
    print(f"Production threshold: {DEFAULT_THRESHOLD:.2f}")
    print(
        "Validation-selected threshold: "
        f"{validation_selected_threshold:.2f}"
    )

    print()
    print(
        f"Saved validation results to: {RESULTS_PATH}"
    )
    print(
        f"Saved threshold information to: {THRESHOLD_PATH}"
    )


if __name__ == "__main__":
    main()