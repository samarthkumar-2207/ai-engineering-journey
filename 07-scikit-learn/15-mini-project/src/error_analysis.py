from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "titanic_logistic_pipeline.joblib"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)


# ==================================================
# 2. LOAD MODEL AND TEST DATA
# ==================================================

model = joblib.load(MODEL_PATH)

test_data = pd.read_csv(TEST_DATA_PATH)

X_test = test_data.drop(columns=["survived"])
y_test = test_data["survived"]


# ==================================================
# 3. GENERATE PREDICTIONS
# ==================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ==================================================
# 4. CREATE ERROR ANALYSIS DATAFRAME
# ==================================================

analysis = X_test.copy()

analysis["actual"] = y_test.values
analysis["predicted"] = predictions
analysis["probability"] = probabilities

analysis["correct"] = (
    analysis["actual"] == analysis["predicted"]
)

analysis["error_type"] = "Correct"

analysis.loc[
    (analysis["actual"] == 0)
    & (analysis["predicted"] == 1),
    "error_type"
] = "False Positive"

analysis.loc[
    (analysis["actual"] == 1)
    & (analysis["predicted"] == 0),
    "error_type"
] = "False Negative"


# ==================================================
# 5. OVERALL ERROR SUMMARY
# ==================================================

print("\nError Analysis")
print("==============")

print(f"Total test samples: {len(analysis)}")

print(
    f"Correct predictions: "
    f"{analysis['correct'].sum()}"
)

print(
    f"Incorrect predictions: "
    f"{(~analysis['correct']).sum()}"
)

print(
    f"Error rate: "
    f"{(~analysis['correct']).mean():.2%}"
)


# ==================================================
# 6. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix")
print("================")

print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ==================================================
# 7. ERROR TYPES
# ==================================================

print("\nError Types")
print("===========")

print(
    analysis["error_type"]
    .value_counts()
)


# ==================================================
# 8. PERFORMANCE BY SEX
# ==================================================

def evaluate_group(
    dataframe,
    group_column,
    group_value
):
    group = dataframe[
        dataframe[group_column] == group_value
    ]

    if len(group) == 0:
        return None

    return {
        "Samples": len(group),
        "Accuracy": accuracy_score(
            group["actual"],
            group["predicted"]
        ),
        "Precision": precision_score(
            group["actual"],
            group["predicted"],
            zero_division=0
        ),
        "Recall": recall_score(
            group["actual"],
            group["predicted"],
            zero_division=0
        ),
        "F1": f1_score(
            group["actual"],
            group["predicted"],
            zero_division=0
        ),
    }


print("\nPerformance by Sex")
print("==================")

sex_results = []

for value in analysis["sex"].dropna().unique():

    result = evaluate_group(
        analysis,
        "sex",
        value
    )

    result["Sex"] = value

    sex_results.append(result)

sex_results = pd.DataFrame(sex_results)

print(
    sex_results[
        [
            "Sex",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ].to_string(index=False)
)


# ==================================================
# 9. PERFORMANCE BY PASSENGER CLASS
# ==================================================

print("\nPerformance by Passenger Class")
print("=============================")

class_results = []

for value in sorted(
    analysis["pclass"].dropna().unique()
):

    result = evaluate_group(
        analysis,
        "pclass",
        value
    )

    result["Pclass"] = value

    class_results.append(result)

class_results = pd.DataFrame(class_results)

print(
    class_results[
        [
            "Pclass",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ].to_string(index=False)
)


# ==================================================
# 10. PERFORMANCE BY ALONE / FAMILY STATUS
# ==================================================

analysis["family_size"] = (
    analysis["sibsp"]
    + analysis["parch"]
    + 1
)

analysis["is_alone"] = (
    analysis["family_size"] == 1
).astype(int)


print("\nPerformance by Alone Status")
print("===========================")

alone_results = []

for value in [0, 1]:

    result = evaluate_group(
        analysis,
        "is_alone",
        value
    )

    result["Is Alone"] = value

    alone_results.append(result)

alone_results = pd.DataFrame(alone_results)

print(
    alone_results[
        [
            "Is Alone",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ].to_string(index=False)
)


# ==================================================
# 11. PERFORMANCE BY AGE GROUP
# ==================================================

analysis["age_group"] = pd.cut(
    analysis["age"],
    bins=[
        -float("inf"),
        12,
        18,
        35,
        60,
        float("inf"),
    ],
    labels=[
        "Child",
        "Teenager",
        "Young Adult",
        "Adult",
        "Senior",
    ]
)


print("\nPerformance by Age Group")
print("========================")

age_results = []

for value in analysis["age_group"].dropna().unique():

    result = evaluate_group(
        analysis,
        "age_group",
        value
    )

    result["Age Group"] = value

    age_results.append(result)

age_results = pd.DataFrame(age_results)

print(
    age_results[
        [
            "Age Group",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ].to_string(index=False)
)


# ==================================================
# 12. FALSE POSITIVES
# ==================================================

false_positives = analysis[
    analysis["error_type"] == "False Positive"
]

print("\nFalse Positives")
print("===============")

print(
    false_positives[
        [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
            "family_size",
            "is_alone",
            "probability",
        ]
    ].to_string(index=False)
)


# ==================================================
# 13. FALSE NEGATIVES
# ==================================================

false_negatives = analysis[
    analysis["error_type"] == "False Negative"
]

print("\nFalse Negatives")
print("===============")

print(
    false_negatives[
        [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
            "family_size",
            "is_alone",
            "probability",
        ]
    ].to_string(index=False)
)


# ==================================================
# 14. LOW-CONFIDENCE PREDICTIONS
# ==================================================

analysis["confidence"] = (
    analysis["probability"] - 0.5
).abs()


low_confidence = analysis[
    analysis["confidence"] < 0.10
].sort_values(
    "confidence"
)


print("\nLow-Confidence Predictions")
print("==========================")

print(
    f"Predictions within ±0.10 of "
    f"the 0.50 threshold: "
    f"{len(low_confidence)}"
)

print(
    low_confidence[
        [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "family_size",
            "is_alone",
            "actual",
            "predicted",
            "probability",
        ]
    ].to_string(index=False)
)


# ==================================================
# 15. SAVE ERROR ANALYSIS RESULTS
# ==================================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_analysis.csv"
)

analysis.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDetailed error analysis saved to:")
print(OUTPUT_PATH)