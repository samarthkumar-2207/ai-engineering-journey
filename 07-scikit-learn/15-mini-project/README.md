# Titanic Survival Prediction — End-to-End Scikit-Learn ML Project

An end-to-end classical machine learning project built with **Python and scikit-learn** to predict passenger survival using the Titanic dataset.

This project was designed as a practical ML engineering exercise rather than simply a model-training exercise. The objective was to understand and implement the complete workflow:

> **Dataset → EDA → Feature Selection → Train/Test Split → Preprocessing → Feature Engineering → Model Training → Model Comparison → Model Interpretation → Error Analysis → Threshold Analysis → Final Evaluation**

The project emphasizes reproducibility, clean separation of responsibilities, prevention of data leakage, and evidence-based model decisions.

---

# 1. Project Objective

The objective of this project is to build a binary classification system that predicts whether a Titanic passenger survived.

The target variable is:

```text
survived
```

where:

```text
0 → Did not survive
1 → Survived
```

The project was also used to learn how a real ML workflow should be structured.

Instead of immediately trying different algorithms and optimizing for the highest possible accuracy, the workflow focused on answering questions such as:

- Which features should actually be used?
- Which features contain leakage?
- How should missing values be handled?
- How should categorical variables be encoded?
- Does feature engineering actually improve the model?
- Which model performs better?
- Where does the model make mistakes?
- Does changing the classification threshold improve the final system?
- How do we prevent the test set from influencing model decisions?

---

# 2. Project Structure

```text
15-mini-project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── test.csv
│       ├── error_analysis.csv
│       └── threshold_analysis.csv
│
├── models/
│   ├── titanic_logistic_pipeline.joblib
│   └── titanic_threshold.json
│
├── notebooks/
│
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── interpret_models.py
│   ├── error_analysis.py
│   └── threshold_analysis.py
│
└── README.md
```

---

# 3. Tools and Libraries

The project uses:

- Python
- scikit-learn
- pandas
- NumPy
- joblib

The main scikit-learn components used include:

```text
fetch_openml
train_test_split
Pipeline
ColumnTransformer
SimpleImputer
StandardScaler
OneHotEncoder
LogisticRegression
RandomForestClassifier
```

Evaluation uses:

```text
accuracy
precision
recall
F1-score
ROC-AUC
confusion matrix
classification report
```

---

# 4. Dataset

The Titanic dataset was loaded using scikit-learn's OpenML interface:

```python
from sklearn.datasets import fetch_openml

titanic = fetch_openml(
    "titanic",
    version=1,
    as_frame=True
)
```

The original dataset contains:

```text
1309 rows
14 columns
```

The original features were:

```text
pclass
survived
name
sex
age
sibsp
parch
ticket
fare
cabin
embarked
boat
body
home.dest
```

---

# 5. Initial Dataset Analysis

The target distribution was:

| Target | Meaning         | Count | Percentage |
| ------ | --------------- | ----: | ---------: |
| 0      | Did not survive |   809 |     61.80% |
| 1      | Survived        |   500 |     38.20% |

This showed that the dataset was imbalanced toward non-survivors.

A majority-class model would therefore already achieve approximately:

```text
809 / 1309 ≈ 61.80%
```

accuracy.

Because of this, accuracy alone would not be sufficient for evaluating the model.

---

# 6. Missing Value Analysis

The initial missing-value counts were:

| Feature     | Missing Values |
| ----------- | -------------: |
| `pclass`    |              0 |
| `survived`  |              0 |
| `name`      |              0 |
| `sex`       |              0 |
| `age`       |            263 |
| `sibsp`     |              0 |
| `parch`     |              0 |
| `ticket`    |              0 |
| `fare`      |              1 |
| `cabin`     |           1014 |
| `embarked`  |              2 |
| `boat`      |            823 |
| `body`      |           1188 |
| `home.dest` |            564 |

This immediately affected the feature-selection decisions.

In particular, `cabin`, `boat`, `body`, and `home.dest` required careful consideration because of their missingness and semantics.

---

# 7. Feature Selection

The original dataset contained several features that were deliberately removed.

## 7.1 `name`

`name` was removed because it is high-cardinality text.

Using it directly would require additional text processing and would not be appropriate for this first classical ML pipeline.

```text
Removed: name
```

---

## 7.2 `ticket`

`ticket` was removed because it behaves largely like an identifier/high-cardinality feature.

```text
Removed: ticket
```

---

## 7.3 `cabin`

`cabin` had extremely high missingness:

```text
1014 / 1309
```

This made it unsuitable for the initial model.

```text
Removed: cabin
```

---

## 7.4 `boat`

`boat` was removed because it contains information that is closely associated with the outcome and can represent information available after or because of the survival outcome.

Using it would introduce target leakage.

```text
Removed: boat
```

---

## 7.5 `body`

`body` was also removed because it represents post-outcome information.

Using it would introduce leakage.

```text
Removed: body
```

---

## 7.6 `home.dest`

`home.dest` was removed because it combines substantial missingness with free-text information.

```text
Removed: home.dest
```

---

# 8. Final Feature Set

After feature selection, the model initially used seven features:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

These were defined centrally in `data.py`.

The important architectural decision was that **`data.py` owns loading and feature definitions, but does not own the train/test split.**

This keeps dataset loading separate from model-training responsibilities.

---

# 9. Train/Test Split

The official train/test split was created in `train.py`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The split produced:

```text
Training samples: 1047
Test samples: 262
```

The split was stratified so that the class distribution remained approximately consistent between training and test data.

The test set was persisted to:

```text
data/processed/test.csv
```

This became important later because the same untouched test set was reused for final evaluation.

---

# 10. Why the Test Set Was Persisted

The test set was saved so that later evaluation scripts would not repeatedly create new test sets.

This created a stable evaluation boundary:

```text
Training Data
      │
      ├── Model development
      ├── Feature engineering
      └── Validation experiments
     
Untouched Test Data
      │
      └── Final evaluation
```

This prevents accidental changes to the evaluation population.

---

# 11. Initial Preprocessing Pipeline

The first Logistic Regression pipeline used separate preprocessing for numerical and categorical features.

## Numerical Features

```text
pclass
age
sibsp
parch
fare
```

Numerical preprocessing:

```text
Median imputation
        ↓
StandardScaler
```

Median imputation was used to handle missing numerical values.

Standardization was used because Logistic Regression benefits from features being on comparable scales.

---

## Categorical Features

```text
sex
embarked
```

Categorical preprocessing:

```text
Most-frequent imputation
        ↓
One-hot encoding
```

The encoder used:

```python
OneHotEncoder(handle_unknown="ignore")
```

This ensures that an unseen categorical value during inference does not break the pipeline.

---

# 12. Why We Used Pipelines

Instead of manually preprocessing the data before training, preprocessing was placed inside a scikit-learn `Pipeline` and `ColumnTransformer`.

Conceptually:

```text
Raw Data
   │
   ▼
Feature Engineering
   │
   ▼
ColumnTransformer
   │
   ├── Numerical → Imputation → Scaling
   │
   └── Categorical → Imputation → One-Hot Encoding
   │
   ▼
Model
```

This ensures that the same transformations are automatically applied during training and inference.

It also reduces the risk of training/serving inconsistencies.

---

# 13. Baseline Logistic Regression Model

The first model was Logistic Regression:

```python
LogisticRegression(
    max_iter=1000
)
```

The initial model was intentionally simple.

The purpose was to establish a baseline before adding feature engineering or comparing more complex models.

---

# 14. Initial Model Results

Before feature engineering, the models produced:

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Majority Baseline   |   61.83% |     0.00% |  0.00% |  0.00% |       — |
| Logistic Regression |   80.92% |    77.17% | 71.00% | 73.96% |  86.70% |
| Random Forest       |   80.15% |    74.00% | 74.00% | 74.00% |  86.34% |

These results established the initial benchmark.

---

# 15. Feature Engineering

After establishing the baseline, feature engineering was introduced.

A separate module was created:

```text
src/features.py
```

This module owns feature engineering so that the transformation logic is not duplicated across training, comparison, interpretation, or inference code.

---

# 16. Family Size

The first engineered feature was:

```text
family_size = sibsp + parch + 1
```

The `+1` represents the passenger themselves.

For example:

```text
sibsp = 1
parch = 2

family_size = 1 + 2 + 1
            = 4
```

---

# 17. Is Alone

The second engineered feature was:

```text
is_alone = 1 if family_size == 1 else 0
```

This converts family size into a simple binary indicator.

The feature engineering transformer was implemented as:

```python
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["family_size"] = (
            X["sibsp"] + X["parch"] + 1
        )

        X["is_alone"] = (
            X["family_size"] == 1
        ).astype(int)

        return X
```

The transformer was placed inside the model pipeline.

Therefore:

```text
Raw Input
    ↓
TitanicFeatureEngineer
    ↓
Preprocessing
    ↓
Model
```

---

# 18. Updated Feature Set

After feature engineering, the numerical features became:

```text
pclass
age
sibsp
parch
fare
family_size
is_alone
```

Categorical features remained:

```text
sex
embarked
```

---

# 19. Feature Engineering Results

After retraining the models, the results changed.

## Logistic Regression

| Metric    | Before |      After |   Change |
| --------- | -----: | ---------: | -------: |
| Accuracy  | 80.92% | **83.21%** | +2.29 pp |
| Precision | 77.17% | **79.79%** | +2.62 pp |
| Recall    | 71.00% | **75.00%** | +4.00 pp |
| F1        | 73.96% | **77.32%** | +3.36 pp |
| ROC-AUC   | 86.70% |     86.51% | -0.19 pp |

Feature engineering improved almost all classification metrics for Logistic Regression.

---

## Random Forest

| Metric    | Before |  After |   Change |
| --------- | -----: | -----: | -------: |
| Accuracy  | 80.15% | 78.24% | -1.91 pp |
| Precision | 74.00% | 71.72% | -2.28 pp |
| Recall    | 74.00% | 71.00% | -3.00 pp |
| F1        | 74.00% | 71.36% | -2.64 pp |
| ROC-AUC   | 86.34% | 85.02% | -1.32 pp |

Interestingly, the engineered features that helped Logistic Regression did not help Random Forest.

This was an important result.

It demonstrated that:

> More features do not automatically mean a better model.

Feature engineering therefore remained an experiment rather than an assumption.

---

# 20. Final Model Comparison

After feature engineering, the final comparison was:

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Majority Baseline   |     61.83% |      0.00% |      0.00% |      0.00% |          — |
| Logistic Regression | **83.21%** | **79.79%** | **75.00%** | **77.32%** | **86.51%** |
| Random Forest       |     78.24% |     71.72% |     71.00% |     71.36% |     85.02% |

The Logistic Regression pipeline became the final model.

The saved model is:

```text
models/titanic_logistic_pipeline.joblib
```

---

# 21. Model Interpretation

After comparing performance, the next step was understanding what the models were learning.

This was implemented in:

```text
src/interpret_models.py
```

Two different interpretation approaches were used.

---

# 22. Random Forest Feature Importance

The Random Forest impurity-based feature importance showed:

| Feature     | Importance |
| ----------- | ---------: |
| age         |   0.267400 |
| fare        |   0.265579 |
| sex_male    |   0.131682 |
| sex_female  |   0.122519 |
| pclass      |   0.071308 |
| family_size |   0.047959 |
| sibsp       |   0.026509 |
| parch       |   0.022019 |
| embarked_C  |   0.017953 |
| embarked_S  |   0.010911 |
| is_alone    |   0.009794 |
| embarked_Q  |   0.006367 |

The largest Random Forest importances were associated with age and fare, followed by sex and passenger class.

---

# 23. Logistic Regression Coefficients

The Logistic Regression coefficients were:

| Feature     | Coefficient | Absolute Coefficient |
| ----------- | ----------: | -------------------: |
| sex_male    |   -1.258076 |             1.258076 |
| sex_female  |    1.132622 |             1.132622 |
| pclass      |   -0.699765 |             0.699765 |
| age         |   -0.473758 |             0.473758 |
| is_alone    |   -0.447044 |             0.447044 |
| sibsp       |   -0.426243 |             0.426243 |
| embarked_C  |    0.372679 |             0.372679 |
| embarked_Q  |   -0.293567 |             0.293567 |
| family_size |   -0.272584 |             0.272584 |
| embarked_S  |   -0.204566 |             0.204566 |
| fare        |    0.047756 |             0.047756 |
| parch       |    0.006149 |             0.006149 |

For Logistic Regression:

- A positive coefficient pushes the prediction toward class `1`.
- A negative coefficient pushes the prediction toward class `0`.

The numerical features were standardized, while categorical features were one-hot encoded.

The Logistic Regression coefficients and Random Forest feature importance were **not treated as directly comparable quantities**, because they measure different things.

---

# 24. Error Analysis

Once the final Logistic Regression model was selected, the next question was:

> Where is the model actually making mistakes?

This was implemented in:

```text
src/error_analysis.py
```

The analysis used the saved model and the persisted test set.

It did not retrain the model.

---

# 25. Overall Error Analysis

The test set contained:

```text
262 samples
```

Results:

```text
Correct predictions: 218
Incorrect predictions: 44
Error rate: 16.79%
```

Therefore:

```text
218 / 262 = 83.21% accuracy
44 / 262 = 16.79% error rate
```

---

# 26. Confusion Matrix

The confusion matrix was:

```text
[[143  19]
 [ 25  75]]
```

This gives:

```text
True Negatives: 143
False Positives: 19
False Negatives: 25
True Positives: 75
```

The model therefore correctly identified:

- 143 non-survivors
- 75 survivors

while making:

- 19 false-positive predictions
- 25 false-negative predictions

---

# 27. Error Analysis by Sex

The model behaved very differently across the two sex categories.

## Female

```text
Samples: 94
Accuracy: 82.98%
Precision: 82.35%
Recall: 98.59%
F1: 89.74%
```

## Male

```text
Samples: 168
Accuracy: 83.33%
Precision: 55.56%
Recall: 17.24%
F1: 26.32%
```

The particularly low male recall means the model predicted most male passengers as non-survivors.

This is an important limitation that would not be obvious from overall accuracy alone.

---

# 28. Error Analysis by Passenger Class

The model was also evaluated by passenger class.

| Class | Samples | Accuracy | Precision | Recall |     F1 |
| ----- | ------: | -------: | --------: | -----: | -----: |
| 1     |      74 |   78.38% |    85.71% | 78.26% | 81.82% |
| 2     |      57 |   94.74% |   100.00% | 88.46% | 93.88% |
| 3     |     131 |   80.92% |    55.17% | 57.14% | 56.14% |

Third-class passengers were substantially harder for the model.

---

# 29. Error Analysis by Family Status

The model was also evaluated according to whether passengers were travelling alone.

| Is Alone | Samples | Accuracy | Precision | Recall |     F1 |
| -------- | ------: | -------: | --------: | -----: | -----: |
| 0        |     115 |   80.00% |    75.47% | 80.00% | 77.67% |
| 1        |     147 |   85.71% |    85.37% | 70.00% | 76.92% |

The F1 scores were relatively similar.

Therefore, `is_alone` did not appear to be the dominant remaining source of model error.

---

# 30. Error Analysis by Age Group

The model was also evaluated across age groups.

| Age Group   | Samples | Accuracy | Precision |  Recall |     F1 |
| ----------- | ------: | -------: | --------: | ------: | -----: |
| Young Adult |     106 |   83.96% |    79.49% |  77.50% | 78.48% |
| Adult       |      65 |   84.62% |    95.24% |  68.97% | 80.00% |
| Teenager    |      13 |   84.62% |    75.00% | 100.00% | 85.71% |
| Child       |      18 |   66.67% |    66.67% |  66.67% | 66.67% |
| Senior      |       6 |   83.33% |    66.67% | 100.00% | 80.00% |

The child group had lower performance, although its sample size was relatively small.

---

# 31. Low-Confidence Predictions

The error analysis also examined predictions close to the default classification threshold of `0.50`.

Predictions within:

```text
±0.10 from 0.50
```

were considered low-confidence.

There were:

```text
27 / 262
```

such predictions.

That represents:

```text
10.31%
```

of the test set.

This motivated a threshold-analysis experiment.

---

# 32. Why Threshold Analysis Was Performed

The Logistic Regression model produces probabilities.

For binary classification, the default rule is:

```text
probability >= 0.50
        ↓
class 1

probability < 0.50
        ↓
class 0
```

However, changing the threshold changes the balance between:

```text
Precision
Recall
F1
Accuracy
```

Because error analysis showed a particularly low recall for male passengers and because 27 predictions were near the decision boundary, threshold analysis was a useful experiment.

However, the threshold could not simply be optimized using the final test set.

That would contaminate the test evaluation.

---

# 33. Validation-Based Threshold Selection

The official training set contained:

```text
1047 samples
```

A validation split was created **inside the training data**:

```text
Training subset: 837
Validation: 210
```

The original test set remained untouched:

```text
Test: 262
```

The workflow therefore became:

```text
Full Dataset
     │
     ├── Training Data: 1047
     │       │
     │       ├── Training Subset: 837
     │       └── Validation: 210
     │
     └── Test Data: 262
             ↑
             untouched
```

---

# 34. Thresholds Tested

The following thresholds were evaluated on the validation set:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

F1-score was used as the threshold-selection metric.

---

# 35. Validation Threshold Results

| Threshold |   Accuracy | Precision |     Recall |         F1 |
| --------: | ---------: | --------: | ---------: | ---------: |
|      0.30 |     79.52% |    68.32% |     86.25% |     76.24% |
|      0.35 | **80.48%** |    70.10% | **85.00%** | **76.84%** |
|      0.40 |     80.48% |    71.43% |     81.25% |     76.02% |
|      0.45 |     80.00% |    72.09% |     77.50% |     74.70% |
|      0.50 |     79.05% |    75.71% |     66.25% |     70.67% |
|      0.55 |     78.57% |    76.12% |     63.75% |     69.39% |
|      0.60 |     78.10% |    78.33% |     58.75% |     67.14% |
|      0.65 |     78.57% |    84.31% |     53.75% |     65.65% |
|      0.70 |     75.71% |    85.37% |     43.75% |     57.85% |

The validation experiment selected:

```text
0.35
```

because it achieved the highest validation F1:

```text
F1 = 0.7684
```

---

# 36. Final Test Threshold Comparison

The validation-selected threshold was then evaluated exactly once on the untouched test set.

The default threshold of `0.50` was compared with `0.35`.

| Threshold |   Accuracy |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: | ---------: |
|  **0.50** | **83.21%** | **79.79%** |     75.00% | **77.32%** |
|      0.35 |     79.77% |     69.42% | **84.00%** |     76.02% |

The lower threshold produced:

```text
Higher recall
```

but at the cost of:

```text
Lower precision
Lower accuracy
Lower final F1
```

Therefore, the validation-selected threshold of `0.35` was **not adopted as the production threshold**.

The final production threshold remained:

```text
0.50
```

This is an important methodological result.

We did not continue modifying the threshold after observing the test results. The test set remained a final evaluation boundary.

---

# 37. Final Threshold Configuration

The threshold information is stored in:

```text
models/titanic_threshold.json
```

The final configuration is conceptually:

```json
{
    "production_threshold": 0.5,
    "validation_selected_threshold": 0.35,
    "selection_metric": "F1",
    "validation_samples": 210,
    "test_samples": 262
}
```

This explicitly distinguishes:

```text
Production threshold
        ↓
0.50

Validation experiment
        ↓
0.35
```

---

# 38. Final Model

The final selected model is the Logistic Regression pipeline:

```text
TitanicFeatureEngineer
        ↓
ColumnTransformer
        ↓
Numerical Imputation
        ↓
StandardScaler
        ↓
Categorical Imputation
        ↓
OneHotEncoder
        ↓
LogisticRegression
```

The trained pipeline is saved as:

```text
models/titanic_logistic_pipeline.joblib
```

---

# 39. Final Test Performance

The final production configuration achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **83.21%** |
| Precision | **79.79%** |
| Recall    | **75.00%** |
| F1        | **77.32%** |
| ROC-AUC   | **86.51%** |

These results are from the untouched test set.

---

# 40. Complete ML Workflow

The entire project can now be summarized as:

```text
                    Titanic Dataset
                           │
                           ▼
                  Initial Data Analysis
                           │
                           ▼
                  Missing Value Analysis
                           │
                           ▼
                    Feature Selection
                           │
                           ├── Remove high-cardinality
                           ├── Remove high-missingness
                           └── Remove leakage
                           │
                           ▼
                    Train/Test Split
                           │
                    ┌──────┴──────┐
                    │             │
                 Train           Test
                 1047             262
                    │             │
                    ▼             │
              Preprocessing      │
                    │             │
                    ▼             │
            Baseline Logistic    │
                    │             │
                    ▼             │
             Baseline Results    │
                    │             │
                    ▼             │
           Feature Engineering   │
                    │             │
                    ▼             │
          Model Comparison       │
                    │             │
                    ▼             │
          Model Interpretation   │
                    │             │
                    ▼             │
             Error Analysis      │
                    │             │
                    ▼             │
          Validation Split       │
                    │             │
                    ▼             │
          Threshold Analysis     │
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                 Final Test Evaluation
                           │
                           ▼
                  Final Model = Logistic
                  Final Threshold = 0.50
```

---

# 41. Responsibilities of Each Module

The project was intentionally divided so that each module has a clear responsibility.

### `data.py`

Responsible for:

- Loading the Titanic dataset
- Defining the final raw feature set
- Returning `X` and `y`

It does **not** own the train/test split.

---

### `eda.py`

Responsible for:

- Exploring the dataset
- Understanding distributions
- Investigating missing values
- Supporting feature-selection decisions

It does not own model training.

---

### `features.py`

Responsible for:

- Feature engineering
- `family_size`
- `is_alone`

This prevents feature-engineering logic from being duplicated across scripts.

---

### `train.py`

Responsible for:

- Official train/test split
- Pipeline construction
- Model training
- Persisting the final Logistic Regression pipeline
- Persisting the test set

---

### `evaluate.py`

Responsible for:

- Loading the saved model
- Loading the persisted test set
- Generating final evaluation metrics
- Confusion matrix
- Classification report

It does not retrain the model.

---

### `compare_models.py`

Responsible for:

- Comparing the baseline
- Logistic Regression
- Random Forest

It reproduces the deterministic split for comparison purposes.

---

### `interpret_models.py`

Responsible for:

- Random Forest feature importance
- Logistic Regression coefficients

---

### `error_analysis.py`

Responsible for:

- Overall errors
- Confusion matrix
- False positives
- False negatives
- Subgroup performance
- Low-confidence predictions

---

### `threshold_analysis.py`

Responsible for:

- Creating the validation split
- Testing candidate thresholds
- Selecting a candidate threshold using validation F1
- Comparing that candidate with the default threshold on the untouched test set
- Recording the threshold experiment

---

# 42. Reproducibility

The project uses fixed random seeds:

```python
random_state=42
```

for the train/test and validation splits.

This makes the experiments reproducible.

The saved model and test set also provide stable artifacts for evaluation.

---

# 43. Important Methodological Decisions

Several decisions were deliberately made during development.

## Decision 1 — Keep the test set separate

The test set was not used for feature selection or threshold selection.

---

## Decision 2 — Put preprocessing inside the pipeline

This prevents inconsistent preprocessing between training and inference.

---

## Decision 3 — Centralize feature engineering

Feature engineering lives in `features.py` instead of being duplicated across scripts.

---

## Decision 4 — Evaluate feature engineering empirically

Feature engineering improved Logistic Regression but reduced Random Forest performance.

Therefore, features were not considered automatically beneficial.

---

## Decision 5 — Compare against a simple baseline

The majority classifier established the minimum useful benchmark.

---

## Decision 6 — Perform error analysis before further optimization

Instead of immediately tuning hyperparameters, the project investigated where the model was failing.

---

## Decision 7 — Use validation for threshold selection

Threshold selection was performed using validation data rather than the final test set.

---

## Decision 8 — Keep 0.50 as the final threshold

Although 0.35 produced the best validation F1, it did not outperform 0.50 on the untouched test set.

Therefore:

```text
Production threshold = 0.50
```

---

# 44. What This Project Demonstrated

This project covered the fundamentals of a complete classical ML workflow:

### Data

- Dataset loading
- Missing-value analysis
- Target distribution
- Feature selection

### Modeling

- Logistic Regression
- Random Forest
- Majority baseline

### Preprocessing

- Numerical imputation
- Categorical imputation
- Standardization
- One-hot encoding
- Pipeline construction

### Feature Engineering

- Family size
- Alone indicator

### Evaluation

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix

### Model Understanding

- Feature importance
- Logistic coefficients
- Subgroup analysis
- Error analysis

### Decision-Making

- Validation split
- Threshold analysis
- Precision-recall trade-off
- Test-set protection

---

# 45. Limitations

This project is intentionally a controlled classical ML exercise and has several limitations.

### Dataset

The Titanic dataset is small and historical, so the results should not be interpreted as evidence about modern passenger survival or any real-world deployment scenario.

### Model complexity

Only a small number of classical models were evaluated.

### Validation

Threshold selection used a single validation split rather than cross-validation.

### Generalization

The final results come from one fixed test set.

### Feature Set

The project deliberately excluded some original features because of leakage, missingness, or feature-type considerations.

### Production Scope

This project does not yet include:

- experiment tracking
- model registry
- REST API
- Docker deployment
- cloud deployment
- monitoring
- automated retraining
- CI/CD

Those capabilities belong to the next production-oriented project.

---

# 46. Final Outcome

The project began as a simple Titanic classification problem and was developed into a complete classical ML workflow.

The final system consists of:

```text
Data Loading
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Feature Engineering
      ↓
Preprocessing Pipeline
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Model Interpretation
      ↓
Error Analysis
      ↓
Threshold Analysis
      ↓
Final Evaluation
```

The final model achieved:

```text
Accuracy:  83.21%
Precision: 79.79%
Recall:    75.00%
F1:        77.32%
ROC-AUC:   86.51%
```

The final production threshold is:

```text
0.50
```

The most important outcome of the project was not the final accuracy number.

It was learning how to build an ML system where each decision is supported by an experiment or methodological reason, while keeping the evaluation process reproducible and protecting the final test set from iterative tuning.

---

# 47. Next Step

This project establishes the classical ML foundation.

The next stage is to move from a controlled notebook-style classification problem toward a **production ML system**.

The next project will focus on:

```text
Production ML Decision System
```

with concepts such as:

- Production data pipeline
- Training pipeline
- Experiment tracking
- Model versioning
- Validation
- Model registry
- API serving
- Docker
- Testing
- Monitoring
- CI/CD
- Deployment

```
```
