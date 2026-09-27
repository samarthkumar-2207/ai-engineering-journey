import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy before saving:", accuracy)


joblib.dump(
    pipeline,
    "07-scikit-learn/breast_cancer_pipeline.joblib"
)

print("Model saved.")


loaded_pipeline = joblib.load(
    "07-scikit-learn/breast_cancer_pipeline.joblib"
)

y_loaded_pred = loaded_pipeline.predict(X_test)

loaded_accuracy = accuracy_score(
    y_test,
    y_loaded_pred
)

print("Accuracy after loading:", loaded_accuracy)