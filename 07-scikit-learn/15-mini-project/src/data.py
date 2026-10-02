from sklearn.datasets import fetch_openml


FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

TARGET = "survived"


def load_raw_titanic():
    titanic = fetch_openml(
        "titanic",
        version=1,
        as_frame=True
    )

    return titanic.frame


def load_titanic():
    df = load_raw_titanic()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    return X, y