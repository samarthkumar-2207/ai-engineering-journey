from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds deterministic Titanic features.

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["is_alone"] = (X["family_size"] == 1).astype(int)

        return X