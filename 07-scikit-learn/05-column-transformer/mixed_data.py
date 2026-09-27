import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


X = np.array([
    [22, 35000, "Delhi", "Basic"],
    [29, 52000, "Mumbai", "Premium"],
    [35, 68000, "Pune", "Premium"],
    [41, 82000, "Delhi", "Basic"],
], dtype=object)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), [0, 1]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [2, 3]),
    ]
)


X_transformed = preprocessor.fit_transform(X)


print("Original:")
print(X)

print("\nTransformed:")
print(X_transformed)

print("\nShape:")
print(X_transformed.shape)