import pandas as pd

from data import load_raw_titanic


# ==================================================
# 1. LOAD ORIGINAL DATASET
# ==================================================

df = load_raw_titanic()


# ==================================================
# 2. BASIC DATASET INFORMATION
# ==================================================

print("Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ==================================================
# 3. MISSING VALUES
# ==================================================

print("\nMissing values:")
print(df.isnull().sum())


# ==================================================
# 4. TARGET DISTRIBUTION
# ==================================================

print("\nTarget distribution:")
print(df["survived"].value_counts())

print("\nTarget proportions:")
print(df["survived"].value_counts(normalize=True))


# ==================================================
# 5. ORIGINAL DATASET FEATURES
# ==================================================

print("\nOriginal Features:")

print("\npclass:")
print("Passenger class: 1st, 2nd, or 3rd class.")

print("\nsurvived:")
print("Target variable: 0 = did not survive, 1 = survived.")

print("\nname:")
print("Passenger name.")

print("\nsex:")
print("Passenger sex.")

print("\nage:")
print("Passenger age.")

print("\nsibsp:")
print("Number of siblings or spouses aboard.")

print("\nparch:")
print("Number of parents or children aboard.")

print("\nticket:")
print("Passenger ticket identifier.")

print("\nfare:")
print("Passenger fare.")

print("\ncabin:")
print("Passenger cabin information.")

print("\nembarked:")
print("Port where the passenger boarded.")

print("\nboat:")
print("Lifeboat information.")

print("\nbody:")
print("Body identification number, when applicable.")

print("\nhome.dest:")
print("Passenger home/destination information.")


# ==================================================
# 6. FEATURE SELECTION
# ==================================================

print("\nFeature Selection:")
print("\nFeatures removed:")

print("\nname:")
print("Removed initially because it is high-cardinality text.")

print("\nticket:")
print("Removed initially because it behaves like a high-cardinality identifier.")

print("\ncabin:")
print("Removed initially because it contains a very large number of missing values.")

print("\nboat:")
print("Removed because it contains post-outcome information and can cause target leakage.")

print("\nbody:")
print("Removed because it contains post-outcome information and can cause target leakage.")

print("\nhome.dest:")
print("Removed initially because it has substantial missing data and is free-text information.")


# ==================================================
# 7. FINAL SELECTED FEATURES
# ==================================================

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"

X = df[features]
y = df[target].astype(int)


print("\nSelected Features:")
print(X.head())

print("\nSelected Feature Names:")
print(features)

print("\nTarget:")
print(y.head())