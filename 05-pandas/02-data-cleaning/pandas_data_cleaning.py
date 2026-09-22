import pandas as pd


df = pd.read_csv("05-pandas/02-data-cleaning/customer_data.csv")

print("DATASET")
print(df)

print("\nSHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(df.columns)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
df.info()

print("\nSTATISTICAL INFO")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nMISSING VALUE PERCENTAGE:")
print(df.isnull().mean() * 100)

print("\nDUPLICATED VALUES:")
print(df[df.duplicated()])

print("\nNUMBER OF DUPLICATED VALUES:")
print(df.duplicated().sum())

#FILL THE MISSING VALUES USINF df.fillna() and also remove duplicated from DATASET

df["age"] = df["age"].fillna(df["age"].median())
df["income"] = df["income"].fillna(df["income"].median())

df = df.drop_duplicates()

print("\n",df.isnull().sum())
print("\n",df.duplicated().sum())


print(df["city"].unique())
print(df["city"].value_counts())

print(df["plan"].unique())
print(df["plan"].value_counts())

print(df["tenure"].unique())
print(df["tenure"].describe())
print(df["tenure"].isnull().sum())

X = df.drop(["customer_id", "churned"], axis=1)
y = df["churned"]

print(X)
print(y)

df.to_csv("05-pandas/02-data-cleaning/customer_data_cleaned.csv", index=False)