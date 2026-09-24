import pandas as pd

df = pd.read_csv("05-pandas/02-data-cleaning/customer_data_cleaned.csv")

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

# print(df.isnull().sum())

# print(df["churned"].value_counts())
# print(df["churned"].value_counts(normalize=True) * 100)

# numerical featueres
# print(df[["age", "tenure", "income"]].describe())
# print()
# # categorial features
# print(df["city"].value_counts(normalize=True) * 100)
# print()
# print(df["plan"].value_counts(normalize=True)  * 100)
# print()
# # churn by plan

# print(
#   pd.crosstab(
#     df["plan"],
#     df["churned"]
#   )
# )
# print()
# print(
#   pd.crosstab(
#     df["plan"],
#     df["churned"],
#     normalize = True
#   ) * 100
# )

# print()
# print(df.groupby("churned")[["income", "age", "tenure"]].mean())
## Correlation
corr = df[["age", "income", "tenure"]].corr()

print(corr)