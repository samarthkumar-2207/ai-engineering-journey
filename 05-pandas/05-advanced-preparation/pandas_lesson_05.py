import pandas as pd


df = pd.read_csv("05-pandas/02-data-cleaning/customer_data_cleaned.csv")

# print(df.head())
# print()

# df["income_category"] = df["income"].apply(
#   lambda x: "High" if x >= 60000 else "Low"
# )
# print(df[["income", "income_category"]])
# print()


# plan_mapping = {
#   "Basic": 0,
#   "Premium": 1
# }

# df["plan_encoded"] = df["plan"].map(plan_mapping)

# print(df[["plan", "plan_encoded"]])
# print()

# practice = df.copy()

# practice["plan"] = practice["plan"].replace({
#   "Basic": "Standard"
# })
# print(practice["plan"])
# print()

# df = df.rename(
#   columns = {
#     "income": "annual_income"
#   }
# )

# print(df.columns)
# print()

# sorted_df = df.sort_values(
#     "annual_income",
#     ascending=False
# )

# print(sorted_df)

# sorted_df = sorted_df.reset_index(drop=True)

# print(sorted_df)

practice = df.copy()

practice.loc[2, "income"] = None

print(practice.isnull().sum())
print()

dropped = practice.dropna()

print(dropped)

filled = practice.fillna({
    "income": practice["income"].median()
})

print(filled)