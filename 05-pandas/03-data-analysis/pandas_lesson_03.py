import pandas as pd

df = pd.read_csv("05-pandas/02-data-cleaning/customer_data_cleaned.csv")

# print("DATASET")
# print(df)

# print("\nFind all customers with: income > 50000")
# income_abv_50000 = df[df["income"] > 50000]
# print(income_abv_50000)

# print("\nFind all customers with: tenure > 15")
# tenure_abv_15 = df[df["tenure"] > 15]
# print(tenure_abv_15)

# print("\nFind all Premium customer from Delhi")
# premium_cust_delhi = df[
#   (df["plan"] == "Premium") &
#   (df["city"] == "Delhi")
# ]
# print(premium_cust_delhi)

# print("\nFind all customers: income > 60000 AND churned == 0")
# income_churned = df[
#   (df["income"] > 60000) &
#   (df["churned"] == 0)
# ]
# print(income_churned)

# print("\nSort customer by: tenure → highest to lowest")
# tenure_htol = df.sort_values(["tenure"], ascending=False)
# print(tenure_htol)

# print("\nFind 3 customers with highest income:")
# highest_income = df.sort_values(["income"], ascending=False).head(3)
# print(highest_income)

# print(df.loc[2])

# print(df.loc[2, ["age", "income", "tenure"]])

# result = df.loc[
#     df["income"] > 60000,
#     ["customer_id", "income", "tenure"]
# ]

# print(result)

# print("\nFind Premium customers with tenure greater than 20:")
# cust_20 = df.loc[
#   (df["plan"] == "Premium") &
#   (df["tenure"] > 20),
#   ["customer_id", "income", "tenure", "city"]
# ]
# print(cust_20)

# print("\nFind customers who churned (churned == 1):")
# cust_churn = df.loc[
#   df["churned"] == 1,
#   ["customer_id", "income", "plan"]
# ]
# print(cust_churn)

print("\nFind the average income for each plan:")
print(df.groupby("plan")["income"].mean())

print("\nFind the average tenure for each city:")
print(df.groupby("city")["tenure"].mean())

print("\nFind the number of customers in each plan:")
print(df.groupby("plan")["customer_id"].count())

print("\nFind the average income for each combination of city and plan:")
print(df.groupby(["city", "plan"])["income"].mean())
