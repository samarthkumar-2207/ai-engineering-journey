import pandas as pd

data = {
    "age": [21, 25, None, 35, 40, 45],
    "salary": [30000, 40000, 55000, None, 90000, 100000],
    "experience": [1, 2, 4, 6, 10, 12],
    "purchased": [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)

# print(df["age"])

# print(df.head(3))
# print(df.tail())

# print(df.shape)
# print(df.columns)
# print(df.dtypes)

# df.info()
# print(df.describe())

# X = df[["age","salary","experience"]]
# y = df["purchased"]

# print(X)
# print(y)


# print(df[["age"]])

# high_salary = df[df["salary"] > 50000]

# print(high_salary)

# result = df[
#     (df["salary"] > 50000) &
#     (df["experience"] > 2)
# ]

# result = df[
#     (df["salary"] > 50000) |
#     (df["experience"] > 8)
# ]

# print(result)

# print(df.isnull().sum())
