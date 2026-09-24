import matplotlib.pyplot as plt
import pandas as pd

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(
  x, 
  y,
  marker="o"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Line Plot")

plt.grid()

plt.show()


months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [120, 150, 170, 160, 210, 250]

plt.plot(
  months,
  sales,
  marker="o"
)

plt.title("Montly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid()

plt.show()


plans = ["Basic", "Premium"]
customers = [4, 6]

plt.bar(plans, customers)

plt.xlabel("Plan")
plt.ylabel("Number of Customers")
plt.title("Customers by Plan")

plt.show()


df = pd.read_csv("05-pandas/02-data-cleaning/customer_data_cleaned.csv")

# Number of customers in each city
city_counts = df["city"].value_counts()
# print(city_counts)

cities = df["city"].unique()

print(cities)

plt.bar(
  cities,
  city_counts
)

plt.xlabel("City")
plt.ylabel("Number of customers")
plt.title("Customers by city")

plt.grid()
plt.show()

plt.hist(df["income"], bins=5)

plt.xlabel("Income")
plt.ylabel("Number of Customers")
plt.title("Income Distribution")

plt.show()

plt.scatter(df["tenure"], df["income"])

plt.xlabel("Tenure")
plt.ylabel("Income")
plt.title("Income vs Tenure")

plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(df["income"], bins=5)
axes[0].set_title("Income Distribution")

axes[1].scatter(df["tenure"], df["income"])
axes[1].set_title("Tenure vs Income")

plt.tight_layout()
plt.savefig("06-matplotlib/01-basics/eda_plots.png", dpi=300, bbox_inches="tight")
plt.show()


