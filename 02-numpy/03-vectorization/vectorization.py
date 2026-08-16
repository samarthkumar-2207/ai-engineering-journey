import numpy as np

prices = np.array([100, 250, 300, 150])
discounts = np.array([10, 25, 30, 15])

# print("Amount: ", amount)
# print("Double: ", amount * 2)
# print("Add 50: ", amount + 50)
# print("Half: ", amount / 2)

final_price = prices - discounts

print("Prices: ", prices)
print("Discounts: ", discounts)
print("Final price: ", final_price)

tax = np.array([5, 10, 15, 8])

final_price = prices - discounts + tax

print("After tax:", final_price)