import numpy as np

from preprocessor import(
  split_features_target,
  shuffle_data,
  train_test_split,
  StandardScaler,
)


# Columns:
# age, income, transactions, fraud

data = np.array([
    [22, 30000, 4, 0],
    [25, 35000, 5, 0],
    [28, 40000, 7, 0],
    [30, 45000, 6, 0],
    [35, 75000, 12, 0],
    [38, 80000, 15, 0],
    [41, 25000, 30, 1],
    [45, 20000, 35, 1],
    [50, 15000, 40, 1],
    [55, 18000, 38, 1],
])

# print("Dataset:")
# print(data)

# print("\nShape:", data.shape)

# # Seperate input features and target

# X = data[:, :-1]
# y = data[:, -1]

# print("\nFeatures (X):")
# print(X)

# print("\nTarget (y):")
# print(y)

# print("\nX shape:", X.shape)
# print("y shape:", y.shape)

# # Shuffle the dataset becuase
# # if we don't then when we train(80%) and test(20%)
# # we will only get fraud cases in test which is bad

# np.random.seed(42)

# indices = np.random.permutation(len(X))

# X = X[indices]
# y = y[indices]

# print("\nShuffled features:")
# print(X)

# print("\nShuffled target:")
# print(y)

# # Train/test split

# split_index = int(0.8 * len(X))

# X_train = X[:split_index]
# X_test = X[split_index:]

# y_train = y[:split_index]
# y_test = y[split_index:]

# print("\nTraining data:")
# print(X_train)

# print("\nTesting data:")
# print(X_test)

# print("\nX_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

# print("\ny_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

# # Feature scaling using training data only 
# # so large-scale features don't dominate small-scale features

# train_mean = np.mean(X_train, axis=0)
# train_std = np.std(X_train, axis=0)

# X_train_scaled = (X_train - train_mean) / train_std
# X_test_scaled = (X_test - train_mean) / train_std

# print("\nTraining mean:")
# print(train_mean)

# print("\nTraining standard deviation:")
# print(train_std)

# print("\nScaled training data:")
# print(X_train_scaled)

# print("\nScaled testing data:")
# print(X_test_scaled)

# # Why? Because X includes the test data.
# # That causes data leakage:
# # Information from the test set influences the training/preprocessing process.


# 1. Seperate the featuresa and target

X, y = split_features_target(data)

# 2. Shuffle

X, y = shuffle_data(X, y)

# 3. Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\ny_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining mean:")
print(np.mean(X_train_scaled, axis=0))

print("\nTraining std:")
print(np.std(X_train_scaled, axis=0))

print("\nScaled training data:")
print(X_train_scaled)

print("\nScaled testing data:")
print(X_test_scaled)