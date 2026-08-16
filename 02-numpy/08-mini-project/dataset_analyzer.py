import numpy as np

#Columns:
#amount, frquency
X = np.array([
   [100, 2],
   [250, 5],
   [300, 8],
   [150, 3],
   [500, 10],
   [120, 1],
])

print("Dataset:")
print(X)

print("\nShape: ", X.shape)
print("Number of samples: ", X.shape[0])
print("Number of features: ", X.shape[1])

#Extract Features
amount = X[:, 0]
frequencies = X[:, 1]

print("\nAmount: ", amount)
print("Frequencies: ", frequencies)

#Statistics
print("\nAmount Statistics:")
print("Mean: ", np.mean(amount))
print("Median: ",np.median(amount))
print("Standard Deviation: ", np.std(amount))
print("Variance: ", np.var(amount))
print("Min: ", np.min(amount))
print("Max: ", np.max(amount))

print("\nFrequency Statistics:")
print("Mean: ", np.mean(frequencies))
print("Median: ",np.median(frequencies))
print("Standard Deviation: ", np.std(frequencies))
print("Variance: ", np.var(frequencies))
print("Min: ", np.min(frequencies))
print("Max: ", np.max(frequencies))

# #Feature Centuring
# feature_means = np.mean(X, axis=0)

# X_centered = X - feature_means

# print("\nFeature means:")
# print(feature_means)

# print("\nCentered means:")
# print(X_centered)

# #Normalize the Features
# feature_std = np.std(X, axis=0)

# X_normalize = (X - feature_means) / feature_std

# print("\nNormalized Dataset")
# print(X_normalize)

def analyze_dataset(X):
    feature_means = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)

    X_centered = X - feature_means
    X_normalized = X_centered / feature_std

    return feature_means, feature_std, X_centered, X_normalized

feature_means, feature_std, X_centered, X_normalized = analyze_dataset(X)

print("\nFeature means:")
print(feature_means)

print("\nFeature standard deviation:")
print(feature_std)

print("\nCentered dataset:")
print(X_centered)

print("\nNormalized dataset:")
print(X_normalized)