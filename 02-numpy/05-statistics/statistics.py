import numpy as np

scores = np.array([90, 85, 78, 92, 88, 95, 80, 70, 75, 85])

print("Scores:", scores)

print("Mean:", np.mean(scores)) 
print("Median:", np.median(scores))
print("Standard Deviation:", np.std(scores))
print("Variance:", np.var(scores))
print("Minimum:", np.min(scores))
print("Maximum:", np.max(scores))

print("Mean + 10:", np.mean(scores) + 10)