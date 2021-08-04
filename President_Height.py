import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("US-presidents-heights.csv")

print(data.head(10))

height = np.array(data["height(cm)"])

print(height)
print("Mean of height = ", height.mean())
print("Standard Deviation of height = ", height.std())
print("Minimum height = ", height.min())
print("Maximum height = ", height.max())
print("Median = ", np.median(height))
print("25th Percentile = ", np.percentile(height,25))
print("75th Percentile = ", np.percentile(height,75))
sns.set()
plt.hist(height)
plt.title("Height Distribution of President of USA")
plt.xlabel("height(cm)")
plt.ylabel("Number")
plt.show()
