import numpy as np
import pandas as pd

data = pd.read_csv("ENJOYSPORT.csv")
print("Data:\n", data, "\n")

attributes = np.array(data)[:, :-1]
target = np.array(data)[:, -1]
print("Attributes:\n", attributes)
print("Target:\n", target)

def train(c, t):
    for i, val in enumerate(t):
        if val == "yes":
            specific_hypothesis = c[i].copy()
            break
    for i, val in enumerate(c):
        if t[i] == "yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
    return specific_hypothesis

print("\nThe final hypothesis is:", train(attributes, target))
