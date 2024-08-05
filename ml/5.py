import numpy as np
import pandas as pd

data = pd.read_csv("play_tennis.csv")
data.drop(["day"], axis=1, inplace=True)

def naive_bayes_predict(data, target_name, test_instance):
    value_counts = data[target_name].value_counts().to_dict()
    pvalue = (data[data.columns[data.shape[1]-1]].value_counts() / data.shape[0]).to_dict()
    pred = {}
    for target, _ in value_counts.items():
        attribute = 1
        subset = data[data[target_name] == target]
        for attr, value in test_instance.items():
            cond_prob = subset[subset[attr] == value].shape[0] / subset.shape[0]
            attribute *= cond_prob
        prob = pvalue[target] * attribute       
        pred[target] = prob
    return pred

test_instance = {"outlook": "Sunny", "temp": "Cool", "humidity": "High", "wind": "Strong"}
prediction = naive_bayes_predict(data.copy(), data.columns[data.shape[1]-1], test_instance)
print("Prediction is:", prediction)
