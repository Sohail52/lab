import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = load_wine()
data, target = wine.data, wine.target
print(wine)

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

model = svm.SVC(kernel='linear')
model.fit(train_data, train_target)

predictions = model.predict(test_data)
accuracy = accuracy_score(test_target, predictions)
print(f'Test accuracy: {accuracy}')

sample_index = 0
new_sample = test_data[sample_index].reshape(1, -1)
predicted_label = model.predict(new_sample)
actual_label = test_target[sample_index]

print(f'Predicted label: {predicted_label[0]}, Actual label: {actual_label}')

feature_names = wine.feature_names
target_names = wine.target_names

plt.figure(figsize=(10, 6))
plt.barh(feature_names, test_data[sample_index])
plt.title(f'Predicted: {target_names[predicted_label[0]]}, Actual: {target_names[actual_label]}')
plt.xlabel('Scaled Feature Values')
plt.show()
