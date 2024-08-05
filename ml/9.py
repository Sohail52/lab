import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_regression(x_query, X_train, y_train, tau=0.1):
    m = X_train.shape[0]
    weights = np.exp(-np.sum((X_train - x_query) ** 2, axis=1) / (2 * tau * tau))
    W = np.diag(weights)
    theta = np.linalg.inv(X_train.T @ W @ X_train) @ (X_train.T @ W @ y_train)
    prediction = x_query @ theta
    return prediction

# Generate synthetic data for demonstration
np.random.seed(0)
X_train = np.linspace(0, 10, 50)
y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape[0])

# Query points
X_query = np.linspace(0, 10, 100)
tau = 0.5
predictions = []
for xq in X_query:
    x_query = np.array([1, xq])  # Adding bias term
    prediction = locally_weighted_regression(x_query, np.c_[np.ones(X_train.shape[0]), X_train], y_train, tau)
    predictions.append(prediction)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_query, predictions, color='red', label='Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.grid(True)
plt.show()
