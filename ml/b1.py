from pandas import read_csv
from numpy import unique, loadtxt

path = './oil-spill.csv'
df = read_csv(path, header=None)
print(df.nunique())

data = loadtxt(path, delimiter=',')
for i in range(data.shape[1]):
    num = len(unique(data[:, i]))
    percentage = float(num) / data.shape[0] * 100
    print('%d, %d, %.1f%%' % (i, num, percentage))

print("Columns with less than 1% unique values:")
for i in range(data.shape[1]):
    num = len(unique(data[:, i]))
    percentage = float(num) / data.shape[0] * 100
    if percentage < 1:
        print('%d, %d, %.1f%%' % (i, num, percentage))

print("Shape of dataset before dropping columns:")
print(df.shape)
counts = df.nunique()
to_del = [i for i, v in enumerate(counts) if (float(v) / df.shape[0] * 100) < 1]
df.drop(to_del, axis=1, inplace=True)
print("Shape of dataset after dropping columns:")
print(df.shape)

from numpy import arange
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot

data = df.values
X = data[:, :-1]
y = data[:, -1]

print("Shape of input and output data:")
print(X.shape, y.shape)

thresholds = arange(0.0, 0.55, 0.05)
results = []
for t in thresholds:
    transform = VarianceThreshold(threshold=t)
    X_sel = transform.fit_transform(X)
    n_features = X_sel.shape[1]
    print('>Threshold=%.2f, Features=%d' % (t, n_features))
    results.append(n_features)

pyplot.plot(thresholds, results)
pyplot.xlabel('Threshold')
pyplot.ylabel('Number of Features Selected')
pyplot.show()

dups = df.duplicated()
if dups.any():
    print(df[dups])

print("Shape of dataset before removing duplicates:")
print(df.shape)
df.drop_duplicates(inplace=True)
print("Shape of dataset after removing duplicates:")
print(df.shape)
