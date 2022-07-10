import logistic_regression
import csv
import numpy as np


n_test = 100
X = []
y = []

with open("wdbc.data", mode = "r") as f:
    for row in csv.reader(f):
        if row[1] == "B":
            y.append(0)
        else:
            y.append(1)
        X.append(row[2:])

X = np.array(X, dtype =np.float64)
y = np.array(y, dtype =np.float64)
X_train = X[:-n_test]
y_train = y[:-n_test]
X_test = X[-n_test:]
y_test = y[-n_test:]

model = logistic_regression.LogisticRegression(tol = 0.01)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
n_hits = (y_test == y_predict).sum()
print("hits:{}/{}".format(n_hits,n_test))
