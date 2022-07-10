import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import axes3d
import multiple_regression
import csv



Xy = []
with open("winequality-red.csv", mode ="r") as f:
    for row in csv.reader(f, delimiter = ";"):
        Xy.append(row)

label = Xy[0]
Xy = np.array(Xy[1:], dtype = np.float64)
np.random.shuffle(Xy)

train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
text_X = Xy[-1000:, :-1]
test_y = Xy[-1000:,-1]

model = multiple_regression.LinearRegression()
model.fit(train_X, train_y)
y = model.predict(text_X)

print(np.sqrt(((test_y -y)**2).mean()))


