import numpy as np
import matplotlib.pyplot as plt

def simple_reg(x,y):
    n = len(x)
    a = (np.dot(x,y) -1/n * x.sum() * y.sum()) / (np.dot(x,x) - 1/n * (x.sum())**2)
    b = 1/n * (y.sum() - a* x.sum())

    return a,b


x = np.array([1,2,4,6,7])
y = np.array([1,3,3,5,4])

a,b = simple_reg(x,y)

xmax = x[-1] 
plt.scatter(x,y, color = "k")
plt.plot([0, xmax], [b, a* xmax + b], color = "k")
plt.show()