import numpy as np
import matplotlib.pyplot as plt
import svm_hard

plt.axes().set_aspect("equal")
np.random.seed(0)
X0 = np.random.randn(40,2)
X1 = np.random.randn(40,2) + np.array([5,5])
y = np.array([1]*40 + [-1]*40)
X = np.r_[X0, X1]

model = svm_hard.SVC()
model.fit(X,y)
plt.scatter(X0[:,0], X0[:,1], color="k", marker="+")
plt.scatter(X1[:,0], X1[:,1], color="k", marker="*")

def f(model, x):
    return (-model.w0_ - model.w_[0] *x) /model.w_[1]

x1 = -2
x2 = 7

plt.plot([x1,x2], [f(model,x1),f(model,x2)], color="k")

plt.show()





