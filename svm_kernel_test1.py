import numpy as np
import matplotlib.pyplot as plt
import svm_kernel

plt.axes().set_aspect("equal")
np.random.seed()
X0 = np.random.randn(100,2)*2
X1 = np.random.randn(100,2)*2 + np.array([4,4])
y = np.array([1]*100 + [-1]*100)
X = np.r_[X0, X1]

model = svm_kernel.SVC()
model.fit(X, y)

xmin, xmax = X[:,0].min() , X[:,0].max()
ymin, ymax = X[:,1].min() , X[:,1].max()

plt.scatter(X0[:,0], X0[:,1], color="k", marker="*")
plt.scatter(X1[:,0], X1[:,1], color="k", marker="+")

xmesh, ymesh = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
Z = model.predict(np.c_[xmesh.ravel(), ymesh.ravel()]).reshape(xmesh.shape)
plt.contour(xmesh,ymesh, Z, levels=[0],colors="k")


plt.show()












