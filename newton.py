import numpy as np
import matplotlib.pyplot as plt

class Newton:
    def __init__(self, f, df, eps= 1e-14, max_iter = 1000):
        self.f = f
        self.df = df
        self.eps = eps
        self.max_iter = max_iter
    
    def solve(self, x0):
        x = x0
        iter = 0
        self.path_ = x0.reshape(1,-1)

        while True:
            x_new = x - np.dot(np.linalg.inv(df(x)), self.f(x))
            self.path_ = np.r_[self.path_,x_new.reshape(1,-1)]
            if ((x-x_new)**2).sum() < (self.eps)**2:
                break
            x = x_new
            iter += 1
            if iter == self.max_iter:
                break
        return x_new

def f1(x,y):
    return x**3-2*y

def f2(x,y):
    return x**2+y**2-1

def f(xx):
    x = xx[0]
    y = xx[1]
    return np.array([f1(x,y), f2(x,y)])

def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([[3*x**2,-2],[2*x,2*y]])

xmin, xmax, ymin, ymax = -3,3,-3,3
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

xs = np.linspace(xmin, xmax, 300)
ys = np.linspace(ymin, ymax, 300)
xmesh, ymesh = np.meshgrid(xs, ys)

z1 = f1(xmesh, ymesh)
plt.contour(xmesh, ymesh, z1, colors = "r", levels = [0])

z2 = f2(xmesh, ymesh)
plt.contour(xmesh, ymesh, z2, colors = "b", levels = [0])

initials = [np.array([1,1]),np.array([-1,-1]), np.array([1,-1])]
markers = ["+", "*", "x"]
solver = Newton(f,df)

for x0, m in zip(initials, markers):
    sol = solver.solve(x0)
    plt.plot(solver.path_[:,0], solver.path_[:,1], color = "k", marker=m)
    print(sol)



plt.show()
