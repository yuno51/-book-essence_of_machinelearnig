import matplotlib.pyplot as plt
import numpy as np
import sympy

def f1(x,y):
    return 5*x**2+6*x*y+5*y**2-26*x-26*y

def f2(x,y):
    return 5*x**2+6*x*y+5*y**2-16*x-16*y

def g(x,y):
    return x**2+y**2-4


def dL1(xx):
    x = xx[0]
    y = xx[1]
    λ = xx[2]
    return np.array([10*x+6*y-26+2*λ*x,6*x+10*y-26+2*λ*x, x**2+y**2-4])

def dL1_lamda_minus(xx):
    x = xx[0]
    y = xx[1]
    return np.array([10*x+6*y-26,6*x+10*y-26])

def dL2(xx):
    x = xx[0]
    y = xx[1]
    λ = xx[2]
    return np.array([10*x+6*y-16+2*λ*x,6*x+10*y-16+2*λ*x, x**2+y**2-4])

def dL2_lamda_minus(xx):
    x = xx[0]
    y = xx[1]
    return np.array([10*x+6*y-16,6*x+10*y-16])

def plot_f(f,g,dL,dL_lamda_minus):
    xs = np.linspace(-3,6,300)
    ys = np.linspace(-3,6,300)
    xmesh , ymesh = np.meshgrid(xs, ys)

    zf = f(xmesh, ymesh)
    zg = g(xmesh, ymesh)
    plt.contourf(xmesh, ymesh, zg, colors = "b", alpha = 0.6, levels = [-10,0])
    plt.contour(xmesh, ymesh, zf, colors = "r", levels = [-42,-40,-30,-20,-10,0,10,20])

    x, y, λ = sympy.symbols("x y λ")    
    res = sympy.solve(dL([x, y, λ]), [x, y, λ])

    λs = np.array([i[2] for i in np.array(res)])
    judge = λs[λs > 0]

    if len(judge) > 0:
        for point in res:
            if point[2] > 0:
                print(point[0], point[1], f1(point[0], point[1]))
                plt.scatter(point[0], point[1], color = "k")
                plt.contour(xmesh, ymesh, zf1, colors = "r", levels = [f(point[0], point[1])])
    else:
        x, y = sympy.symbols("x y") 
        minus = sympy.solve(dL_lamda_minus([x, y]), [x, y])
        print(minus[x], minus[y], f(minus[x], minus[y]))
        plt.scatter(minus[x], minus[y], color = "k")
            
    plt.axes().set_aspect('equal')
    plt.show()

plot_f(f2,g,dL2, dL2_lamda_minus)

