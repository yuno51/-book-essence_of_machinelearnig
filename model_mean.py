import numpy as np
import matplotlib.pyplot as plt
import multiple_regression
import polynomial_regression
import warnings

def f(x):
    return 1/(x + 1)

def sample(n):
    x = np.random.random(n) * 5
    y = f(x)
    return x,y

xx = np.arange(0,5,0.01)
np.random.seed(0)

y_poly_sum = np.zeros(len(xx))
y_lin_sum = np.zeros(len(xx))

n = 1000
warnings.filterwarnings("ignore")

for _ in range(n):
    x,y = sample(5)
    poly = polynomial_regression.PolynpmicalRegression(4)
    poly.fit(x,y)

    lin = multiple_regression.LinearRegression()
    lin.fit(x,y)

    y_poly = poly.predict(xx)
    y_poly_sum += y_poly
    y_lin = lin.predict(xx.reshape(-1,1))
    y_lin_sum += y_lin

plt.plot(xx, f(xx), label = "truth", color = "k", linestyle = "solid")
plt.plot(xx, y_poly_sum/n , label = "poly", linestyle = "dotted")
plt.plot(xx, y_lin_sum/n, label = "linear", linestyle = "dashed")
plt.legend()
plt.show()





