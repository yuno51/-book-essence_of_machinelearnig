import numpy as np
import matplotlib.pyplot as plt
import multiple_regression
import polynomial_regression

np.random.seed(0)

def f(x):
    return 2*x + 1

x = np.random.random(10) * 5
y = f(x) + np.random.randn(10)

model = polynomial_regression.PolynpmicalRegression(10)
model.fit(x,y)

plt.scatter(x,y,color = "k")
plt.ylim([y.min()-1, y.max()+1])
xx = np.linspace(x.min(), x.max(), 300)
yy = np.array([model.predict(u) for u in xx])
plt.plot(xx,yy ,color = "k")

model = multiple_regression.LinearRegression()
model.fit(x,y)

plt.plot([x.min(), x.max()], [f(x.min()), f(x.max())], color = "k", linestyle = "dashed")
plt.show()



