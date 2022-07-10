import numpy as np
import matplotlib.pyplot as plt
import kmeans

#np.random.seed(0)

point1 = np.random.randn(50,2)
point2 = np.random.randn(50,2) + np.array([3,0])
point3 = np.random.randn(50,2) + np.array([3,3])

points = np.r_[point1, point2, point3]
np.random.shuffle(points)

model = kmeans.kMeans(3)
model.fit(points)

markers = ["+","*","o"]

for i in range(3):
    p = points[model.labels_ == i,:]
    plt.scatter(p[:,0], p[:,1], marker=markers[i])

plt.show()











