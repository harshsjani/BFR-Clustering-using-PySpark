import bfr
import matplotlib.pyplot as plt

points = [(4, 10), (7, 10), (4, 8), (6, 8), (10, 5), (12, 6), (11, 4), (3, 4), (2, 2), (5, 2), (9, 3), (12, 3)]
x = [p[0] for p in points]
y = [p[1] for p in points]
plt.scatter(x, y)
# plt.show()
print(bfr.HCluster.get_centroids(points, 3))
