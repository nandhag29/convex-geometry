import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graham_scan(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = sorted(set(map(tuple, points)), key=lambda p: (p[0], p[1]))
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

N = 5
h = 0.01
x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))

square_vertices = [[0, 0], [3, 0], [3, 3], [0, 3]]
hull = graham_scan(square_vertices)

dom = np.zeros(x.shape)
s1, s2 = x.shape

for i in range(s1):
    for j in range(s2):
        if point_in_polygon((x[i,j], y[i,j]), hull):
            dom[i,j] = 1

cell_area = h * h
estimated_area = np.sum(dom) * cell_area
print(f"Estimated area: {estimated_area}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, dom, cmap='viridis')
plt.show()
