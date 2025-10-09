import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def graham_scan(points):
    points = sorted(points, key=lambda p: (p[0], p[1]))
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


#np.random.seed(42)
points = np.random.standard_normal((1000, 2))
print("Points:\n", points)

hull = graham_scan(points.tolist())
hull_array = np.array(hull)

n_vertices = len(hull)

perimeter = sum(distance(hull[i], hull[(i+1) % len(hull)]) for i in range(len(hull)))

diameter = max(distance(hull[i], hull[j]) for i in range(len(hull)) for j in range(i+1, len(hull)))

N = 5
h = 0.01
x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))

dom = np.zeros(x.shape)
s1, s2 = x.shape

for i in range(s1):
    for j in range(s2):
        if point_in_polygon(x[i,j], y[i,j], hull):
            dom[i,j] = 1

cell_area = h * h
estimated_area = np.sum(dom) * cell_area

print(f"Number of vertices: {n_vertices}")
print(f"Estimated area: {estimated_area:.4f}")
print(f"Perimeter: {perimeter:.4f}")
print(f"Diameter: {diameter:.4f}")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.scatter(points[:, 0], points[:, 1], c='blue', s=50, alpha=0.6)
hull_plot = np.vstack([hull_array, hull_array[0]])
ax1.plot(hull_plot[:, 0], hull_plot[:, 1], 'r-', linewidth=2)
ax1.scatter(hull_array[:, 0], hull_array[:, 1], c='red', s=100, marker='s')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Convex Hull')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, dom, cmap='viridis')
ax2.set_title('Mesh Domain')

plt.tight_layout()
plt.show()
