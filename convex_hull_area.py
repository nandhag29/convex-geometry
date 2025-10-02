import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 5
h = 0.01
x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))

square_vertices = np.array([[-2, 0], [3, 0], [0, 3], [3, 3]])

dom = np.zeros(x.shape)
s1, s2 = x.shape

x_min, x_max = -2, 3
y_min, y_max = 0, 4

for i in range(s1):
    for j in range(s2):
        if x_min <= x[i,j] <= x_max and y_min <= y[i,j] <= y_max:
            dom[i,j] = 1

cell_area = h * h
estimated_area = np.sum(dom) * cell_area
print(f"Estimated area: {estimated_area}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, dom, cmap='viridis')
plt.show()
