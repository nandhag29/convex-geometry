import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

def plot_convex_hull(points):
    """
    Given a 2xN array of 2D points, plot the points and their convex hull.
    Uses Caratheodory's theorem: in 2D, any point in the convex hull can be written as a convex combination of at most 3 points.
    """
    points = np.asarray(points)
    if points.shape[0] != 2:
        raise ValueError("Input must be a 2xN array of 2D points.")
    points = points.T  # shape (N, 2) for ConvexHull

    hull = ConvexHull(points)
    plt.figure(figsize=(6,6))
    plt.plot(points[:,0], points[:,1], 'o', label='Points')

    # Draw convex hull
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    hull_pts = points[hull.vertices]
    idx = np.random.choice(len(hull_pts), 3, replace=False)
    lambdas = np.random.rand(3)
    lambdas /= lambdas.sum()
    caratheodory_point = np.dot(lambdas, hull_pts[idx])

    plt.plot(*caratheodory_point, 'rx', markersize=12, label="Caratheodory point")
    plt.legend()
    plt.title("Convex Hull of 2D Points (Caratheodory's theorem)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example: 2xN array
    points = np.random.rand(2, 10)
    plot_convex_hull(points)