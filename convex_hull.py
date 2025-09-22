import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def is_in_convex_hull(x1, x2, x3, w):
    """
    Check if point w is in the convex hull of x1, x2, x3 (all 2D points).
    Returns True if w is in the convex hull, False otherwise.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    w = np.asarray(w)
    A = np.column_stack([x1 - x3, x2 - x3])
    b = w - x3
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        l1, l2 = sol
        l3 = 1 - l1 - l2
        lambdas = np.array([l1, l2, l3])
        return np.all(lambdas >= 0) and np.all(lambdas <= 1)
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    # Test is_in_convex_hull
    x1 = np.array([0, 0])
    x2 = np.array([4, 0])
    x3 = np.array([2, 4])
    w = np.array([1, 2])
    print(is_in_convex_hull(x1, x2, x3, w))