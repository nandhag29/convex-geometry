import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# No scipy imports for the manual function!

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

def solve_lp_manual(c, A, b):
    """
    Solve Linear Program manually using the Simplex Method (Tableau).
    Maximize c^T x subject to A x <= b, x >= 0.
    
    Returns: x_optimal, max_val, success
    """
    m, n = A.shape
    
    # Create Tableau
    # Rows: m constraints + 1 objective
    # Cols: n variables + m slacks + 1 RHS
    tableau = np.zeros((m + 1, n + m + 1))
    
    # Fill A
    tableau[:m, :n] = A
    
    # Fill Slacks (Identity)
    tableau[:m, n:n+m] = np.eye(m)
    
    # Fill RHS
    tableau[:m, -1] = b
    
    # Fill Objective Row (Maximize c^T x  =>  Z - c^T x = 0)
    # We put -c in the bottom row
    tableau[-1, :n] = -c
    
    # Simplex Iterations
    max_iter = 100
    tol = 1e-9
    
    for _ in range(max_iter):
        # 1. Identify Entering Variable (Most negative in bottom row)
        # Only look at non-basic variables (first n+m columns)
        obj_row = tableau[-1, :-1]
        min_val = np.min(obj_row)
        
        if min_val >= -tol:
            # Optimality reached (no negative coefficients)
            break
            
        pivot_col = np.argmin(obj_row)
        
        # 2. Identify Leaving Variable (Minimum Ratio Test)
        # Ratio = RHS / Column_Val, for Column_Val > 0
        ratios = []
        for i in range(m):
            val = tableau[i, pivot_col]
            rhs = tableau[i, -1]
            if val > tol:
                ratios.append(rhs / val)
            else:
                ratios.append(np.inf)
        
        ratios = np.array(ratios)
        if np.all(ratios == np.inf):
            # Unbounded
            return None, None, False
            
        pivot_row = np.argmin(ratios)
        
        # 3. Pivot Operation
        pivot_val = tableau[pivot_row, pivot_col]
        
        # Normalize pivot row
        tableau[pivot_row, :] /= pivot_val
        
        # Eliminate other rows
        for i in range(m + 1):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
                
    # Extract Solution
    # Variables are basic if column is unit vector
    x_sol = np.zeros(n)
    for j in range(n):
        col = tableau[:, j]
        # Check if unit vector (one 1, rest 0s)
        if np.sum(np.abs(col) > tol) == 1 and np.abs(np.sum(col) - 1.0) < tol:
            # Find the row index of the 1
            row_idx = np.where(np.abs(col - 1.0) < tol)[0][0]
            if row_idx < m: # Usually basic variables are in constraint rows
                x_sol[j] = tableau[row_idx, -1]
                
    return x_sol, tableau[-1, -1], True

def find_john_ellipsoid(hull_points):
    """
    Find maximum volume inscribed ellipsoid using fully MANUAL implementation.
    1. Manual Linear Programming (Simplex) for Chebyshev Center Initialization.
    2. Manual Barrier Method for Ellipsoid Optimization.
    """
    hull_points = np.array(hull_points)
    N_points = len(hull_points)
    n_dim = 2

    # --- Step 1: Build Constraints ---
    # a_i^T x <= b_i
    A_constr = []
    b_constr = []
    centroid = np.mean(hull_points, axis=0)
    
    for i in range(N_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % N_points]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9: continue
        normal = normal / norm_len
        
        # Ensure outward normal
        midpoint = (p1 + p2) / 2
        if np.dot(normal, centroid - midpoint) > 0:
            normal = -normal
            
        A_constr.append(normal)
        b_constr.append(np.dot(normal, p1))
        
    A_constr = np.array(A_constr)
    b_constr = np.array(b_constr)
    
    # --- Step 2: Manual LP for Initialization (Chebyshev Center) ---
    # Maximize r
    # s.t. a_i^T d + r <= b_i
    # Variables: d_x, d_y, r
    
    # To use Simplex (x >= 0), we shift variables.
    # d_x = x_pos - x_shift
    # d_y = y_pos - y_shift
    # We set shifts based on bounding box to ensure x_pos, y_pos > 0
    min_x, min_y = np.min(hull_points, axis=0)
    shift_x = abs(min_x) + 10.0
    shift_y = abs(min_y) + 10.0
    
    # Transformed Variables: X = [d_x', d_y', r]  (all >= 0)
    # Actual d_x = d_x' - shift_x
    # Actual d_y = d_y' - shift_y
    
    # Constraint Substitution:
    # a_x (d_x' - sx) + a_y (d_y' - sy) + r <= b
    # a_x d_x' + a_y d_y' + r <= b + a_x sx + a_y sy
    
    # LP Matrix Construction
    # c = [0, 0, 1] (Maximize r)
    c_lp = np.array([0.0, 0.0, 1.0])
    
    # A_lp rows are [a_x, a_y, 1]
    A_lp = np.hstack([A_constr, np.ones((N_points, 1))])
    
    # b_lp = b + A_constr @ shifts
    shifts = np.array([shift_x, shift_y])
    b_lp = b_constr + A_constr @ shifts
    
    # Solve Manual Simplex
    x_sol, max_r, success = solve_lp_manual(c_lp, A_lp, b_lp)
    
    if success and x_sol[2] > 0:
        d_x = x_sol[0] - shift_x
        d_y = x_sol[1] - shift_y
        r_init = x_sol[2]
        
        d = np.array([d_x, d_y])
        # Start with safe interior radius (90% of max)
        B = np.eye(2) * (r_init * 0.9)
    else:
        # Fallback (should not happen for valid polygons)
        d = centroid
        B = np.eye(2) * 0.01

    # --- Step 3: Barrier Method Optimization ---
    t = 1.0
    mu = 5.0
    
    for outer in range(15):
        for inner in range(30):
            # Slacks
            Ba = A_constr @ B.T
            norms = np.linalg.norm(Ba, axis=1)
            slacks = b_constr - norms - A_constr @ d
            
            if np.any(slacks <= 1e-9):
                B *= 0.9
                continue
                
            # Gradients
            # Grad_d
            grad_d = np.sum((A_constr.T / slacks).T, axis=0) / t
            
            # Grad_B
            try:
                B_invT = np.linalg.inv(B).T
            except:
                break
            grad_B = -B_invT
            
            valid = norms > 1e-9
            if np.any(valid):
                w = 1.0 / (t * slacks[valid] * norms[valid])
                term = (w.reshape(-1,1) * Ba[valid]).T @ A_constr[valid]
                grad_B += term
            
            # Update Steps
            step_d = -grad_d
            step_B = -grad_B
            
            # Backtracking
            alpha = 1.0
            beta = 0.5
            accepted = False
            
            for _ in range(10):
                d_new = d + alpha * step_d
                B_new = B + alpha * step_B
                
                Ba_new = A_constr @ B_new.T
                slacks_new = b_constr - np.linalg.norm(Ba_new, axis=1) - A_constr @ d_new
                
                if np.all(slacks_new > 0):
                    d = d_new
                    B = B_new
                    accepted = True
                    break
                alpha *= beta
            
            if not accepted:
                break
        t *= mu

    # Extract Results
    BTB = B.T @ B
    evals, evecs = np.linalg.eigh(BTB)
    radii = np.sqrt(np.abs(evals))
    rotation = evecs
    center = d
    
    return center, radii, rotation


# ---------------- MAIN EXECUTION ----------------
#np.random.seed(10)
# Generate a clear convex shape
points = np.random.standard_normal((30, 2)) * 2
hull = graham_scan(points.tolist())
hull_array = np.array(hull)

# Compute Ellipsoid
print("Computing Manual John Ellipsoid (Simplex Init)...")
center, radii, rotation = find_john_ellipsoid(hull_array)
print(f"Center: {center}")
print(f"Radii: {radii}")

# Plotting
theta = np.linspace(0, 2*np.pi, 100)
ellipse_points = np.array([radii[0] * np.cos(theta), radii[1] * np.sin(theta)])
ellipse_rotated = rotation @ ellipse_points
ellipse_x = ellipse_rotated[0, :] + center[0]
ellipse_y = ellipse_rotated[1, :] + center[1]

# Mesh for Area
N = 5
h = 0.1
x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))
dom = np.zeros(x.shape)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if point_in_polygon(x[i,j], y[i,j], hull):
            dom[i,j] = 1
estimated_area = np.sum(dom) * h * h

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
ax1.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.4)
hull_plot = np.vstack([hull_array, hull_array[0]])
ax1.plot(hull_plot[:, 0], hull_plot[:, 1], 'k-', linewidth=2)
ax1.plot(ellipse_x, ellipse_y, 'r-', linewidth=2, label='Manual John Ellipsoid')
ax1.scatter(center[0], center[1], c='red', marker='x', s=100)
ax1.set_aspect('equal')
ax1.legend()
ax1.set_title('John Ellpsoid')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, dom, cmap='viridis', alpha=0.7)
ax2.set_title(f'Area ~ {estimated_area:.2f}')

plt.tight_layout()
plt.show()
