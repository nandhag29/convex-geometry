import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

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

def find_john_ellipsoid(hull_points):
    """
    Find maximum volume inscribed ellipsoid using a Fully Manual Implementation.
    - Initialization: Grid search for deepest interior point (replaces linprog).
    - Optimization: Barrier Method with Gradient Descent and Backtracking.
    """
    hull_points = np.array(hull_points)
    N_points = len(hull_points)
    n_dim = 2

    # 1. Compute Constraints: a_i^T x <= b_i
    # For ellipsoid E(B, d), constraint is ||B a_i|| + a_i^T d <= b_i
    A = []
    b = []
    centroid = np.mean(hull_points, axis=0)
    
    for i in range(N_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % N_points]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-9: continue 
        normal = normal / norm_len
        
        # Ensure normal points OUTWARD relative to centroid
        midpoint = (p1 + p2) / 2
        if np.dot(normal, centroid - midpoint) > 0:
            normal = -normal
            
        A.append(normal)
        b.append(np.dot(normal, p1))
        
    A = np.array(A)
    b = np.array(b)
    
    # 2. Manual Initialization (Grid Search for Chebyshev Center)
    # Find a point d deep inside the polygon to ensure feasibility.
    
    min_x, min_y = np.min(hull_points, axis=0)
    max_x, max_y = np.max(hull_points, axis=0)
    
    # Create a coarse grid of candidate points
    grid_res = 15
    gx = np.linspace(min_x, max_x, grid_res)
    gy = np.linspace(min_y, max_y, grid_res)
    grid_x, grid_y = np.meshgrid(gx, gy)
    
    # Candidates: Grid points + Centroid
    candidates = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    candidates = np.vstack([candidates, centroid])
    
    # Calculate "Depth" (minimum slack) for all candidates
    # Slack = b - A @ x. If min(slack) > 0, point is inside.
    slacks_matrix = b - candidates @ A.T
    min_slacks = np.min(slacks_matrix, axis=1)
    
    valid_mask = min_slacks > 0
    if np.any(valid_mask):
        best_idx = np.argmax(min_slacks)
        best_d = candidates[best_idx]
        best_r = min_slacks[best_idx]
    else:
        # Should not happen for a valid polygon, but fallback just in case
        best_d = centroid
        best_r = 1e-3

    # Start with a circle at the deep center
    # Slightly shrink radius to be safely strictly interior
    d = best_d
    B = np.eye(n_dim) * (best_r * 0.85)
    
    # 3. Optimization via Barrier Method (Manual Gradient Descent)
    t = 1.0         # Barrier parameter
    mu = 4.0        # Barrier update factor
    max_outer = 15
    max_inner = 30
    
    for outer in range(max_outer):
        for inner in range(max_inner):
            # Compute Slacks
            Ba = A @ B.T
            norms = np.linalg.norm(Ba, axis=1)
            slacks = b - norms - A @ d
            
            # Emergency check
            if np.any(slacks <= 1e-9):
                B *= 0.8 # Shrink to recover
                continue
            
            # --- Compute Gradients of Objective Function ---
            # Obj = -log det(B) - (1/t) * sum( log(slacks) )
            
            # Gradient w.r.t d
            # Grad_d = (1/t) * sum( a_i / slack_i )
            grad_d = np.sum((A.T / slacks).T, axis=0) / t
            
            # Gradient w.r.t B
            # Grad_B = -B^-T + (1/t) * sum( (Ba_i * a_i^T) / (slack_i * ||Ba_i||) )
            try:
                B_invT = np.linalg.inv(B).T
            except:
                break
            
            grad_B = -B_invT
            
            # Vectorized computation for the sum term
            # Only consider constraints with non-zero norm impact
            valid_n = norms > 1e-9
            if np.any(valid_n):
                # Weights: 1 / (t * slack * norm)
                w = 1.0 / (t * slacks[valid_n] * norms[valid_n])
                w = w.reshape(-1, 1)
                
                # Term: w_i * (Ba_i) outer (a_i)
                # Can be computed as (W * Ba)^T @ A
                scaled_Ba = w * Ba[valid_n] # Shape (m_valid, 2)
                grad_B += scaled_Ba.T @ A[valid_n]

            # --- Descent Step with Backtracking ---
            step_d = -grad_d
            step_B = -grad_B
            
            alpha = 1.0
            beta = 0.6
            
            # Current Barrier Objective Value (for monitoring descent)
            # obj_val = -np.linalg.slogdet(B)[1] - (1/t) * np.sum(np.log(slacks))
            
            accepted = False
            for _ in range(12): # Line search
                d_new = d + alpha * step_d
                B_new = B + alpha * step_B
                
                # Check Feasibility First
                Ba_new = A @ B_new.T
                norms_new = np.linalg.norm(Ba_new, axis=1)
                slacks_new = b - norms_new - A @ d_new
                
                if np.all(slacks_new > 0):
                    # Feasible. We assume descent direction is good and accept.
                    d = d_new
                    B = B_new
                    accepted = True
                    break
                alpha *= beta
            
            if not accepted:
                break # Line search failed, likely converged for this t
                
        t *= mu # Increase barrier stiffness

    # Extract Ellipsoid Parameters
    BTB = B.T @ B
    evals, evecs = np.linalg.eigh(BTB)
    radii = np.sqrt(np.abs(evals))
    rotation = evecs
    center = d
    
    return center, radii, rotation

def find_john_ellipsoid_scipy(hull_points):
    """
    Library implementation using Scipy minimize for comparison.
    """
    hull_points = np.array(hull_points)
    N = len(hull_points)
    A = []
    b = []
    centroid = np.mean(hull_points, axis=0)
    for i in range(N):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % N]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, centroid - (p1+p2)/2) > 0: normal = -normal
        A.append(normal)
        b.append(np.dot(normal, p1))
    A = np.array(A)
    b = np.array(b)

    # Initialization
    x0 = np.concatenate([np.eye(2).flatten() * 0.1, centroid])
    
    def obj(x):
        B = x[:4].reshape(2, 2)
        s, ld = np.linalg.slogdet(B)
        return -ld if s > 0 else 1e9
        
    def cons(x):
        B = x[:4].reshape(2, 2)
        d = x[4:]
        return b - np.linalg.norm(A @ B.T, axis=1) - A @ d
        
    res = minimize(obj, x0, constraints={'type':'ineq','fun':cons}, method='SLSQP', tol=1e-6)
    B = res.x[:4].reshape(2,2)
    d = res.x[4:]
    evals, evecs = np.linalg.eigh(B.T @ B)
    return d, np.sqrt(evals), evecs

# ---------------- MAIN EXECUTION ----------------

np.random.seed(42)
points = np.random.standard_normal((40, 2)) * 1.5
hull = graham_scan(points.tolist())
hull_array = np.array(hull)
n_vertices = len(hull)

# Mesh Domain for 3D Plot
N = 4
h = 0.05
x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))
dom = np.zeros(x.shape)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if point_in_polygon(x[i,j], y[i,j], hull):
            dom[i,j] = 1
estimated_area = np.sum(dom) * h * h

# 1. Compute Manual John Ellipsoid
print("Computing Manual John Ellipsoid...")
center_man, radii_man, rotation_man = find_john_ellipsoid(hull_array)
vol_man = np.pi * radii_man[0] * radii_man[1]

# 2. Compute Library John Ellipsoid
print("Computing Library John Ellipsoid...")
center_lib, radii_lib, rotation_lib = find_john_ellipsoid_scipy(hull_array)
vol_lib = np.pi * radii_lib[0] * radii_lib[1]

print(f"\nManual Volume: {vol_man:.4f}")
print(f"Library Volume: {vol_lib:.4f}")

# Plotting
theta = np.linspace(0, 2*np.pi, 100)

# Manual Ellipsoid Points
pts_man = np.array([radii_man[0] * np.cos(theta), radii_man[1] * np.sin(theta)])
pts_man = rotation_man @ pts_man
x_man = pts_man[0, :] + center_man[0]
y_man = pts_man[1, :] + center_man[1]

# Library Ellipsoid Points
pts_lib = np.array([radii_lib[0] * np.cos(theta), radii_lib[1] * np.sin(theta)])
pts_lib = rotation_lib @ pts_lib
x_lib = pts_lib[0, :] + center_lib[0]
y_lib = pts_lib[1, :] + center_lib[1]

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121)
ax1.scatter(points[:, 0], points[:, 1], c='gray', s=30, alpha=0.4)
hull_plot = np.vstack([hull_array, hull_array[0]])
ax1.plot(hull_plot[:, 0], hull_plot[:, 1], 'k-', linewidth=2, label='Convex Hull')

ax1.plot(x_man, y_man, 'r-', linewidth=2.5, label='Manual (Grid Init)')
ax1.scatter(center_man[0], center_man[1], c='red', s=80, marker='x')

ax1.plot(x_lib, y_lib, 'b--', linewidth=2, label='Scipy Library')

ax1.legend()
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('John Ellipsoid Comparison')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, dom, cmap='viridis', alpha=0.8)
ax2.set_title(f'Polygon Area ~ {estimated_area:.2f}')

plt.tight_layout()
plt.show()
