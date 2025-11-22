import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- HELPER FUNCTIONS (Unchanged) ---

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

def solve_lp_manual(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c
    
    max_iter = 100
    tol = 1e-9
    
    for _ in range(max_iter):
        obj_row = tableau[-1, :-1]
        min_val = np.min(obj_row)
        if min_val >= -tol: break
            
        pivot_col = np.argmin(obj_row)
        ratios = []
        for i in range(m):
            val = tableau[i, pivot_col]
            rhs = tableau[i, -1]
            if val > tol: ratios.append(rhs / val)
            else: ratios.append(np.inf)
        
        ratios = np.array(ratios)
        if np.all(ratios == np.inf): return None, None, False
            
        pivot_row = np.argmin(ratios)
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
                
    x_sol = np.zeros(n)
    for j in range(n):
        col = tableau[:, j]
        if np.sum(np.abs(col) > tol) == 1 and np.abs(np.sum(col) - 1.0) < tol:
            row_idx = np.where(np.abs(col - 1.0) < tol)[0][0]
            if row_idx < m: x_sol[j] = tableau[row_idx, -1]
                
    return x_sol, tableau[-1, -1], True

# --- MODIFIED SOLVER WITH HISTORY ---

def find_john_ellipsoid_trace(hull_points):
    """
    Same logic as before, but yields the state (center, radii, rotation) 
    at each step for animation.
    """
    hull_points = np.array(hull_points)
    N_points = len(hull_points)
    
    # 1. Build Constraints
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
        midpoint = (p1 + p2) / 2
        if np.dot(normal, centroid - midpoint) > 0:
            normal = -normal
        A_constr.append(normal)
        b_constr.append(np.dot(normal, p1))
        
    A_constr = np.array(A_constr)
    b_constr = np.array(b_constr)
    
    # 2. Initialization (Simplex)
    min_x, min_y = np.min(hull_points, axis=0)
    shift_x = abs(min_x) + 10.0
    shift_y = abs(min_y) + 10.0
    c_lp = np.array([0.0, 0.0, 1.0])
    A_lp = np.hstack([A_constr, np.ones((N_points, 1))])
    shifts = np.array([shift_x, shift_y])
    b_lp = b_constr + A_constr @ shifts
    
    x_sol, max_r, success = solve_lp_manual(c_lp, A_lp, b_lp)
    
    if success and x_sol[2] > 0:
        d_x = x_sol[0] - shift_x
        d_y = x_sol[1] - shift_y
        r_init = x_sol[2]
        d = np.array([d_x, d_y])
        B = np.eye(2) * (r_init * 0.9)
    else:
        d = centroid
        B = np.eye(2) * 0.01

    # List to store history frames
    history = []

    def save_frame(curr_d, curr_B):
        BTB = curr_B.T @ curr_B
        evals, evecs = np.linalg.eigh(BTB)
        radii = np.sqrt(np.abs(evals))
        rotation = evecs
        history.append((curr_d.copy(), radii.copy(), rotation.copy()))

    # Save initial state
    save_frame(d, B)

    # 3. Barrier Method Optimization
    t = 1.0
    mu = 5.0
    
    # Reduced iterations slightly for smoother animation length, 
    # but kept logic identical
    for outer in range(12): 
        for inner in range(20):
            
            # Save frame at start of iteration
            save_frame(d, B)

            Ba = A_constr @ B.T
            norms = np.linalg.norm(Ba, axis=1)
            slacks = b_constr - norms - A_constr @ d
            
            if np.any(slacks <= 1e-9):
                B *= 0.9
                continue
                
            grad_d = np.sum((A_constr.T / slacks).T, axis=0) / t
            
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
            
            step_d = -grad_d
            step_B = -grad_B
            
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

    # Save final state
    save_frame(d, B)
    return history

# ---------------- ANIMATION SETUP ----------------

# 1. Generate Data
#np.random.seed(42)
points = np.random.standard_normal((30, 2)) * 2
hull = graham_scan(points.tolist())
hull_array = np.array(hull)

# 2. Run Algorithm and Record History
print("Computing optimization trace...")
history = find_john_ellipsoid_trace(hull_array)
print(f"Optimization complete. Generated {len(history)} frames.")

# 3. Setup Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Gradient Descent: Finding John Ellipsoid")

# Static elements: Points and Hull
hull_plot = np.vstack([hull_array, hull_array[0]])
ax.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.4, label='Points')
ax.plot(hull_plot[:, 0], hull_plot[:, 1], 'k-', linewidth=2, label='Convex Hull')

# Dynamic elements: Ellipsoid and Center
ellipse_line, = ax.plot([], [], 'r-', linewidth=2, label='Current Ellipsoid')
center_point, = ax.plot([], [], 'rx', markersize=10, label='Center')
iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

ax.legend(loc='lower right')

# 4. Animation Update Function
theta = np.linspace(0, 2*np.pi, 100)

def update(frame_idx):
    # Retrieve state from history
    center, radii, rotation = history[frame_idx]
    
    # Reconstruct ellipse points
    ellipse_points = np.array([radii[0] * np.cos(theta), radii[1] * np.sin(theta)])
    ellipse_rotated = rotation @ ellipse_points
    ellipse_x = ellipse_rotated[0, :] + center[0]
    ellipse_y = ellipse_rotated[1, :] + center[1]
    
    # Update plot objects
    ellipse_line.set_data(ellipse_x, ellipse_y)
    center_point.set_data([center[0]], [center[1]]) # Pass as sequence
    
    iteration_text.set_text(f"Iteration: {frame_idx}/{len(history)}")
    
    return ellipse_line, center_point, iteration_text

# 5. Create Animation
# interval=50 means 50ms per frame (fast updates)
anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)

print("Displaying animation window...")
plt.show()

# Optional: Save to file (requires ffmpeg installed)
# anim.save('john_ellipsoid_descent.mp4', writer='ffmpeg', fps=30)
