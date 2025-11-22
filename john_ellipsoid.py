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


def find_john_ellipsoid(hull_points, max_iter=1000, tol=1e-8):
    """
    Find maximum volume inscribed ellipsoid in a 2D convex polygon.
    
    The ellipsoid is E = {Bu + d | ||u|| <= 1}
    We maximize log det(B) subject to ||B*a_i|| + a_i^T*d <= b_i
    
    Parameters:
    -----------
    hull_points : array-like, shape (N, 2)
        Vertices of convex hull
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance for convergence
        
    Returns:
    --------
    center : ndarray, shape (2,)
        Center of ellipsoid
    radii : ndarray, shape (2,)
        Semi-axes lengths
    rotation : ndarray, shape (2, 2)
        Rotation matrix
    """
    hull_points = np.array(hull_points)
    N = len(hull_points)
    n = 2  # dimension
    
    # Compute constraint normals (a_i) and offsets (b_i)
    # For each edge, compute outward normal
    A = []
    b = []
    centroid = np.mean(hull_points, axis=0)
    
    for i in range(N):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % N]
        
        # Edge vector
        edge = p2 - p1
        # Perpendicular (potential normal)
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Make sure normal points outward
        midpoint = (p1 + p2) / 2
        if np.dot(normal, centroid - midpoint) > 0:
            normal = -normal
        
        A.append(normal)
        b.append(np.dot(normal, p1))
    
    A = np.array(A)  # Shape: (m, n) where m = number of constraints
    b = np.array(b)  # Shape: (m,)
    
    # Initialize: B as identity scaled down, d as centroid
    B = np.eye(n) * 0.1
    d = centroid.copy()
    
    # Gradient descent with backtracking line search
    alpha = 0.01  # Initial step size
    beta = 0.5    # Step size reduction factor
    
    for iteration in range(max_iter):
        # Check feasibility: ||B*a_i|| + a_i^T*d <= b_i for all i
        Ba = A @ B.T  # Shape: (m, n)
        norms = np.linalg.norm(Ba, axis=1)  # Shape: (m,)
        Ad = A @ d  # Shape: (m,)
        constraints = norms + Ad - b  # Should be <= 0
        
        max_violation = np.max(constraints)
        
        # Compute objective: log det(B)
        try:
            sign, logdet = np.linalg.slogdet(B)
            if sign <= 0:
                # B is not positive definite, reduce scale
                B = B * 0.5
                continue
            obj = logdet
        except:
            B = B * 0.5
            continue
        
        # Compute gradients
        # Gradient of log det(B) w.r.t B is B^{-T}
        try:
            B_inv = np.linalg.inv(B)
        except:
            B = np.eye(n) * 0.05
            continue
            
        grad_B = B_inv.T
        
        # Gradient of constraints w.r.t. B and d
        # For constraint i: ||B*a_i|| + a_i^T*d - b_i
        # Gradient w.r.t B: (B*a_i/||B*a_i||) * a_i^T
        # Gradient w.r.t d: a_i
        
        # Use barrier method: add log barriers for constraints
        # Modified objective: log det(B) - (1/t) * sum(log(-constraint_i))
        t = 10.0 * (iteration + 1)  # Increase barrier parameter
        
        grad_B_barrier = grad_B.copy()
        grad_d_barrier = np.zeros(n)
        
        for i in range(len(A)):
            if constraints[i] < -tol:
                # Constraint is satisfied
                barrier_grad_i = 1.0 / (-constraints[i])
                
                # Gradient of constraint i w.r.t. B
                if norms[i] > 1e-10:
                    grad_constraint_B = np.outer(Ba[i] / norms[i], A[i])
                else:
                    grad_constraint_B = np.zeros((n, n))
                
                # Gradient of constraint i w.r.t. d
                grad_constraint_d = A[i]
                
                # Update gradients with barrier
                grad_B_barrier -= (1.0 / t) * barrier_grad_i * grad_constraint_B
                grad_d_barrier -= (1.0 / t) * barrier_grad_i * grad_constraint_d
            else:
                # Constraint violated, project back to feasible
                # Move d toward satisfying constraint
                if constraints[i] > 0:
                    d = d - (constraints[i] + 0.01) * A[i]
                B = B * 0.95
                break
        
        # Update with backtracking line search
        step = alpha
        for _ in range(20):
            B_new = B + step * grad_B_barrier
            d_new = d + step * grad_d_barrier
            
            # Check if new point is feasible
            Ba_new = A @ B_new.T
            norms_new = np.linalg.norm(Ba_new, axis=1)
            Ad_new = A @ d_new
            constraints_new = norms_new + Ad_new - b
            
            if np.max(constraints_new) < -tol:
                # Check if objective improved
                try:
                    sign_new, logdet_new = np.linalg.slogdet(B_new)
                    if sign_new > 0 and logdet_new > obj:
                        B = B_new
                        d = d_new
                        break
                except:
                    pass
            
            step *= beta
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_B_barrier) + np.linalg.norm(grad_d_barrier)
        if grad_norm < tol:
            break
    
    # Extract ellipsoid parameters from B
    # The ellipsoid is {x | (x-d)^T (B^T B)^{-1} (x-d) <= 1}
    # Which is equivalent to {x | (x-d)^T A (x-d) <= 1} where A = (B^T B)^{-1}
    
    try:
        # Compute A = (B^T B)^{-1} = B^{-1} (B^T)^{-1}
        BTB = B.T @ B
        eigenvalues, eigenvectors = np.linalg.eigh(BTB)
        
        # Semi-axes are square roots of eigenvalues of B^T B
        radii = np.sqrt(np.abs(eigenvalues))
        rotation = eigenvectors
        
        center = d
    except:
        # Fallback
        center = centroid
        radii = np.array([0.1, 0.1])
        rotation = np.eye(2)
    
    return center, radii, rotation


#np.random.seed(42)
points = np.random.standard_normal((100, 2))
#points = np.array([[-1, -1], [-1,  1], [ 1, -1], [ 1,  1]])
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

# Compute John ellipsoid
print("\nComputing John ellipsoid...")
center, radii, rotation = find_john_ellipsoid(hull_array)
print(f"Ellipsoid center: {center}")
print(f"Ellipsoid radii: {radii}")

ellipsoid_volume = np.pi * radii[0] * radii[1]
print(f"John ellipsoid volume: {ellipsoid_volume:.4f}")

# Generate ellipsoid points for plotting
theta = np.linspace(0, 2*np.pi, 100)
ellipse_points = np.array([radii[0] * np.cos(theta), radii[1] * np.sin(theta)])
ellipse_rotated = rotation @ ellipse_points
ellipse_x = ellipse_rotated[0, :] + center[0]
ellipse_y = ellipse_rotated[1, :] + center[1]


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

# Plot John ellipsoid
ax1.plot(ellipse_x, ellipse_y, 'g-', linewidth=2, label='John Ellipsoid')
ax1.scatter(center[0], center[1], c='green', s=100, marker='*', label='Ellipsoid Center')
ax1.legend()

ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Convex Hull with John Ellipsoid')


ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, dom, cmap='viridis')
ax2.set_title('Mesh Domain')


plt.tight_layout()
plt.show()
