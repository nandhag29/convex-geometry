import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import time


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


def generate_triangle_domain(v1, v2, N=5, h=0.01, plot=True):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))
    
    # Create matrix A with v1 and v2 as columns
    A = np.column_stack([v1, v2])
    
    # Initialize domain matrix
    dom = np.zeros(x.shape)
    
    # Get dimensions
    s1, s2 = x.shape
    
    # Loop through all points and check if they're in the triangle
    for i in range(s1):
        for j in range(s2):
            # Transform the point using inverse of A
            point = np.array([x[i, j], y[i, j]])
            try:
                a = np.linalg.solve(A, point)
                # Check triangle conditions: a[0] >= 0, a[1] >= 0, a[0] + a[1] <= 1
                if a[0] >= 0 and a[1] >= 0 and a[0] + a[1] <= 1:
                    dom[i, j] = 1
            except np.linalg.LinAlgError:
                continue
    
    if plot:
        # Create plots showing both 3D mesh and 2D contour views
        fig = plt.figure(figsize=(12, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, dom, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Domain')
        ax1.set_title('3D Mesh Plot of Triangle Domain')
        
        # 2D contour plot
        ax2 = fig.add_subplot(122)
        ax2.contourf(x, y, dom, levels=[0, 0.5, 1], colors=['white', 'blue'], alpha=0.7)
        ax2.contour(x, y, dom, levels=[0.5], colors=['black'], linewidths=2)
        
        # Plot the triangle vertices
        origin = np.array([0, 0])
        vertices = np.array([origin, v1, v2, origin])
        ax2.plot(vertices[:, 0], vertices[:, 1], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('2D View: Triangle Domain (Blue = Inside)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    return x, y, dom


def generate_triangle_domain_vectorized(v1, v2, N=5, h=0.01, plot=True):
    """
    Vectorized version of generate_triangle_domain for better performance.
    Same parameters and returns as generate_triangle_domain.
    """
    # Convert to numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(-N, N+h, h), np.arange(-N, N+h, h))
    
    # Create matrix A with v1 and v2 as columns
    A = np.column_stack([v1, v2])
    
    # Vectorized computation
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    try:
        transformed = np.linalg.solve(A, points.T).T
        mask = (transformed[:, 0] >= 0) & (transformed[:, 1] >= 0) & (transformed[:, 0] + transformed[:, 1] <= 1)
        dom = mask.reshape(x.shape).astype(int)
    except np.linalg.LinAlgError:
        dom = np.zeros(x.shape)
    
    if plot:
        # Same plotting code as the loop-based version
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, dom, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Domain')
        ax1.set_title('3D Mesh Plot of Triangle Domain')
        
        ax2 = fig.add_subplot(122)
        ax2.contourf(x, y, dom, levels=[0, 0.5, 1], colors=['white', 'blue'], alpha=0.7)
        ax2.contour(x, y, dom, levels=[0.5], colors=['black'], linewidths=2)
        
        origin = np.array([0, 0])
        vertices = np.array([origin, v1, v2, origin])
        ax2.plot(vertices[:, 0], vertices[:, 1], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('2D View: Triangle Domain (Blue = Inside)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    return x, y, dom

def analyze_triangle_containment(points=None, N=10, bounds=(-5, 5), seed=None, plot=True, max_triangles_to_plot=5):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate or use provided points
    if points is None:
        points = np.random.uniform(bounds[0], bounds[1], size=(N, 2))
    else:
        points = np.asarray(points)
        N = len(points)
    
    print(f"Analyzing {N} points...")
    print(f"Total number of possible triangles: {len(list(combinations(range(N), 3)))}")
    
    results = {
        'points': points,
        'triangles': [],
        'summary_stats': {}
    }
    
    triangle_count = 0
    start_time = time.time()
    
    # Iterate through all combinations of 3 points
    for i, (idx1, idx2, idx3) in enumerate(combinations(range(N), 3)):
        triangle_vertices = [points[idx1], points[idx2], points[idx3]]
        
        # Find all other points
        other_indices = [j for j in range(N) if j not in [idx1, idx2, idx3]]
        contained_points = []
        contained_indices = []
        
        # Check which other points are in this triangle
        for other_idx in other_indices:
            if is_in_convex_hull(triangle_vertices[0], triangle_vertices[1], 
                                triangle_vertices[2], points[other_idx]):
                contained_points.append(points[other_idx])
                contained_indices.append(other_idx)
        
        # Store results for this triangle
        triangle_result = {
            'triangle_indices': [idx1, idx2, idx3],
            'triangle_vertices': triangle_vertices,
            'contained_indices': contained_indices,
            'contained_points': contained_points,
            'num_contained': len(contained_points)
        }
        
        results['triangles'].append(triangle_result)
        triangle_count += 1
        
        if triangle_count % 50 == 0:
            print(f"Processed {triangle_count} triangles...")
    
    elapsed_time = time.time() - start_time
    print(f"Completed analysis in {elapsed_time:.2f} seconds")
    
    # Calculate summary statistics
    containment_counts = [t['num_contained'] for t in results['triangles']]
    results['summary_stats'] = {
        'total_triangles': len(results['triangles']),
        'total_points': N,
        'avg_points_per_triangle': np.mean(containment_counts),
        'max_points_in_triangle': max(containment_counts) if containment_counts else 0,
        'min_points_in_triangle': min(containment_counts) if containment_counts else 0,
        'triangles_with_points': sum(1 for count in containment_counts if count > 0),
        'empty_triangles': sum(1 for count in containment_counts if count == 0)
    }
    
    if plot and N <= 50:  # Only plot for reasonable number of points
        plot_containment_analysis(results, max_triangles_to_plot)
    elif plot:
        print(f"Skipping plot due to large number of points ({N}). Set plot=False or use fewer points.")
    
    return results


def plot_containment_analysis(results, max_triangles=5):
    """
    Create visualization of the triangle containment analysis.
    Modified to show all triangles instead of just top ones and removed distribution plot.
    """
    points = results['points']
    triangles = results['triangles']

    # Create single plot showing all triangles
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], c='lightblue', s=50, alpha=0.7, zorder=2)

    # Annotate points with their indices
    for i, point in enumerate(points):
        ax.annotate(str(i), (point[0], point[1]), xytext=(3, 3), 
                    textcoords='offset points', fontsize=8)

    # Plot all triangles
    colors = plt.cm.Set3(np.linspace(0, 1, len(triangles)))

    for i, triangle in enumerate(triangles):
        vertices = triangle['triangle_vertices']
        triangle_points = np.array([vertices[0], vertices[1], vertices[2], vertices[0]])

        # Plot triangle outline
        ax.plot(triangle_points[:, 0], triangle_points[:, 1], 
                color=colors[i], linewidth=1.5, alpha=0.6)

        # If triangle contains points, highlight them
        if triangle['num_contained'] > 0:
            for contained_point in triangle['contained_points']:
                ax.scatter(contained_point[0], contained_point[1], 
                           color=colors[i], s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_title(f"All Triangles and Point Containment\n({len(triangles)} triangles total)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def print_summary(results):
    """
    Print a summary of the triangle containment analysis.
    """
    stats = results['summary_stats']
    print("\n" + "="*50)
    print("TRIANGLE CONTAINMENT ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total points: {stats['total_points']}")
    print(f"Total triangles: {stats['total_triangles']}")
    print(f"Average points per triangle: {stats['avg_points_per_triangle']:.2f}")
    print(f"Maximum points in any triangle: {stats['max_points_in_triangle']}")
    print(f"Minimum points in any triangle: {stats['min_points_in_triangle']}")
    print(f"Triangles containing at least one point: {stats['triangles_with_points']}")
    print(f"Empty triangles: {stats['empty_triangles']}")

if __name__ == "__main__":
    # Test is_in_convex_hull
    x1 = np.array([0, 0])
    x2 = np.array([4, 0])
    x3 = np.array([2, 4])
    w = np.array([1, 2])
    print("Point is in convex hull:", is_in_convex_hull(x1, x2, x3, w))
    
    # Test triangle domain generation with the original MATLAB vectors
    v1 = np.array([3, 0])
    v2 = np.array([1, 2])
    print(f"\nGenerating triangle domain with v1={v1}, v2={v2}")
    
    x, y, dom = generate_triangle_domain(v1, v2, N=3, h=0.05)
    print(f"Grid shape: {x.shape}")
    print(f"Points inside triangle: {np.sum(dom)}")

    results = analyze_triangle_containment(N=8, seed=42, bounds=(-3, 3))
    print_summary(results)
    
    # Example with custom points
    custom_points = np.array([[0, 0], [3, 0], [1.5, 2], [1, 1], [2, 0.5]])
    results2 = analyze_triangle_containment(points=custom_points, plot=True)
    print_summary(results2)
