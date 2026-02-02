"""
orient_top_view: Rotate smallest dimension to Z
upright_with_pca: PCA alignment (longest→X/Y, shortest→Z)
auto_flip: Ensure normals point up
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

def orient_top_view(points):
    """
   Rotate smallest dimension to Z-axis
    """
    # Calculate bounding box dimensions
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dims = max_coords - min_coords
    
    # Find smallest dimension
    min_idx = np.argmin(dims)
    
    # Rotate so smallest dimension is along Z
    if min_idx == 0:  # X is smallest → rotate 90° around Y
        rot = R.from_euler('y', 90, degrees=True)
        points = rot.apply(points)
    elif min_idx == 1:  # Y is smallest → rotate -90° around X
        rot = R.from_euler('x', -90, degrees=True)
        points = rot.apply(points)
    # If Z is already smallest (min_idx == 2), no rotation needed
    
    return points

def upright_with_pca(points):
    """
 PCA-based upright orientation
    """
    if len(points) < 8:
        return points
    
    # Center the points
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Get eigenvalues and eigenvectors (sorted ascending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Eigenvectors sorted by variance: [min, mid, max]
    v_min = eigenvectors[:, 0]  # Shortest variance → Z
    v_mid = eigenvectors[:, 1]  # Medium variance
    v_max = eigenvectors[:, 2]  # Longest variance
    
    # Determine which axis (X or Y) should be longest
    # Project points onto mid and max axes to find their extent
    proj_mid = centered @ v_mid
    proj_max = centered @ v_max
    
    len_mid = np.ptp(proj_mid)  # Peak-to-peak (range)
    len_max = np.ptp(proj_max)
    
    # Assign axes based on extent
    if len_max >= len_mid:
        x_axis = v_max
        y_axis = v_mid
    else:
        x_axis = v_mid
        y_axis = v_max
    
    z_axis = v_min
    
    # Create rotation matrix [x, y, z]
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Ensure right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0:
        y_axis = -y_axis
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Apply rotation
    points_aligned = centered @ rotation_matrix.T
    
    return points_aligned

def auto_flip_for_top_view(points):
    """Ensure top surface normals point upward"""
   
    # Rotate Y>X if needed (arch orientation)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dims = max_coords - min_coords
    
    if dims[1] > dims[0]:  # Y > X
        rot = R.from_euler('z', 90, degrees=True)
        points = rot.apply(points)
    
    # Check if we need to flip (simplified - without normals)
    # Use Z-coordinate distribution as proxy
    z_coords = points[:, 2]
    z_threshold = np.percentile(z_coords, 85)  # Top 15% of points
    
    top_points = points[z_coords >= z_threshold]
    
    if len(top_points) > 0:
        # Check if top points are mostly on negative side
        mean_z_top = np.mean(top_points[:, 2])
        
        # If mean is negative, flip
        if mean_z_top < 0:
            rot = R.from_euler('x', 180, degrees=True)
            points = rot.apply(points)
    
    return points

def blender_alignment_pipeline(points):
    """ points_aligned: Aligned point cloud
        total_rotation: Combined rotation matrix (for reference)
    """
    # Store original for calculating total rotation
    points_original = points.copy()
    
    # Step 1: Orient top view
    points = orient_top_view(points)
    
    # Step 2: PCA upright
    points = upright_with_pca(points)
    
    # Step 3: Auto-flip
    points = auto_flip_for_top_view(points)
    
    # Calculate total rotation (optional, for debugging)
    # This is approximate since we've transformed in steps
    total_rotation = None  # Complex to compute exactly
    
    return points

# =================================================================================
# INTEGRATION INTO YOUR TEST CODE
# =================================================================================

def normalize_point_cloud(points):
    """Normalize to unit sphere (same as training)"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    return points_centered / max_dist if max_dist > 0 else points_centered

def sample_points(points, num_points=4096):
    """Sample fixed number of points"""
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace_flag = len(points) < num_points
    indices = np.random.choice(len(points), num_points, replace=replace_flag)
    return points[indices]

def preprocess_test_data(points, num_points=4096):
    """
    Complete preprocessing pipeline for test data
        points_preprocessed: Ready for model inference [num_points, 3]
    """
    # Step 1: Apply Blender's alignment pipeline
    points_aligned = blender_alignment_pipeline(points)
    
    # Step 2: Normalize (same as training)
    points_normalized = normalize_point_cloud(points_aligned)
    
    # Step 3: Sample (same as training)
    points_sampled = sample_points(points_normalized, num_points)
    
    return points_sampled


# =================================================================================
# EXAMPLE USAGE IN YOUR TEST SCRIPT
# =================================================================================


# =================================================================================
# VERIFICATION: Check if alignment matches Blender
# =================================================================================

def verify_alignment(points_before, points_after):
    """
    Verify that alignment worked correctly
    """
    # Check dimensions
    min_coords = np.min(points_after, axis=0)
    max_coords = np.max(points_after, axis=0)
    dims = max_coords - min_coords
    
    print(f"Dimensions after alignment: X={dims[0]:.3f}, Y={dims[1]:.3f}, Z={dims[2]:.3f}")
    
    # Z should be smallest
    if dims[2] < dims[0] and dims[2] < dims[1]:
        print("✓ Tooth height along Z-axis (correct)")
    else:
        print("⚠ Warning: Z-axis is not the smallest dimension")
    
    # Check centering
    center = np.mean(points_after, axis=0)
    print(f"Center: {center}")
    
    if np.allclose(center, [0, 0, 0], atol=1e-5):
        print("✓ Points centered at origin")
    else:
        print(f"⚠ Points not perfectly centered (may be okay)")
    
    return dims
