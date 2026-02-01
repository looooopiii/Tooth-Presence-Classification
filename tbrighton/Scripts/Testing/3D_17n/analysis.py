"""
apply the SAME preprocessing to test data.
Analyze the spatial distribution (PCA, bounding box, etc.)
Determine canonical orientation
Apply inverse rotation to align test data to training orientation
"""

import torch
import numpy as np
from pathlib import Path
import json
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ============= CONFIGURATION =============
TRAIN_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Analysis/orientation_analysis"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 1800  # How many training samples to analyze

# =========================================
# UTILITIES
# =========================================

def load_obj_vertices(obj_path):
    """Load vertices from OBJ file"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(p) for p in parts[1:4]])
    return np.array(vertices, dtype=np.float32)

def normalize_point_cloud(points):
    """Normalize to unit sphere centered at origin"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    return points_centered / max_dist if max_dist > 0 else points_centered

def analyze_orientation(points):
    """
    Analyze the orientation of a point cloud
    Returns various orientation indicators
    """
    # 1. PCA Analysis
    pca = PCA(n_components=3)
    pca.fit(points)
    principal_axes = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    # 2. Bounding Box Analysis
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    extent = max_coords - min_coords
    
    # 3. Center of Mass
    center = np.mean(points, axis=0)
    
    # 4. Dominant Direction (which axis has most spread)
    dominant_axis = np.argmax(extent)
    
    return {
        'principal_axes': principal_axes,
        'explained_variance': explained_variance,
        'bounding_box_min': min_coords,
        'bounding_box_max': max_coords,
        'extent': extent,
        'center': center,
        'dominant_axis': dominant_axis,
        'dominant_axis_name': ['X', 'Y', 'Z'][dominant_axis]
    }

def visualize_orientation(points, title="Point Cloud", save_path=None):
    """Visualize point cloud with orientation axes"""
    fig = plt.figure(figsize=(12, 4))
    
    # Subsample for visualization
    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        points_vis = points[indices]
    else:
        points_vis = points
    
    # XY view
    ax1 = fig.add_subplot(131)
    ax1.scatter(points_vis[:, 0], points_vis[:, 1], s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'{title} - XY View')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # XZ view
    ax2 = fig.add_subplot(132)
    ax2.scatter(points_vis[:, 0], points_vis[:, 2], s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'{title} - XZ View')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # YZ view
    ax3 = fig.add_subplot(133)
    ax3.scatter(points_vis[:, 1], points_vis[:, 2], s=1, alpha=0.5)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title(f'{title} - YZ View')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =========================================
# MAIN ANALYSIS
# =========================================

def main():
    print("="*80)
    print("🔍 ANALYZING TRAINING SET ORIENTATION")
    print("="*80)
    
    # Collect training samples
    print(f"\n[1/4] Collecting {NUM_SAMPLES} training samples...")
    all_obj_files = []
    
    for data_path_str in TRAIN_DATA_PATHS:
        data_path = Path(data_path_str)
        if not data_path.exists():
            continue
        
        for case_dir in sorted(data_path.iterdir()):
            if case_dir.is_dir():
                case_id = case_dir.name
                for jaw_type in ['upper', 'lower']:
                    obj_file = case_dir / f"{case_id}_{jaw_type}.obj"
                    if obj_file.exists():
                        all_obj_files.append(obj_file)
    
    # Sample randomly
    np.random.seed(42)
    if len(all_obj_files) > NUM_SAMPLES:
        sampled_files = np.random.choice(all_obj_files, NUM_SAMPLES, replace=False)
    else:
        sampled_files = all_obj_files
    
    print(f"✓ Found {len(all_obj_files)} total training files")
    print(f"✓ Analyzing {len(sampled_files)} samples")
    
    # Analyze each sample
    print("\n[2/4] Analyzing orientation statistics...")
    all_stats = []
    
    for obj_file in sampled_files:
        points = load_obj_vertices(obj_file)
        if len(points) < 100:
            continue
        
        # Normalize (same as training preprocessing)
        points_norm = normalize_point_cloud(points)
        
        # Analyze orientation
        stats = analyze_orientation(points_norm)
        all_stats.append(stats)
    
    print(f"✓ Analyzed {len(all_stats)} point clouds")
    
    # Aggregate statistics
    print("\n[3/4] Aggregating orientation patterns...")
    
    # Average PCA principal axes
    all_pc1 = np.array([s['principal_axes'][0] for s in all_stats])
    all_pc2 = np.array([s['principal_axes'][1] for s in all_stats])
    all_pc3 = np.array([s['principal_axes'][2] for s in all_stats])
    
    avg_pc1 = np.mean(all_pc1, axis=0)
    avg_pc2 = np.mean(all_pc2, axis=0)
    avg_pc3 = np.mean(all_pc3, axis=0)
    
    # Average extent (bounding box size per axis)
    all_extents = np.array([s['extent'] for s in all_stats])
    avg_extent = np.mean(all_extents, axis=0)
    std_extent = np.std(all_extents, axis=0)
    
    # Dominant axis distribution
    dominant_axes = [s['dominant_axis'] for s in all_stats]
    dominant_axis_counts = np.bincount(dominant_axes, minlength=3)
    
    # Print Analysis
    print("\n" + "="*80)
    print("📊 ORIENTATION ANALYSIS RESULTS")
    print("="*80)
    
    print("\n1. AVERAGE BOUNDING BOX EXTENT (normalized):")
    print(f"   X-axis: {avg_extent[0]:.4f} ± {std_extent[0]:.4f}")
    print(f"   Y-axis: {avg_extent[1]:.4f} ± {std_extent[1]:.4f}")
    print(f"   Z-axis: {avg_extent[2]:.4f} ± {std_extent[2]:.4f}")
    
    print("\n2. DOMINANT AXIS (most spread):")
    axis_names = ['X', 'Y', 'Z']
    for i, count in enumerate(dominant_axis_counts):
        percentage = (count / len(all_stats)) * 100
        print(f"   {axis_names[i]}-axis: {count}/{len(all_stats)} ({percentage:.1f}%)")
    
    print("\n3. AVERAGE PRINCIPAL COMPONENT DIRECTIONS:")
    print(f"   PC1 (most variance): [{avg_pc1[0]:+.3f}, {avg_pc1[1]:+.3f}, {avg_pc1[2]:+.3f}]")
    print(f"   PC2:                 [{avg_pc2[0]:+.3f}, {avg_pc2[1]:+.3f}, {avg_pc2[2]:+.3f}]")
    print(f"   PC3 (least variance):[{avg_pc3[0]:+.3f}, {avg_pc3[1]:+.3f}, {avg_pc3[2]:+.3f}]")
    
    print("\n4. INTERPRETATION:")
    most_dominant_axis = np.argmax(dominant_axis_counts)
    print(f"   ✓ Training data is primarily oriented along the {axis_names[most_dominant_axis]}-axis")
    print(f"   ✓ Average extent ratios: X:{avg_extent[0]:.2f} Y:{avg_extent[1]:.2f} Z:{avg_extent[2]:.2f}")
    
    # Determine if data needs rotation based on expected dental arch orientation
    # Dental arches typically have:
    # - Long axis (arch curve): should be in XY plane
    # - Short axis (tooth height): should be along Z
    
    print("\n5. RECOMMENDED PREPROCESSING FOR TEST DATA:")
    if avg_extent[2] < avg_extent[0] and avg_extent[2] < avg_extent[1]:
        print("   ✓ Training data appears correctly oriented:")
        print("     - Dental arch spreads in XY plane")
        print("     - Tooth height along Z-axis")
        print("   ✓ NO ROTATION needed for test data")
        recommended_rotation = (0, 0, 0)
    else:
        print("   ⚠ Training data may have non-standard orientation")
        print("   ✓ Recommend analyzing individual samples to determine rotation")
        # Simple heuristic: if Z has most spread, might need X rotation
        if most_dominant_axis == 2:  # Z-axis
            recommended_rotation = (-90, 0, 0)  # Rotate around X to bring Z to Y
            print(f"   ✓ Suggested rotation: {recommended_rotation}")
        else:
            recommended_rotation = (0, 0, 0)
            print(f"   ✓ Suggested rotation: {recommended_rotation} (no rotation)")
    
    # Visualize sample
    print("\n[4/4] Generating visualizations...")
    sample_file = sampled_files[0]
    sample_points = normalize_point_cloud(load_obj_vertices(sample_file))
    
    visualize_orientation(
        sample_points, 
        title=f"Training Sample: {sample_file.stem}",
        save_path=Path(OUTPUT_DIR) / "training_orientation_sample.png"
    )
    
    # Save results
    results = {
        'num_samples_analyzed': len(all_stats),
        'average_extent': {
            'x': float(avg_extent[0]),
            'y': float(avg_extent[1]),
            'z': float(avg_extent[2])
        },
        'extent_std': {
            'x': float(std_extent[0]),
            'y': float(std_extent[1]),
            'z': float(std_extent[2])
        },
        'dominant_axis_distribution': {
            'x_count': int(dominant_axis_counts[0]),
            'y_count': int(dominant_axis_counts[1]),
            'z_count': int(dominant_axis_counts[2])
        },
        'average_principal_components': {
            'pc1': avg_pc1.tolist(),
            'pc2': avg_pc2.tolist(),
            'pc3': avg_pc3.tolist()
        },
        'recommended_rotation': recommended_rotation
    }
    
    with open(Path(OUTPUT_DIR) / "orientation_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Visualization saved to: {OUTPUT_DIR}/training_orientation_sample.png")
    print(f"✓ Full analysis saved to: {OUTPUT_DIR}/orientation_analysis.json")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNEXT STEPS:")
    print(f"1. Review the visualization to confirm orientation")
    print(f"2. Apply rotation {recommended_rotation} to test data if needed")
    print(f"3. This rotation is based on training data, so it's VALID preprocessing")
    print("="*80)

if __name__ == "__main__":
    main()