import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import trimesh

# ============= CONFIGURATION =============
TEST_PLY_DIR = "/home/user/tbrighton/blender_outputs/parsed_ply"
TEST_LABELS_CSV = "/home/user/tbrighton/Scripts/Testing/3D/label_processed.csv"
MODEL_PATH = "/home/user/tbrighton/Scripts/Training/3D/trained_models/best_model_f1.pth"
OUTPUT_DIR = "/home/user/tbrighton/Scripts/Testing/3D/test_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

NUM_POINTS = 2048
NUM_TEETH = 32

# FDI tooth numbering to index mapping
FDI_TO_INDEX = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    38: 16, 37: 17, 36: 18, 35: 19, 34: 20, 33: 21, 32: 22, 31: 23,
    41: 24, 42: 25, 43: 26, 44: 27, 45: 28, 46: 29, 47: 30, 48: 31
}

# ============= MODEL ARCHITECTURE =============
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2)[0]
        return x

class ToothClassificationModel(nn.Module):
    def __init__(self, num_teeth=32, feature_dim=1024):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=3, feature_dim=feature_dim)
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_teeth)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self, x):
        features = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============= DATA LOADING FUNCTIONS =============
def load_ply_file(ply_path):
    try:
        mesh = trimesh.load(ply_path, process=False)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        return vertices
    except Exception as e:
        print(f"❌ Error loading {ply_path}: {e}")
        return np.array([], dtype=np.float32)

def normalize_point_cloud(points):
    if len(points) == 0:
        return points
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points

def sample_points(points, num_points=2048):
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    num_vertices = len(points)
    if num_vertices >= num_points:
        indices = np.random.choice(num_vertices, num_points, replace=False)
    else:
        indices = np.random.choice(num_vertices, num_points, replace=True)
    return points[indices]

def load_test_labels(csv_path):
    print("\n🔍 DEBUG: Loading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Debug: Show column types and names
    print("📋 CSV Columns (with types):")
    for col in df.columns:
        print(f"  {col} ({type(col).__name__})")
    
    print(f"\n📊 CSV Shape: {df.shape}")
    print(f"🔍 First few rows of 'new_id': {df['new_id'].head().tolist()}")
    
    # Ensure 'new_id' is string
    df['new_id'] = df['new_id'].astype(str)
    
    # Check if FDI columns exist as integers or strings
    tooth_columns = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
                     38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
    
    # Try both int and str column names
    available_cols = set(df.columns)
    print(f"\n🔍 Available columns in CSV: {sorted(available_cols)}")
    
    labels_dict = {}
    for _, row in df.iterrows():
        new_id = str(row['new_id']).strip()
        label_vector = np.zeros(NUM_TEETH, dtype=np.float32)
        
        for tooth_fdi in tooth_columns:
            # Try both int and str keys
            found = False
            if tooth_fdi in available_cols:
                val = row[tooth_fdi]
                found = True
            elif str(tooth_fdi) in available_cols:
                val = row[str(tooth_fdi)]
                found = True
            
            if found and pd.notna(val) and val == 1:
                idx = FDI_TO_INDEX[tooth_fdi]
                label_vector[idx] = 1.0
        
        labels_dict[new_id] = label_vector
    
    print(f"\n✅ Loaded labels for {len(labels_dict)} cases")
    
    # Debug: Print first few label vectors
    print("\n🔍 First 5 label vectors (debug):")
    for i, (case_id, labels) in enumerate(list(labels_dict.items())[:5]):
        present_teeth = [k for k, v in FDI_TO_INDEX.items() if labels[v] == 1]
        print(f"  {case_id}: {present_teeth if present_teeth else 'No teeth present'}")
    
    return labels_dict

# ============= MAIN DEBUGGING PIPELINE =============
def main():
    print("🚀 Starting debugging script for 3D tooth classification")
    
    # Load labels
    labels_dict = load_test_labels(TEST_LABELS_CSV)
    if not labels_dict:
        print("❌ No labels loaded. Check CSV path and format.")
        return
    
    # Load PLY files
    test_ply_dir = Path(TEST_PLY_DIR)
    if not test_ply_dir.exists():
        print(f"❌ PLY directory not found: {TEST_PLY_DIR}")
        return
    
    ply_files = sorted(test_ply_dir.glob("*.ply"))
    print(f"\n📁 Found {len(ply_files)} PLY files")
    
    # Show first few PLY filenames
    print("\n🔍 First 10 PLY filenames (stems):")
    for f in ply_files[:10]:
        print(f"  {f.stem}")
    
    # Match analysis
    print("\n🔗 Matching PLY files with CSV labels...")
    matched = []
    unmatched = []
    
    for f in ply_files:
        case_id = f.stem
        if case_id in labels_dict:
            matched.append(case_id)
        else:
            unmatched.append(case_id)
    
    print(f"\n✅ Matched: {len(matched)}")
    print(f"❌ Unmatched: {len(unmatched)}")
    
    if matched:
        print("\n🔍 First 5 matched case IDs:")
        for case_id in matched[:5]:
            print(f"  {case_id}")
    
    # Load and inspect first matched sample
    if matched:
        print("\n🧪 Loading first matched sample for inspection:")
        first_case = matched[0]
        first_ply = test_ply_dir / f"{first_case}.ply"
        
        points = load_ply_file(first_ply)
        print(f"  Point cloud shape: {points.shape}")
        
        labels = labels_dict[first_case]
        present_teeth = [k for k, v in FDI_TO_INDEX.items() if labels[v] == 1]
        print(f"  True labels: {present_teeth if present_teeth else 'All teeth missing (0)'}")
        
        if len(present_teeth) == 0:
            print("⚠️ WARNING: First sample has no teeth labeled as present — check if this is expected.")
    
    print("\n✅ Debugging complete. Review output above to identify label loading issues.")

if __name__ == "__main__":
    main()
