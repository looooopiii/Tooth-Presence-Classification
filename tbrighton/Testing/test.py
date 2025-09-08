import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import warnings
import time

# --- Suppress UserWarning for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# === CONFIGURATION ===
MODEL_PATH = Path("/home/user/tbrighton/blender-scripts/trained_models/best_resnet_weighted_model.pth")
TEST_IMAGE_DIR = Path("/home/user/tbrighton/blender_outputs/test_views_ply3")
TEST_LABEL_CSV_PATH = Path("/home/user/tbrighton/blender-scripts/Testing/label.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# --- Define ALL possible FDI tooth numbers ---
UPPER_TEETH_COLS = [str(t) for t in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
LOWER_TEETH_COLS = [str(t) for t in [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]]


# === MODEL DEFINITION (To match the original training script) ===
def get_original_trained_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 32)
    )
    return model


# === DATASET PREPARATION (Corrected to handle duplicate file keys) ===
class TeethTestDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        # --- ROBUST LOGIC: Handle potential duplicates in the 'File' column ---
        # Keep only the first occurrence of each file key to prevent ambiguity
        original_len = len(df)
        df_cleaned = df.drop_duplicates(subset=['File'], keep='first')
        if len(df_cleaned) < original_len:
            print(f"  -> INFO: Removed {original_len - len(df_cleaned)} duplicate file entries from labels.")
        
        self.df = df_cleaned
        self.available_columns = set(self.df.columns)
        self.df = self.df.set_index('File')
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.image_dir.glob("*.png")))
        
        self.LOWER_TEETH_MAP = {tooth: i for i, tooth in enumerate(sorted(LOWER_TEETH_COLS, key=int))}
        self.UPPER_TEETH_MAP = {tooth: i for i, tooth in enumerate(sorted(UPPER_TEETH_COLS, key=int))}

        self.samples = []
        for img_path in tqdm(self.image_paths, desc="Matching images to test labels"):
            file_stem = img_path.stem
            file_key = file_stem.removesuffix('_top').removesuffix('_lingual')
            if file_key in self.df.index:
                self.samples.append((img_path, file_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, file_key = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        labels = torch.zeros(16, dtype=torch.float32)
        
        if 'LowerJawScan' in file_key:
            ideal_cols = LOWER_TEETH_COLS
            tooth_map = self.LOWER_TEETH_MAP
        else:
            ideal_cols = UPPER_TEETH_COLS
            tooth_map = self.UPPER_TEETH_MAP

        # This will now be a Series, not a DataFrame, so the error is resolved
        labels_series = self.df.loc[file_key]
        
        for tooth_str in ideal_cols:
            if tooth_str in self.available_columns and labels_series[tooth_str] == 1:
                target_index = tooth_map[tooth_str]
                labels[target_index] = 1.0
        
        return image, labels

# --- IMAGE TRANSFORMS ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === EVALUATION LOGIC ===
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Running inference on {DEVICE}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds_probs = torch.sigmoid(outputs)
            predicted_labels = (preds_probs > 0.5).cpu().numpy()
            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def print_metrics(true_labels, pred_labels):
    print("\n" + "="*60 + "\n           MODEL PERFORMANCE METRICS\n" + "="*60 + "\n")
    exact_match_ratio = accuracy_score(true_labels, pred_labels)
    print(f"Overall Accuracy (Exact Match Ratio): {exact_match_ratio:.4f}\n")
    fdi_labels = sorted(UPPER_TEETH_COLS, key=int)
    target_names = [f"Tooth {label}" for label in fdi_labels]
    print("--- Per-Tooth Classification Report ---\n")
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))
    print("\n--- Per-Tooth Confusion Matrices ---\n")
    mcm = multilabel_confusion_matrix(true_labels, pred_labels)
    for i, tooth_fdi in enumerate(fdi_labels):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"Tooth {tooth_fdi}: TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print("--- Starting Model Evaluation Script ---")
    start_time = time.time()
    
    # Phase 1: Load Test Labels
    print(f"\n[Phase 1/4] Loading test labels from '{TEST_LABEL_CSV_PATH.name}'...")
    try:
        df_test_labels = pd.read_csv(TEST_LABEL_CSV_PATH)
        print(f"-> Success: Loaded {len(df_test_labels)} labels.")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Test label file not found at '{TEST_LABEL_CSV_PATH}'.")
        exit()

    # Phase 2: Build, Load, and Adapt the Model
    print("\n[Phase 2/4] Loading and adapting the trained model...")
    try:
        model = get_original_trained_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        num_ftrs = model.fc[3].in_features
        model.fc[3] = nn.Linear(num_ftrs, 16)
        model.to(DEVICE)
        model.eval()
        print(f"-> Success: Model is adapted and ready on {DEVICE}.")
    except Exception as e:
        print(f"\nFATAL ERROR during model loading and adaptation. Error: {e}")
        exit()
        
    # Phase 3: Prepare Dataset
    print("\n[Phase 3/4] Preparing test dataset...")
    test_dataset = TeethTestDataset(df=df_test_labels, image_dir=TEST_IMAGE_DIR, transform=test_transforms)
    if not test_dataset.samples:
        print("\nERROR: No matching images found. Check paths and filenames.")
        exit()
    print(f"-> Success: Found and matched {len(test_dataset)} test samples.")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Phase 4: Evaluate and Report
    print("\n[Phase 4/4] Running evaluation...")
    ground_truth, predictions = evaluate(model, test_dataloader)
    print("-> Success: Inference complete.")
    
    if ground_truth.shape[0] > 0:
        print_metrics(ground_truth, predictions)
    else:
        print("-> Warning: No data was processed.")

    end_time = time.time()
    print(f"\n--- Evaluation Script Finished in {end_time - start_time:.2f} seconds ---")