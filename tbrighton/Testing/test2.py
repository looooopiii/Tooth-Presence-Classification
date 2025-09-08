"""
Tooth Classification Testing Script (fixed)
- Robust ID extraction (upper/lower + '_' or '-')
- Matches CSV new_id with on-disk PNGs
- Safe model weight loading (handles DataParallel 'module.' prefix)
- Prints all matched IDs and corresponding images
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import os

# === CONFIGURATION ===
TEST_CSV = Path("/home/user/tbrighton/blender-scripts/Testing/label1.csv")
IMAGE_DIR = Path("/home/user/tbrighton/blender_outputs/test_ply_views")
MODEL_PATH = Path("/home/user/tbrighton/blender-scripts/trained_models/best_resnet_weighted_model.pth")
RESULTS_DIR = Path("/home/user/tbrighton/blender-scripts/Results")
BATCH_SIZE = 16

# GPU Configuration (same as training)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Same teeth configuration as training
UPPER_TEETH_L = [18, 17, 16, 15, 14, 13, 12, 11]
UPPER_TEETH_R = [21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH_L = [48, 47, 46, 45, 44, 43, 42, 41]
LOWER_TEETH_R = [31, 32, 33, 34, 35, 36, 37, 38]
ALL_TEETH = sorted(UPPER_TEETH_L + UPPER_TEETH_R + LOWER_TEETH_L + LOWER_TEETH_R)
NUM_CLASSES = len(ALL_TEETH)

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
ID_PATTERN = re.compile(r"^(\d+)")  # digits at start

def build_new_id(name: str) -> str:
    """
    From a raw filename field like:
      '142595_2023-12-07 UpperJawScan' OR '190250-2023-12-14 lower view'
    produce '142595_upper' or '190250_lower'.
    - separator before date may be '_' or '-'
    - 'upper'/'lower' appears anywhere, case-insensitive
    """
    if not isinstance(name, str):
        return ""
    m = ID_PATTERN.match(name)
    if not m:
        return ""
    file_id = m.group(1)
    low = name.lower()
    if "upper" in low:
        return f"{file_id}_upper"
    if "lower" in low:
        return f"{file_id}_lower"
    # fallback if neither term present
    return file_id

def strip_module_prefix(state_dict):
    """Handle DataParallel checkpoints with 'module.' prefix."""
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# === DATASET CLASS ===
class ToothTestDataset(Dataset):
    def __init__(self, df, all_teeth_map, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.all_teeth_map = {tooth: i for i, tooth in enumerate(all_teeth_map)}
        self.image_dir = Path(image_dir)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # images on disk are PNGs named like '<new_id>.png'
        img_path = self.image_dir / f"{row['new_id']}.png"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Create ground truth labels from CSV columns
        labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for tooth_num in ALL_TEETH:
            col = str(tooth_num)
            if tooth_num in self.df.columns:
                val = row[tooth_num]
            elif col in self.df.columns:
                val = row[col]
            else:
                continue
            if not pd.isna(val):
                tooth_idx = self.all_teeth_map[tooth_num]
                labels[tooth_idx] = float(val)
        
        return image, labels, row['new_id']

# === MODEL LOADING ===
def load_trained_model(model_path, num_classes, device):
    """Load the trained ResNet50 model"""
    print(f"Loading model from: {model_path}")
    
    # Create same architecture as training
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    state_dict = strip_module_prefix(state_dict)  # Handle DataParallel prefix
    model.load_state_dict(state_dict)
    
    # Move to DEVICE
    model = model.to(device)
    model.eval()
    return model

# === PREDICTION FUNCTION ===
def predict_batch(model, dataloader, device):
    print("Running inference on test dataset...")
    all_predictions, all_probabilities, all_labels, all_filenames = [], [], [], []
    with torch.no_grad():
        for i, (inputs, labels, filenames) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_filenames.extend(filenames)
            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * BATCH_SIZE} samples...")
    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return predictions, probabilities, labels, all_filenames

# === METRICS CALCULATION ===
def calculate_detailed_metrics(y_true, y_pred, y_probs, tooth_labels):
    metrics_data = []
    print("\n" + "="*60)
    print("DETAILED PER-TOOTH METRICS")
    print("="*60)
    for i, tooth in enumerate(tooth_labels):
        y_true_tooth = y_true[:, i]
        y_pred_tooth = y_pred[:, i]
        y_prob_tooth = y_probs[:, i]
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_tooth, y_pred_tooth, labels=[0, 1]).ravel()
        except ValueError:
            if np.all(y_true_tooth == 0) and np.all(y_pred_tooth == 0):
                tn, fp, fn, tp = len(y_true_tooth), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true_tooth)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        pos_indices = y_true_tooth == 1
        neg_indices = y_true_tooth == 0
        avg_prob_pos = np.mean(y_prob_tooth[pos_indices]) if np.any(pos_indices) else 0
        avg_prob_neg = np.mean(y_prob_tooth[neg_indices]) if np.any(neg_indices) else 0
        metrics_data.append({
            'Tooth': tooth,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1-Score': f1,
            'Accuracy': accuracy,
            'Avg_Prob_Pos': avg_prob_pos,
            'Avg_Prob_Neg': avg_prob_neg,
            'Total_Present': int(np.sum(y_true_tooth)),
            'Total_Missing': int(len(y_true_tooth) - np.sum(y_true_tooth))
        })
        print(f"Tooth {tooth:2d}: Precision={precision:.3f}, Recall={recall:.3f}, "
              f"F1={f1:.3f}, Acc={accuracy:.3f}, Present={int(np.sum(y_true_tooth))}")
    metrics_df = pd.DataFrame(metrics_data).set_index('Tooth')
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    sample_accuracy = np.mean(np.all(y_true == y_pred, axis=1))
    hamming_accuracy = np.mean(y_true == y_pred)
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"Micro Precision:    {micro_precision:.4f}")
    print(f"Micro Recall:       {micro_recall:.4f}")
    print(f"Micro F1-Score:     {micro_f1:.4f}")
    print(f"Sample Accuracy:    {sample_accuracy:.4f}")
    print(f"Label Accuracy:     {hamming_accuracy:.4f}")
    overall_metrics = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'sample_accuracy': sample_accuracy,
        'label_accuracy': hamming_accuracy
    }
    return metrics_df, overall_metrics

# === VISUALIZATION ===
def create_visualizations(metrics_df, overall_metrics, save_dir):
    plt.figure(figsize=(20, 8))
    metrics_to_plot = metrics_df[['Precision', 'Recall', 'F1-Score']]
    ax = metrics_to_plot.plot(kind='bar', width=0.8)
    plt.title('Per-Tooth Performance Metrics (Test Set)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Tooth Number (FDI)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_tooth_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    tooth_presence = metrics_df['Total_Present'].values
    tooth_missing = metrics_df['Total_Missing'].values
    summary_data = np.array([tooth_presence, tooth_missing]).T
    sns.heatmap(summary_data, 
                xticklabels=['Present', 'Missing'],
                yticklabels=[f'Tooth {t}' for t in metrics_df.index],
                annot=True, fmt='d', cmap='Blues')
    plt.title('Ground Truth Tooth Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'tooth_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    macro_metrics = [overall_metrics['macro_precision'], overall_metrics['macro_recall'], overall_metrics['macro_f1']]
    micro_metrics = [overall_metrics['micro_precision'], overall_metrics['micro_recall'], overall_metrics['micro_f1']]
    x = np.arange(3); width = 0.35
    ax1.bar(x - width/2, macro_metrics, width, label='Macro', alpha=0.8)
    ax1.bar(x + width/2, micro_metrics, width, label='Micro', alpha=0.8)
    ax1.set_ylabel('Score'); ax1.set_title('Macro vs Micro Metrics')
    ax1.set_xticks(x); ax1.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    ax2.hist(metrics_df['F1-Score'], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('F1-Score'); ax2.set_ylabel('Number of Teeth'); ax2.set_title('Distribution of Per-Tooth F1-Scores'); ax2.grid(axis='y', alpha=0.3)
    ax3.scatter(metrics_df['Precision'], metrics_df['Recall'], alpha=0.7, s=60)
    ax3.set_xlabel('Precision'); ax3.set_ylabel('Recall'); ax3.set_title('Precision vs Recall per Tooth'); ax3.grid(True, alpha=0.3)
    accuracies = ['Sample Accuracy', 'Label Accuracy']
    accuracy_values = [overall_metrics['sample_accuracy'], overall_metrics['label_accuracy']]
    ax4.bar(accuracies, accuracy_values, alpha=0.8)
    ax4.set_ylabel('Accuracy'); ax4.set_title('Different Accuracy Measures'); ax4.set_ylim(0, 1); ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'overall_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# === SAVE RESULTS ===
def save_detailed_results(predictions, probabilities, labels, filenames, metrics_df, overall_metrics, save_dir):
    results_data = []
    for i, filename in enumerate(filenames):
        row_data = {'filename': filename}
        for j, tooth in enumerate(ALL_TEETH):
            row_data[f'gt_tooth_{tooth}'] = int(labels[i, j])
            row_data[f'pred_tooth_{tooth}'] = int(predictions[i, j])
            row_data[f'prob_tooth_{tooth}'] = float(probabilities[i, j])
        sample_gt = labels[i]; sample_pred = predictions[i]
        sample_accuracy = np.mean(sample_gt == sample_pred)
        row_data['sample_accuracy'] = sample_accuracy
        row_data['teeth_present_gt'] = int(np.sum(sample_gt))
        row_data['teeth_present_pred'] = int(np.sum(sample_pred))
        row_data['teeth_difference'] = row_data['teeth_present_pred'] - row_data['teeth_present_gt']
        results_data.append(row_data)
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(save_dir / 'detailed_predictions.csv', index=False)
    metrics_df.to_csv(save_dir / 'per_tooth_metrics.csv')
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(save_dir / 'overall_metrics.csv', index=False)
    print(f"\nResults saved to: {save_dir}")
    print(f"  - detailed_predictions.csv")
    print(f"  - per_tooth_metrics.csv")
    print(f"  - overall_metrics.csv")

# === MAIN TESTING FUNCTION ===
def main():
    print("="*60)
    print("TOOTH CLASSIFICATION MODEL TESTING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Image Directory: {IMAGE_DIR}")
    print(f"Total Classes (Teeth): {NUM_CLASSES}")
    
    # Load test data
    print(f"\nLoading test data...")
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} test samples")
    
    # Match dataframe IDs with available images
    df_ids = set(df["new_id"].astype(str))
    folder_ids = {f.stem for f in IMAGE_DIR.glob("*.png")}
    
    missing_in_folder = df_ids - folder_ids
    missing_in_df = folder_ids - df_ids
    matches = df_ids & folder_ids
    
    print(f"\nTotal IDs in CSV:     {len(df_ids)}")
    print(f"Total .png in folder: {len(folder_ids)}")
    print(f"Matching:             {len(matches)}")
    
    # PRINT ALL MATCHED IDs AND IMAGES
    print(f"\n" + "="*60)
    print("ALL MATCHED IDs AND CORRESPONDING IMAGES:")
    print("="*60)
    
    matched_files = sorted(list(matches))
    for i, match in enumerate(matched_files, 1):
        corresponding_image = f"{match}.png"
        print(f"{i:3d}. ID: {match:<30} -> Image: {corresponding_image}")
    
    print(f"\nTotal matched files: {len(matched_files)}")
    print("="*60)
    
    if missing_in_folder:
        print(f"\nMissing in folder ({len(missing_in_folder)}):")
        for missing in sorted(list(missing_in_folder)):
            print(f"  - {missing}")
    
    if missing_in_df:
        print(f"\nExtra in folder ({len(missing_in_df)}):")
        for extra in sorted(list(missing_in_df)):
            print(f"  - {extra}.png")
    
    # Keep only rows that have a matching image
    df = df[df["new_id"].isin(matches)].reset_index(drop=True)
    print(f"\nProceeding with {len(df)} matched samples")
    
    # Transforms (same as training/validation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset & Loader
    test_dataset = ToothTestDataset(df, ALL_TEETH, IMAGE_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = load_trained_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    
    # Inference
    start_time = time.time()
    predictions, probabilities, labels, filenames = predict_batch(model, test_loader, DEVICE)
    inference_time = time.time() - start_time
    
    print(f"\nInference completed in {inference_time:.2f} seconds")
    print(f"Average time per sample: {inference_time/len(df):.4f} seconds")
    
    # Metrics
    metrics_df, overall_metrics = calculate_detailed_metrics(labels, predictions, probabilities, ALL_TEETH)
    
    # Visualizations & Results
    print(f"\nCreating visualizations...")
    create_visualizations(metrics_df, overall_metrics, RESULTS_DIR)
    
    print(f"\nSaving detailed results...")
    save_detailed_results(predictions, probabilities, labels, filenames, 
                          metrics_df, overall_metrics, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Key metrics:")
    print(f"  - Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
    print(f"  - Sample Accuracy: {overall_metrics['sample_accuracy']:.4f}")
    print(f"  - Label Accuracy: {overall_metrics['label_accuracy']:.4f}")

if __name__ == "__main__":
    main()
