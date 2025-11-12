import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess
from collections import OrderedDict
from PIL import Image
from torchvision import transforms, models


# ============= CONFIGURATION =============
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

# 2D rendered image roots
IMG_ROOT_LOWER = "/home/user/lzhou/week4/multi_views/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week4/multi_views/upperjaw"

# Outputs
OUTPUT_DIR = "/home/user/lzhou/week11/output/Train2D/dynamit_17out"
PLOT_DIR = "/home/user/lzhou/week11/output/Train2D/dynamit_17out/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "dynamit_loss_best_2d_17out.pth" 
LAST_MODEL_FILENAME = "dynamit_loss_last_2d_17out.pth" 
PLOT_FILENAME = "dynamit_loss_training_metrics_2d_17out.png"
METRICS_FILENAME = "dynamit_loss_detailed_metrics_2d_17out.json"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 1e-3
IMG_SIZE = 320
# NUM_TEETH = 32 
NUM_OUTPUTS = 17  # 16 teeth + 1 jaw indicator
NUM_TEETH_PER_JAW = 16
SEED = 41
BACKBONE = "resnet18"  # or "resnet50"

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4

# FDI Notation Mapping ---
# new mapping for 16 teeth per jaw
# upperjaw: 18-11, 21-28 reflect to 0-15
UPPER_FDI_TO_IDX16 = {
    18: 0, 17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15
}
# lowerjaw: 48-41, 31-38 reflect to 0-15
LOWER_FDI_TO_IDX16 = {
    48: 0, 47: 1, 46: 2, 45: 3, 44: 4, 43: 5, 42: 6, 41: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13, 37: 14, 38: 15
}

available_gpus = []
device = None
# =========================================


# --- Dynamit Loss Function (Numerically Stable Version) ---
class Dynamit_Loss(nn.Module):
    """
    Batch-adaptive class-balancing loss.
    Positive class = missing (1), Negative = present (0).
    Weights are computed per-batch from target counts.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, logits, targets):
        S_pos = (targets == 1).sum().float()
        S_neg = (targets == 0).sum().float()
        pos_coeff_val = min(1.0, (S_neg / S_pos).item()) if S_pos > 0 else 0.0
        neg_coeff_val = min(1.0, (S_pos / S_neg).item()) if S_neg > 0 else 0.0
        pos_coeff = torch.tensor(pos_coeff_val, device=self.device)
        neg_coeff = torch.tensor(neg_coeff_val, device=self.device)
        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


# --- 2D Dataset: PNGs + JSON labels ---
class Tooth2DDataset(Dataset):
    """
    Read rendered top-view PNGs and match labels from JSON by case_id + jaw.
    Expected PNG file naming (examples):
      {case_id}_lower_top.png  or  {case_id}_upper_top.png
    JSON files are expected at:
      lower: {JSON_ROOT_LOWER}/{case_id}/{case_id}_lower.json
      upper: {JSON_ROOT_UPPER}/{case_id}/{case_id}_upper.json
    """
    def __init__(self, img_roots, json_roots, transform=None):
        self.samples = []
        self.transform = transform
        lower_img_root, upper_img_root = [Path(p) for p in img_roots]
        lower_json_root, upper_json_root = [Path(p) for p in json_roots]

        def add_samples(img_root, jaw):
            if not img_root.exists():
                return
            for png in sorted(img_root.glob("*_top.png")):
                stem = png.stem  # e.g., 0AAQ6BO3_lower_top
                core = stem[:-4] if stem.endswith("_top") else stem
                case_id, cur_jaw = None, None
                if core.endswith("_lower"):
                    case_id, cur_jaw = core[:-6], "lower"
                elif core.endswith("_upper"):
                    case_id, cur_jaw = core[:-6], "upper"
                else:
                    parts = core.split("_")
                    if len(parts) >= 2:
                        case_id, cur_jaw = parts[0], parts[1].lower()
                if cur_jaw != jaw or case_id is None:
                    continue
                json_path = (lower_json_root if jaw == "lower" else upper_json_root) / case_id / f"{case_id}_{jaw}.json"
                if json_path.exists():
                    self.samples.append({"img": str(png), "json": str(json_path), "case_id": case_id, "jaw": jaw})

        add_samples(lower_img_root, "lower")
        add_samples(upper_img_root, "upper")
        print(f"[Info] Loaded {len(self.samples)} 2D samples")

    def __len__(self): 
        return len(self.samples)

    # 17 vector creation
    def _create_label_vector(self, json_path, jaw):
        # load JSON
        with open(json_path, "r") as f:
            data = json.load(f)
        labels_set = set(data.get("labels", []))

        # Initialize 17-dimensional vector
        output_vector = np.zeros(NUM_OUTPUTS, dtype=np.float32)

        # determine mapping and jaw label
        if jaw == "upper":
            mapping = UPPER_FDI_TO_IDX16
            jaw_label = 0.0  # 0.0 upper jaw
        elif jaw == "lower":
            mapping = LOWER_FDI_TO_IDX16
            jaw_label = 1.0  # 1.0 lower jaw
        else:
            return output_vector # invalid jaw

        # Fill 16-teeth labels (0=present, 1=missing)
        present_teeth = np.zeros(NUM_TEETH_PER_JAW, dtype=np.float32)
        for fdi in labels_set:
            idx = mapping.get(fdi)
            if idx is not None:
                present_teeth[idx] = 1.0
        
        missing_teeth = 1.0 - present_teeth

        # Combine into final 17-dimensional vector
        output_vector[:NUM_TEETH_PER_JAW] = missing_teeth
        output_vector[NUM_TEETH_PER_JAW] = jaw_label # Last element is jaw label

        return output_vector

    # new __getitem__ ---
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img"]).convert("RGB")

        # Get jaw from sample info
        jaw = sample["jaw"]
        # Create new 17-dimensional label
        y = self._create_label_vector(sample["json"], jaw)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(y).float()


# --- ResNet Multi-label Model ---
class ResNetMultiLabel(nn.Module):
    # --- use num_outputs ---
    def __init__(self, backbone="resnet18", num_outputs=17, pretrained=True):
        super().__init__()
        bb = (backbone or "resnet18").lower()
        if bb == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_outputs) 
        self.net = net
    def forward(self, x):
        return self.net(x)


def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        free_gpus = [int(line.split(', ')[0]) for line in result.stdout.strip().split('\n') if int(line.split(', ')[1]) < threshold_mb]
        if len(free_gpus) > max_gpus: free_gpus = free_gpus[:max_gpus]
        if not free_gpus: print("Warning: No free GPUs found, using GPU 0"); return [0]
        print(f"✓ Free GPUs detected: {free_gpus}"); return free_gpus
    except Exception as e:
        print(f"Error detecting free GPUs: {e}\nFalling back to GPU 0"); return [0]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def calculate_classification_metrics(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(int)
    target = target.cpu().numpy().astype(int)
    pred_flat, target_flat = pred.flatten(), target.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(target_flat, pred_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy_score(target_flat, pred_flat)}

# --- calculate_jaw_accuracy ---
def calculate_jaw_accuracy(pred_jaw, target_jaw):
    """计算牙颌分类的准确率"""
    pred = (pred_jaw > 0.5).cpu().numpy().astype(int)
    target = target_jaw.cpu().numpy().astype(int)
    return {'jaw_accuracy': accuracy_score(target, pred)}


# def calculate_per_tooth_metrics(pred, target, num_teeth=32):
#     ... (旧代码)

# changed train_epoch
def train_epoch(model, dataloader, criterion_teeth, criterion_jaw, optimizer, device):
    model.train()
    total_loss = 0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []
    
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # separate labels
        labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
        labels_jaw = labels[:, NUM_TEETH_PER_JAW]

        optimizer.zero_grad()
        logits = model(imgs) # Shape [B, 17]
        
        # separate logits
        logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
        logits_jaw = logits[:, NUM_TEETH_PER_JAW]

        # calculate losses separately
        loss_teeth = criterion_teeth(logits_teeth, labels_teeth) # Dynamit loss
        loss_jaw = criterion_jaw(logits_jaw, labels_jaw)         # BCE loss
        loss = loss_teeth + loss_jaw # total loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # store predictions and labels
        all_preds_teeth.append(torch.sigmoid(logits_teeth.detach()))
        all_labels_teeth.append(labels_teeth)
        all_preds_jaw.append(torch.sigmoid(logits_jaw.detach()))
        all_labels_jaw.append(labels_jaw)
        
    all_preds_teeth = torch.cat(all_preds_teeth, dim=0)
    all_labels_teeth = torch.cat(all_labels_teeth, dim=0)
    all_preds_jaw = torch.cat(all_preds_jaw, dim=0)
    all_labels_jaw = torch.cat(all_labels_jaw, dim=0)

    # calculate metrics
    avg_metrics = calculate_classification_metrics(all_preds_teeth, all_labels_teeth)
    avg_metrics.update(calculate_jaw_accuracy(all_preds_jaw, all_labels_jaw))
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics

# changed validate
def validate(model, dataloader, criterion_teeth, criterion_jaw, device):
    model.eval()
    total_loss = 0
    all_preds_teeth, all_labels_teeth = [], []
    all_preds_jaw, all_labels_jaw = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            labels_teeth = labels[:, :NUM_TEETH_PER_JAW]
            labels_jaw = labels[:, NUM_TEETH_PER_JAW]

            logits = model(imgs)
            logits_teeth = logits[:, :NUM_TEETH_PER_JAW]
            logits_jaw = logits[:, NUM_TEETH_PER_JAW]

            # calculate losses separately
            loss_teeth = criterion_teeth(logits_teeth, labels_teeth) # Dynamit loss
            loss_jaw = criterion_jaw(logits_jaw, labels_jaw)         # BCE loss
            loss = loss_teeth + loss_jaw
            
            total_loss += loss.item()
            
            all_preds_teeth.append(torch.sigmoid(logits_teeth))
            all_labels_teeth.append(labels_teeth)
            all_preds_jaw.append(torch.sigmoid(logits_jaw))
            all_labels_jaw.append(labels_jaw)

    all_preds_teeth = torch.cat(all_preds_teeth, dim=0)
    all_labels_teeth = torch.cat(all_labels_teeth, dim=0)
    all_preds_jaw = torch.cat(all_preds_jaw, dim=0)
    all_labels_jaw = torch.cat(all_labels_jaw, dim=0)

    avg_metrics = calculate_classification_metrics(all_preds_teeth, all_labels_teeth)
    avg_metrics.update(calculate_jaw_accuracy(all_preds_jaw, all_labels_jaw))
    avg_metrics['loss'] = total_loss / len(dataloader)

    # Return all predictions and labels
    return avg_metrics, all_preds_teeth, all_labels_teeth, all_preds_jaw, all_labels_jaw

# --- plot_training_curves ---
def plot_training_curves(history, save_dir, plot_filename):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # set up subplots
    fig, axes = plt.subplots(2, 3, figsize=(22, 12)) 
    fig.suptitle('2D ResNet Training Metrics (Dynamit Loss, 17 Out)', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # F1 Score (Micro - Teeth)
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1 (Teeth)')
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1 (Teeth)')
    axes[0, 1].set_title('F1 Score (Micro - Teeth)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Jaw Accuracy
    axes[0, 2].plot(epochs, history.get('train_jaw_acc', []), 'b-', label='Train Jaw Acc')
    axes[0, 2].plot(epochs, history.get('val_jaw_acc', []), 'r-', label='Val Jaw Acc')
    axes[0, 2].set_title('Jaw Classification Accuracy')
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    # Accuracy (Teeth)
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train Acc (Teeth)')
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Val Acc (Teeth)')
    axes[1, 0].set_title('Accuracy (Teeth)')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Precision - teeth
    axes[1, 1].plot(epochs, history['train_precision'], 'b--', label='Train Precision')
    axes[1, 1].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    axes[1, 1].set_title('Precision (Teeth)'); 
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # Recall - teeth
    axes[1, 2].plot(epochs, history['train_recall'], 'b:', label='Train Recall')
    axes[1, 2].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    axes[1, 2].set_title('Recall (Teeth)'); 
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / plot_filename
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"\n Training plots saved to: {save_path}")

def main():
    global available_gpus, device
    set_seed(SEED)
    
    print("\n[0/5] Detecting free GPUs...")
    available_gpus = get_free_gpus(); device = torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device: {device}")
    
    print("\n[1/5] Loading dataset...")
    # added data augmentations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = Tooth2DDataset(
        img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
        json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER],
        transform=transform
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print("\n[2/5] Initializing ResNet model...")

    model = ResNetMultiLabel(backbone=BACKBONE, num_outputs=NUM_OUTPUTS).to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)

 
    criterion_teeth = Dynamit_Loss(device)
    criterion_jaw = nn.BCEWithLogitsLoss().to(device)
    print(" Using Dynamit_Loss for teeth and BCEWithLogitsLoss for jaw.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # history add jaw_acc
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 
               'train_acc': [], 'val_acc': [], 'train_precision': [], 'val_precision': [], 
               'train_recall': [], 'val_recall': [], 'train_jaw_acc': [], 'val_jaw_acc': []}

    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs..."); print("=" * 80)
    best_f1 = 0.0
    best_val_preds_teeth = None
    best_val_targets_teeth = None

    for epoch in range(NUM_EPOCHS):
        # train
        train_metrics = train_epoch(model, train_loader, criterion_teeth, criterion_jaw, optimizer, device)
        val_metrics, val_preds_teeth, val_targets_teeth, val_preds_jaw, val_labels_jaw = validate(model, val_loader, criterion_teeth, criterion_jaw, device)
        scheduler.step()

        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics.get(key.replace('acc', 'accuracy'), 0))
            history[f'val_{key}'].append(val_metrics.get(key.replace('acc', 'accuracy'), 0))
        history['train_jaw_acc'].append(train_metrics.get('jaw_accuracy', 0))
        history['val_jaw_acc'].append(val_metrics.get('jaw_accuracy', 0))

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1 (teeth): {val_metrics['f1']:.4f} | Val Jaw Acc: {val_metrics['jaw_accuracy']:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_val_preds_teeth = val_preds_teeth
            best_val_targets_teeth = val_targets_teeth
            torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        →  Best F1 model saved (F1: {val_metrics['f1']:.4f})")

    print("\n" + "=" * 80 + f"\n[4/5] Training complete!\nBest validation F1 (teeth micro): {best_f1:.4f}")


    # print("\n" + "=" * 80 + "\n[4.5/5] Calculating per-tooth metrics on best model...")
    # --------------------------------------------------

    print(f"\n[5/5] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n All done!")

if __name__ == "__main__":
    main()