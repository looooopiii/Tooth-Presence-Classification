import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
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

# ============= CONFIGURATION =============
DATA_SOURCES_CSV = [
    ("/home/user/tbrighton/blender_outputs/augment_test/train_labels_augmented.csv", Path("/home/user/tbrighton/blender_outputs/augment_test")),
    ("/home/user/tbrighton/blender_outputs/augment_random/train_labels_random.csv", Path("/home/user/tbrighton/blender_outputs/augment_random"))
]
ORIGINAL_DATA_PATHS = [
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower",
    "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"
]
OUTPUT_DIR = Path("/home/user/tbrighton/Scripts/Training/3D/trained_models")
PLOT_DIR = Path("/home/user/tbrighton/Scripts/Training/3D/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "dynamit_loss_full_dataset_best.pth"
LAST_MODEL_FILENAME = "dynamit_loss_full_dataset_last.pth"
PLOT_FILENAME = "dynamit_loss_full_dataset_metrics.png"
METRICS_FILENAME = "dynamit_loss_full_dataset_detailed_metrics.json"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_POINTS = 4096
NUM_TEETH = 32
SEED = 41

VALID_FDI_LABELS = sorted([18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48])
FDI_TO_INDEX = {fdi: i for i, fdi in enumerate(VALID_FDI_LABELS)}
INDEX_TO_FDI = {i: fdi for fdi, i in FDI_TO_INDEX.items()}

available_gpus = []; device = None
# =========================================


# =========================================
# MODELS AND LOSS FUNCTION (WITH FIX)
# =========================================
class Dynamit_Loss(nn.Module):
    """Numerically stable Dynamit Loss that avoids the UserWarning."""
    def __init__(self, device):
        super(Dynamit_Loss, self).__init__()
        self.device = device

    def forward(self, predictions, targets):
        S_pos = (targets == 1).sum().float()
        S_neg = (targets == 0).sum().float()
        epsilon = 1e-8
        
        # Keep calculations as tensors and detach at the end
        pos_coeff_raw = S_neg / (S_pos + epsilon)
        neg_coeff_raw = S_pos / (S_neg + epsilon)
        
        pos_coeff = torch.clamp(pos_coeff_raw, max=1.0).detach()
        neg_coeff = torch.clamp(neg_coeff_raw, max=1.0).detach()
        
        weights = torch.where(targets == 1, pos_coeff, neg_coeff)
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weights)

class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super(PointNetEncoder, self).__init__()
        self.conv1=nn.Conv1d(input_dim,64,1); self.bn1=nn.BatchNorm1d(64); self.conv2=nn.Conv1d(64,128,1); self.bn2=nn.BatchNorm1d(128); self.conv3=nn.Conv1d(128,256,1); self.bn3=nn.BatchNorm1d(256); self.conv4=nn.Conv1d(256,512,1); self.bn4=nn.BatchNorm1d(512); self.conv5=nn.Conv1d(512,feature_dim,1); self.bn5=nn.BatchNorm1d(feature_dim)
    def forward(self, x):
        x=x.transpose(2,1); x=F.relu(self.bn1(self.conv1(x))); x=F.relu(self.bn2(self.conv2(x))); x=F.relu(self.bn3(self.conv3(x))); x=F.relu(self.bn4(self.conv4(x))); x=F.relu(self.bn5(self.conv5(x))); return torch.max(x,2)[0]

class ToothClassificationModel(nn.Module):
    def __init__(self, num_teeth=32, feature_dim=1024):
        super(ToothClassificationModel, self).__init__(); self.encoder=PointNetEncoder(input_dim=3,feature_dim=feature_dim); self.fc1=nn.Linear(feature_dim,512); self.bn1=nn.BatchNorm1d(512); self.dropout1=nn.Dropout(0.3); self.fc2=nn.Linear(512,256); self.bn2=nn.BatchNorm1d(256); self.dropout2=nn.Dropout(0.3); self.fc3=nn.Linear(256,num_teeth)
    def forward(self, x):
        features=self.encoder(x); x=self.dropout1(F.relu(self.bn1(self.fc1(features)))); x=self.dropout2(F.relu(self.bn2(self.fc2(x)))); return self.fc3(x)


# =========================================
# DATASET CLASSES
# =========================================
class CSVToothDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_points=4096):
        self.root_dir = Path(root_dir); self.num_points = num_points; self.df = pd.read_csv(csv_file)
        self.labels_df = self.df[[str(fdi) for fdi in VALID_FDI_LABELS]].astype(np.float32)
    def __len__(self): return len(self.df)
    def _load_and_process_points(self, obj_path):
        vertices = [];
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '): vertices.append([float(p) for p in line.strip().split()[1:4]])
        except FileNotFoundError: return np.zeros((self.num_points, 3), dtype=np.float32)
        points = np.array(vertices, dtype=np.float32)
        if len(points)==0: return np.zeros((self.num_points, 3), dtype=np.float32)
        centroid = np.mean(points, axis=0); points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)));
        if max_dist > 0: points /= max_dist
        indices = np.random.choice(len(points), self.num_points, replace=len(points) < self.num_points)
        return points[indices]
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; obj_path = self.root_dir / row['filename']
        points = self._load_and_process_points(obj_path)
        labels = self.labels_df.iloc[idx].values
        return torch.from_numpy(points).float(), torch.from_numpy(labels).float()

class OriginalToothDataset(Dataset):
    def __init__(self, data_paths, num_points=4096):
        self.num_points=num_points; self.samples=[]
        for data_path_str in data_paths:
            path = Path(data_path_str)
            if not path.exists(): continue
            jaw_type = "lower" if "lower" in str(path) else "upper"
            for case_dir in sorted(path.iterdir()):
                if case_dir.is_dir():
                    obj, json_f = case_dir/f"{case_dir.name}_{jaw_type}.obj", case_dir/f"{case_dir.name}_{jaw_type}.json"
                    if obj.exists() and json_f.exists(): self.samples.append({'obj': str(obj), 'json': str(json_f)})
    def __len__(self): return len(self.samples)
    def _load_and_process_points(self, obj_path):
        vertices = [];
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '): vertices.append([float(p) for p in line.strip().split()[1:4]])
        points = np.array(vertices, dtype=np.float32)
        if len(points)==0: return np.zeros((self.num_points, 3), dtype=np.float32)
        centroid = np.mean(points, axis=0); points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0: points /= max_dist
        indices = np.random.choice(len(points), self.num_points, replace=len(points)<self.num_points)
        return points[indices]
    def _load_labels_from_json(self, json_path):
        with open(json_path, 'r') as f: data = json.load(f)
        present_teeth_fdi = {label for label in data.get('labels', []) if label != 0}
        labels = np.ones(NUM_TEETH, dtype=np.float32)
        for fdi in present_teeth_fdi:
            if fdi in FDI_TO_INDEX: labels[FDI_TO_INDEX[fdi]] = 0.0
        return labels
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = self._load_and_process_points(sample['obj'])
        labels = self._load_labels_from_json(sample['json'])
        return torch.from_numpy(points).float(), torch.from_numpy(labels).float()


# =========================================
# HELPER AND TRAINING FUNCTIONS (READABLE)
# =========================================
def get_free_gpus(threshold_mb=1000, max_gpus=2):
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,nounits,noheader'],capture_output=True,text=True)
        gpus = [int(line.split(', ')[0]) for line in r.stdout.strip().split('\n') if int(line.split(', ')[1]) < threshold_mb]
        if len(gpus)>max_gpus: gpus=gpus[:max_gpus]
        if not gpus: print("Warning: No free GPUs found, using GPU 0"); return [0]
        print(f"✓ Free GPUs detected: {gpus}"); return gpus
    except Exception as e: print(f"Error detecting GPUs: {e}\nFalling back to GPU 0"); return [0]

def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def calculate_classification_metrics(predictions, targets):
    preds_binary = (predictions > 0.5).cpu().numpy(); targets_np = targets.cpu().numpy()
    preds_flat, targets_flat = preds_binary.flatten(), targets_np.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(targets_flat, preds_flat, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'acc': accuracy_score(targets_flat, preds_flat)}

def calculate_per_tooth_metrics(predictions, targets):
    preds_binary, targets_np = (predictions>0.5).cpu().numpy(), targets.cpu().numpy()
    metrics = OrderedDict()
    for i in range(NUM_TEETH):
        fdi = INDEX_TO_FDI[i]
        p, r, f1, _ = precision_recall_fscore_support(targets_np[:, i], preds_binary[:, i], average='binary', zero_division=0)
        acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
        metrics[fdi] = {'precision': p, 'recall': r, 'f1': f1, 'accuracy': acc, 'support': int(targets_np[:, i].sum())}
    macro_metrics = {'macro_precision':np.mean([m['precision'] for m in metrics.values()]), 'macro_recall':np.mean([m['recall'] for m in metrics.values()]), 'macro_f1':np.mean([m['f1'] for m in metrics.values()]), 'macro_accuracy':np.mean([m['accuracy'] for m in metrics.values()])}
    return metrics, macro_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train(); total_loss = 0.0; all_preds, all_labels = [], []
    for points, labels in tqdm(dataloader, desc="Training", leave=False):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits.detach())); all_labels.append(labels)
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval(); total_loss = 0.0; all_preds, all_targets = [], []
    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Validating", leave=False):
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits)); all_targets.append(labels)
    metrics = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics, torch.cat(all_preds), torch.cat(all_targets)

def plot_training_curves(history, save_dir):
    epochs=range(1,len(history['train_loss'])+1); fig,ax=plt.subplots(2,2,figsize=(15,12)); fig.suptitle('PointNet Training Metrics (Full Dataset)',fontsize=16,fontweight='bold')
    ax[0,0].plot(epochs,history['train_loss'],'b-',label='Train Loss'); ax[0,0].plot(epochs,history['val_loss'],'r-',label='Val Loss'); ax[0,0].set_title('Loss'); ax[0,0].legend(); ax[0,0].grid(True,alpha=0.3)
    ax[0,1].plot(epochs,history['train_f1'],'b-',label='Train F1'); ax[0,1].plot(epochs,history['val_f1'],'r-',label='Val F1'); ax[0,1].set_title('F1 Score (Micro)'); ax[0,1].legend(); ax[0,1].grid(True,alpha=0.3)
    ax[1,0].plot(epochs,history['train_acc'],'b-',label='Train Acc'); ax[1,0].plot(epochs,history['val_acc'],'r-',label='Val Acc'); ax[1,0].set_title('Accuracy'); ax[1,0].legend(); ax[1,0].grid(True,alpha=0.3)
    ax[1,1].plot(epochs,history['train_precision'],'b--',label='Train Precision'); ax[1,1].plot(epochs,history['val_precision'],'r--',label='Val Precision'); ax[1,1].plot(epochs,history['train_recall'],'b:',label='Train Recall'); ax[1,1].plot(epochs,history['val_recall'],'r:',label='Val Recall'); ax[1,1].set_title('Precision & Recall'); ax[1,1].legend(); ax[1,1].grid(True,alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.96]); sp=Path(save_dir)/PLOT_FILENAME; plt.savefig(sp,dpi=300); plt.close(); print(f"\n✓ Plots saved: {sp}")


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    global available_gpus, device; set_seed(SEED)
    print("\n[0/5] Setting up environment..."); available_gpus=get_free_gpus(); device=torch.device(f"cuda:{available_gpus[0]}" if torch.cuda.is_available() else "cpu"); print(f"Primary device: {device}")
    print("\n[1/5] Loading and combining datasets..."); datasets=[OriginalToothDataset(ORIGINAL_DATA_PATHS)]; [datasets.append(CSVToothDataset(csv,root)) for csv,root in DATA_SOURCES_CSV if Path(csv).exists()]; full_dataset=ConcatDataset(datasets); print(f"✓ Combined dataset loaded with {len(full_dataset)} samples.")
    train_size=int(0.9*len(full_dataset)); val_size=len(full_dataset)-train_size; train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size],generator=torch.Generator().manual_seed(SEED))
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,pin_memory=True,drop_last=True); val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=8,pin_memory=True)
    print("\n[2/5] Initializing model..."); model=ToothClassificationModel().to(device)
    if len(available_gpus)>1: model=torch.nn.DataParallel(model,device_ids=available_gpus); print(f"✓ Model wrapped for {len(available_gpus)} GPUs.")
    criterion=Dynamit_Loss(device); optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01); scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS)
    history={'train_loss':[],'val_loss':[],'train_f1':[],'val_f1':[],'train_acc':[],'val_acc':[],'train_precision':[],'val_precision':[],'train_recall':[],'val_recall':[]}; best_f1=0.0
    print(f"\n[3/5] Starting training for {NUM_EPOCHS} epochs..."); print("="*80)
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        for key in ['loss', 'f1', 'acc', 'precision', 'recall']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        print(f"E {epoch+1:2d}/{NUM_EPOCHS}|Train L:{train_metrics['loss']:.4f}, F1:{train_metrics['f1']:.4f}|Val L:{val_metrics['loss']:.4f}, F1:{val_metrics['f1']:.4f}")
        model_to_save=model.module if hasattr(model,'module') else model; torch.save({'epoch':epoch,'model_state_dict':model_to_save.state_dict()},OUTPUT_DIR/LAST_MODEL_FILENAME)
        if val_metrics['f1']>best_f1: best_f1=val_metrics['f1']; bvp,bvt=val_preds,val_targets; torch.save({'epoch':epoch,'model_state_dict':model_to_save.state_dict()},OUTPUT_DIR/BEST_MODEL_FILENAME); print(f"  → ✓ Best F1 model saved (F1: {val_metrics['f1']:.4f})")
    print("\n"+"="*80+f"\n[4/5] Training complete! Best F1: {best_f1:.4f}"); print("\n[4.5/5] Calculating metrics..."); ptm,mm=calculate_per_tooth_metrics(bvp,bvt)
    print("\n📊 MACRO-AVERAGED METRICS:");print("-" * 80);print(f"  Macro Precision: {mm['macro_precision']:.4f}\n  Macro Recall:    {mm['macro_recall']:.4f}\n  Macro F1:        {mm['macro_f1']:.4f}\n  Macro Accuracy:  {mm['macro_accuracy']:.4f}")
    print("\n🦷 PER-TOOTH METRICS:");print("-" * 80);print(f"{'FDI':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Accuracy':<12}{'Support':<10}");print("-" * 80)
    for fdi,mets in ptm.items(): print(f"  {fdi:<8} {mets['precision']:>10.4f}  {mets['recall']:>10.4f}  {mets['f1']:>10.4f}  {mets['accuracy']:>10.4f}  {mets['support']:>8}")
    mf=OUTPUT_DIR/METRICS_FILENAME; 
    with open(mf,'w') as f: json.dump({'macro_metrics':mm,'per_tooth_metrics':{str(k):v for k,v in ptm.items()}},f,indent=2); print(f"\n✓ Metrics saved: {mf}")
    print(f"\n[5/5] Generating plots..."); plot_training_curves(history,PLOT_DIR); print("\n✓ All done!")

if __name__ == "__main__":
    main()
