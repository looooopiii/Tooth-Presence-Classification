import json
import random
import subprocess
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from PIL import Image
from tqdm import tqdm

matplotlib.use('Agg')

# ==================== CONFIGURATION ====================
JSON_ROOT_LOWER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/lower"
JSON_ROOT_UPPER = "/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/upper"

IMG_ROOT_LOWER = "/home/user/lzhou/week4/multi_views/lowerjaw"
IMG_ROOT_UPPER = "/home/user/lzhou/week4/multi_views/upperjaw"

OUTPUT_DIR = "/home/user/lzhou/week11/output/Train2D/normal"
PLOT_DIR = "/home/user/lzhou/week11/output/Train2D/normal/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILENAME = "resnet18_bce_best_2d.pth"
LAST_MODEL_FILENAME = "resnet18_bce_last_2d.pth"
PLOT_FILENAME = "resnet18_bce_training_metrics_2d.png"
METRICS_FILENAME = "detailed_metrics_2d.json"
LABEL_EXPORT_FILENAME = "training_labels_with_jaw.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 1e-3
IMG_SIZE = 320
NUM_TEETH = 16  # per jaw
NUM_OUTPUTS = NUM_TEETH + 1  # 16 teeth + 1 jaw classifier
SEED = 41
BACKBONE = "resnet18"  # or "resnet50"

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4

JAW_UPPER_LABEL = 0
JAW_LOWER_LABEL = 1
UPPER_FDI_ORDER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_FDI_ORDER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

BALANCE_MIN_SAMPLES = 100
BALANCE_MAX_SAMPLES = 150
JAW_LOSS_WEIGHT = 1.0

VALID_FDI_LABELS = sorted([
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48
])
FDI_TO_INDEX = {fdi_label: i for i, fdi_label in enumerate(VALID_FDI_LABELS)}

available_gpus = []
available_gpu_info = []
device = None
# =======================================================


def get_free_gpus(min_free_mb=1500, max_gpus=2):
    if not torch.cuda.is_available():
        return []
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        entries = []
        for line in lines:
            parts = [token.strip() for token in line.split(',')]
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            if idx >= gpu_count:
                continue
            used = int(parts[1])
            total = int(parts[2])
            free = total - used
            entries.append({'index': idx, 'used': used, 'total': total, 'free': free})
        if not entries:
            return []
        entries.sort(key=lambda item: item['free'], reverse=True)
        viable = [item for item in entries if item['free'] >= min_free_mb]
        selected = viable if viable else entries
        if not viable:
            best = entries[0]
            if best['free'] < 512:
                print("Warning: No GPU with >=512MB free; falling back to CPU.")
                return []
            print(f"Warning: Using least busy GPU {best['index']} with {best['free']}MB free.")
        chosen = selected[:max_gpus]
        print(f" Free GPUs detected: {[item['index'] for item in chosen]}")
        return chosen
    except Exception as exc:
        print(f"Error detecting free GPUs: {exc}\nFalling back to CPU.")
        return []


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tooth2DDataset(Dataset):
    """
    Returns a unified jaw dataset with 16 tooth outputs and a jaw label.
    Each sample corresponds to one rendered jaw top-view image.
    """
    def __init__(self, img_roots, json_roots, transform=None,
                 balance=True, balance_min=BALANCE_MIN_SAMPLES, balance_max=BALANCE_MAX_SAMPLES):
        self.transform = transform
        lower_img_root, upper_img_root = [Path(p) for p in img_roots]
        lower_json_root, upper_json_root = [Path(p) for p in json_roots]

        self.samples = []
        self._load_samples(lower_img_root, lower_json_root, "lower")
        self._load_samples(upper_img_root, upper_json_root, "upper")
        print(f"[Info] Loaded {len(self.samples)} base 2D samples")

        if balance and self.samples:
            self._apply_balancing(balance_min, balance_max)

        self._assign_unique_ids()
        print(f"[Info] Final dataset size: {len(self.samples)}")

    def _load_samples(self, img_root: Path, json_root: Path, jaw: str):
        if not img_root.exists():
            print(f"[warn] Image root missing: {img_root}")
            return
        for png in sorted(img_root.glob("*_top.png")):
            case_id, inferred_jaw = self._parse_case_and_jaw(png.stem)
            if inferred_jaw != jaw:
                continue
            json_path = json_root / case_id / f"{case_id}_{jaw}.json"
            if not json_path.exists():
                continue
            labels_set = self._load_labels(json_path)
            labels16 = self._create_missing_vector(labels_set, jaw)
            if labels16 is None:
                continue
            sample = {
                'img': str(png),
                'json': str(json_path),
                'case_id': case_id,
                'jaw': jaw,
                'jaw_label': JAW_LOWER_LABEL if jaw == "lower" else JAW_UPPER_LABEL,
                'labels16': labels16,
                'fdi_order': np.array(LOWER_FDI_ORDER if jaw == "lower" else UPPER_FDI_ORDER, dtype=np.int32),
            }
            self.samples.append(sample)

    @staticmethod
    def _parse_case_and_jaw(stem: str):
        name = stem[:-4] if stem.endswith("_top") else stem
        if name.endswith("_lower"):
            return name[:-6], "lower"
        if name.endswith("_upper"):
            return name[:-6], "upper"
        parts = name.split("_")
        if len(parts) >= 2 and parts[-1] in ("lower", "upper"):
            return "_".join(parts[:-1]), parts[-1]
        return name, ""

    @staticmethod
    def _load_labels(json_path: Path):
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        labels = data.get('labels', [])
        if isinstance(labels, dict):
            labels = list(labels.values())
        return {int(fdi) for fdi in labels if str(fdi).isdigit()}

    def _create_missing_vector(self, present_labels, jaw: str):
        present32 = np.zeros(len(VALID_FDI_LABELS), dtype=np.float32)
        for fdi in present_labels:
            idx = FDI_TO_INDEX.get(int(fdi))
            if idx is not None:
                present32[idx] = 1.0
        missing32 = 1.0 - present32
        order = UPPER_FDI_ORDER if jaw == "upper" else LOWER_FDI_ORDER if jaw == "lower" else None
        if order is None:
            return None
        vec = np.zeros(NUM_TEETH, dtype=np.float32)
        for pos, fdi in enumerate(order):
            vec[pos] = missing32[FDI_TO_INDEX[fdi]]
        return vec

    def _apply_balancing(self, min_samples, max_samples):
        base_counts = self._compute_tooth_counts()
        print(f"[Info] Initial missing counts per position: {base_counts.tolist()}")
        if not base_counts.any():
            print("[warn] No missing-teeth labels detected; skipping balancing.")
            return

        base_samples = self.samples
        rng = random.Random(SEED)

        tooth_to_indices = defaultdict(list)
        for idx, sample in enumerate(base_samples):
            for pos, flag in enumerate(sample['labels16']):
                if flag >= 0.5:
                    tooth_to_indices[pos].append(idx)

        missing_positions = [pos for pos in range(NUM_TEETH) if not tooth_to_indices[pos]]
        if missing_positions:
            print(f"[warn] No positive samples found for tooth positions: {[pos + 1 for pos in missing_positions]}")

        target_counts = []
        for pos in range(NUM_TEETH):
            available = len(tooth_to_indices[pos])
            if available == 0:
                target_counts.append(0)
                continue
            capped = min(max_samples, available)
            target_counts.append(max(min_samples, capped))

        sample_positive_counts = [int(sample['labels16'].sum()) for sample in base_samples]

        counts = np.zeros(NUM_TEETH, dtype=np.int32)
        balanced_samples = []
        used_once = [False] * len(base_samples)
        clone_counter = 0

        def append_sample(sample, source_idx, is_clone):
            nonlocal clone_counter
            new_sample = {
                'img': sample['img'],
                'json': sample['json'],
                'case_id': sample['case_id'],
                'jaw': sample['jaw'],
                'jaw_label': sample['jaw_label'],
                'labels16': sample['labels16'].copy(),
                'fdi_order': sample['fdi_order'].copy(),
                'balanced_clone': is_clone or sample.get('balanced_clone', False),
                'source_index': source_idx,
            }
            balanced_samples.append(new_sample)
            for pos_idx, flag in enumerate(new_sample['labels16']):
                if flag >= 0.5:
                    counts[pos_idx] += 1
            if is_clone:
                clone_counter += 1

        # Phase 1: include unique samples that help fill deficits without exceeding caps.
        base_indices = list(range(len(base_samples)))
        base_indices.sort(key=lambda idx: (sample_positive_counts[idx], rng.random()))
        for idx in base_indices:
            sample = base_samples[idx]
            positives = [pos for pos, flag in enumerate(sample['labels16']) if flag >= 0.5]
            if not positives:
                continue
            if all(counts[pos] >= target_counts[pos] for pos in positives):
                continue
            if any(target_counts[pos] == 0 for pos in positives):
                continue
            if any(counts[pos] >= max_samples for pos in positives):
                continue
            append_sample(sample, idx, is_clone=False)
            used_once[idx] = True

        # Phase 2: oversample to meet minimum requirements.
        def deficits():
            return [pos for pos in range(NUM_TEETH) if target_counts[pos] > 0 and counts[pos] < target_counts[pos]]

        attempt_budget = {pos: len(tooth_to_indices[pos]) * 4 + 1 for pos in range(NUM_TEETH)}
        deficit_positions = deficits()
        while deficit_positions:
            progress = False
            for pos in sorted(deficit_positions, key=lambda p: counts[p]):
                indices = tooth_to_indices[pos]
                if not indices:
                    continue
                indices_sorted = sorted(indices, key=lambda idx: (sample_positive_counts[idx], rng.random()))
                appended = False
                for base_idx in indices_sorted:
                    sample = base_samples[base_idx]
                    positives = [p for p, flag in enumerate(sample['labels16']) if flag >= 0.5]
                    if any(target_counts[p] > 0 and counts[p] >= max_samples for p in positives if p != pos):
                        continue
                    append_sample(sample, base_idx, is_clone=used_once[base_idx])
                    used_once[base_idx] = True
                    appended = True
                    progress = True
                    break
                if appended:
                    continue
                attempt_budget[pos] -= 1
                if attempt_budget[pos] <= 0:
                    if indices:
                        base_idx = rng.choice(indices)
                        sample = base_samples[base_idx]
                        append_sample(sample, base_idx, is_clone=used_once[base_idx])
                        used_once[base_idx] = True
                        progress = True
            if not progress:
                break
            deficit_positions = deficits()

        # Phase 3: trim any remaining oversupply beyond max_samples while keeping min_samples.
        trimmed = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(balanced_samples) - 1, -1, -1):
                sample = balanced_samples[i]
                positives = [pos for pos, flag in enumerate(sample['labels16']) if flag >= 0.5]
                if not positives:
                    continue
                if any(counts[pos] <= min_samples for pos in positives if target_counts[pos] > 0):
                    continue
                if all(counts[pos] > max_samples for pos in positives if target_counts[pos] > 0):
                    for pos in positives:
                        counts[pos] -= 1
                    balanced_samples.pop(i)
                    trimmed += 1
                    changed = True
                    break

        if not balanced_samples:
            print("[warn] Balancing produced an empty dataset; reverting to original samples.")
            return

        self.samples = balanced_samples
        final_counts = self._compute_tooth_counts()
        if trimmed:
            print(f"[Info] Trimmed {trimmed} samples to respect max cap.")
        if clone_counter:
            print(f"[Info] Added {clone_counter} augmented copies during balancing.")

        out_of_range = [pos + 1 for pos in range(NUM_TEETH)
                        if final_counts[pos] and (final_counts[pos] < min_samples or final_counts[pos] > max_samples)]
        if out_of_range:
            print(f"[warn] Unable to keep all tooth positions within [{min_samples}, {max_samples}]. "
                  f"Positions affected: {out_of_range}")
        print(f"[Info] Balanced missing counts per position: {final_counts.tolist()}")

    def _compute_tooth_counts(self):
        counts = np.zeros(NUM_TEETH, dtype=np.int32)
        for sample in self.samples:
            counts += sample['labels16'].astype(np.int32)
        return counts

    def _assign_unique_ids(self):
        for idx, sample in enumerate(self.samples):
            sample['uid'] = f"{sample['case_id']}_{sample['jaw']}_{idx:05d}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        teeth_vec = torch.from_numpy(sample['labels16']).float()
        jaw_label = torch.tensor(sample['jaw_label'], dtype=torch.float32)
        return img, teeth_vec, jaw_label

    def get_missing_counts(self):
        return self._compute_tooth_counts().tolist()

    def export_labels(self, export_path: Path):
        records = []
        for sample in self.samples:
            record = {
                'sample_id': sample['uid'],
                'case_id': sample['case_id'],
                'jaw': sample['jaw'],
                'jaw_label': sample['jaw_label'],
            }
            order = sample['fdi_order']
            for pos, fdi in enumerate(order):
                record[f'missing_{fdi}'] = int(sample['labels16'][pos])
            records.append(record)
        df = pd.DataFrame(records)
        df.to_csv(export_path, index=False)
        print(f"\n Exported balanced label table with jaw column -> {export_path}")


class ResNetMultiLabel(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights="IMAGENET1K_V2")
        else:
            net = resnet18(weights="IMAGENET1K_V1")
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, NUM_OUTPUTS)
        self.net = net

    def forward(self, x):
        return self.net(x)


def calculate_teeth_metrics(pred, target):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    pred_bin = (pred > 0.5).astype(int)
    target_bin = target.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0
    )
    accuracy = accuracy_score(target_bin.flatten(), pred_bin.flatten())
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def calculate_jaw_metrics(pred, target):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    pred_bin = (pred > 0.5).astype(int)
    target_bin = target.astype(int)
    accuracy = accuracy_score(target_bin, pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_bin, pred_bin, average='binary', zero_division=0
    )
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def calculate_per_tooth_metrics(preds, targets, jaw_targets):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(jaw_targets):
        jaw_targets = jaw_targets.cpu().numpy()

    aggregator = defaultdict(lambda: {'pred': [], 'target': []})
    for i in range(len(preds)):
        order = UPPER_FDI_ORDER if int(jaw_targets[i]) == JAW_UPPER_LABEL else LOWER_FDI_ORDER
        for pos, fdi in enumerate(order):
            aggregator[fdi]['pred'].append(preds[i, pos])
            aggregator[fdi]['target'].append(targets[i, pos])

    per_fdi = OrderedDict()
    macro_collect = []
    for fdi in sorted(aggregator.keys()):
        pred_arr = np.array(aggregator[fdi]['pred'])
        target_arr = np.array(aggregator[fdi]['target']).astype(int)
        if target_arr.size == 0:
            continue
        pred_bin = (pred_arr > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_arr, pred_bin, average='binary', zero_division=0
        )
        accuracy = accuracy_score(target_arr, pred_bin)
        support = int(target_arr.sum())
        per_fdi[fdi] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'support': support
        }
        macro_collect.append((precision, recall, f1, accuracy))

    if macro_collect:
        macro_precision = float(np.mean([m[0] for m in macro_collect]))
        macro_recall = float(np.mean([m[1] for m in macro_collect]))
        macro_f1 = float(np.mean([m[2] for m in macro_collect]))
        macro_accuracy = float(np.mean([m[3] for m in macro_collect]))
    else:
        macro_precision = macro_recall = macro_f1 = macro_accuracy = 0.0

    return per_fdi, {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy
    }


def train_epoch(model, dataloader, teeth_criterion, jaw_criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    teeth_preds, teeth_targets = [], []
    jaw_preds, jaw_targets = [], []

    for imgs, teeth_labels, jaw_labels in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        teeth_labels = teeth_labels.to(device)
        jaw_labels = jaw_labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        teeth_logits = logits[:, :NUM_TEETH]
        jaw_logits = logits[:, NUM_TEETH:].squeeze(-1)

        loss_teeth = teeth_criterion(teeth_logits, teeth_labels)
        loss_jaw = jaw_criterion(jaw_logits, jaw_labels)
        loss = loss_teeth + JAW_LOSS_WEIGHT * loss_jaw
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        teeth_preds.append(torch.sigmoid(teeth_logits).detach().cpu())
        teeth_targets.append(teeth_labels.detach().cpu())
        jaw_preds.append(torch.sigmoid(jaw_logits).detach().cpu())
        jaw_targets.append(jaw_labels.detach().cpu())

    teeth_preds = torch.cat(teeth_preds, dim=0)
    teeth_targets = torch.cat(teeth_targets, dim=0)
    jaw_preds = torch.cat(jaw_preds, dim=0)
    jaw_targets = torch.cat(jaw_targets, dim=0)

    teeth_metrics = calculate_teeth_metrics(teeth_preds, teeth_targets)
    jaw_metrics = calculate_jaw_metrics(jaw_preds, jaw_targets)
    teeth_metrics.update({
        'loss': total_loss / max(len(dataloader), 1),
        'jaw_accuracy': jaw_metrics['accuracy'],
        'jaw_precision': jaw_metrics['precision'],
        'jaw_recall': jaw_metrics['recall'],
        'jaw_f1': jaw_metrics['f1'],
    })
    return teeth_metrics


def validate(model, dataloader, teeth_criterion, jaw_criterion, device):
    model.eval()
    total_loss = 0.0
    teeth_preds, teeth_targets = [], []
    jaw_preds, jaw_targets = [], []
    with torch.no_grad():
        for imgs, teeth_labels, jaw_labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            teeth_labels = teeth_labels.to(device)
            jaw_labels = jaw_labels.to(device)

            logits = model(imgs)
            teeth_logits = logits[:, :NUM_TEETH]
            jaw_logits = logits[:, NUM_TEETH:].squeeze(-1)

            loss_teeth = teeth_criterion(teeth_logits, teeth_labels)
            loss_jaw = jaw_criterion(jaw_logits, jaw_labels)
            loss = loss_teeth + JAW_LOSS_WEIGHT * loss_jaw
            total_loss += loss.item()

            teeth_preds.append(torch.sigmoid(teeth_logits).cpu())
            teeth_targets.append(teeth_labels.cpu())
            jaw_preds.append(torch.sigmoid(jaw_logits).cpu())
            jaw_targets.append(jaw_labels.cpu())

    teeth_preds = torch.cat(teeth_preds, dim=0)
    teeth_targets = torch.cat(teeth_targets, dim=0)
    jaw_preds = torch.cat(jaw_preds, dim=0)
    jaw_targets = torch.cat(jaw_targets, dim=0)

    teeth_metrics = calculate_teeth_metrics(teeth_preds, teeth_targets)
    jaw_metrics = calculate_jaw_metrics(jaw_preds, jaw_targets)
    teeth_metrics.update({
        'loss': total_loss / max(len(dataloader), 1),
        'jaw_accuracy': jaw_metrics['accuracy'],
        'jaw_precision': jaw_metrics['precision'],
        'jaw_recall': jaw_metrics['recall'],
        'jaw_f1': jaw_metrics['f1'],
    })
    return teeth_metrics, teeth_preds, teeth_targets, jaw_preds, jaw_targets


def plot_training_curves(history, save_dir, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('2D ResNet Training Metrics (16 Teeth + Jaw)', fontsize=16, fontweight='bold')

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[0, 1].set_title('Teeth F1'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[0, 2].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[0, 2].set_title('Teeth Accuracy'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_precision'], 'b--', label='Train Precision')
    axes[1, 0].plot(epochs, history['val_precision'], 'r--', label='Val Precision')
    axes[1, 0].set_title('Teeth Precision'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['train_recall'], 'b:', label='Train Recall')
    axes[1, 1].plot(epochs, history['val_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Teeth Recall'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history['train_jaw_acc'], 'b-', label='Train Jaw Acc')
    axes[1, 2].plot(epochs, history['val_jaw_acc'], 'r-', label='Val Jaw Acc')
    axes[1, 2].set_title('Jaw Accuracy'); axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n Training plots saved to: {save_path}")


def main():
    global available_gpus, available_gpu_info, device
    set_seed(SEED)

    print("\n[0/6] Detecting free GPUs...")
    gpu_entries = get_free_gpus()
    available_gpu_info = gpu_entries
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available() and gpu_count > 0 and gpu_entries:
        valid_entries = [item for item in gpu_entries if item['index'] < gpu_count]
        if not valid_entries:
            available_gpus = []
            device = torch.device("cpu")
            print("Warning: Filtered GPU list empty after validation; falling back to CPU.")
        else:
            available_gpus = [item['index'] for item in valid_entries]
            device = torch.device(f"cuda:{available_gpus[0]}")
    else:
        available_gpus = []
        available_gpu_info = []
        device = torch.device("cpu")
    print(f"Primary device: {device}")

    print("\n[1/6] Building dataset with unified jaws and balancing...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Tooth2DDataset(
        img_roots=[IMG_ROOT_LOWER, IMG_ROOT_UPPER],
        json_roots=[JSON_ROOT_LOWER, JSON_ROOT_UPPER],
        transform=transform,
        balance=True
    )
    print(f"Missing counts per position after balancing: {dataset.get_missing_counts()}")

    label_export_path = Path(OUTPUT_DIR) / LABEL_EXPORT_FILENAME
    dataset.export_labels(label_export_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    effective_batch_size = min(BATCH_SIZE, max(1, len(dataset)))
    free_mb = None
    if device.type == 'cuda' and available_gpu_info:
        free_mb = available_gpu_info[0]['free']
        if free_mb < 1500:
            effective_batch_size = min(effective_batch_size, 16)
        if free_mb < 900:
            effective_batch_size = min(effective_batch_size, 8)
        if free_mb < 600:
            effective_batch_size = min(effective_batch_size, 4)
        if free_mb < 300:
            effective_batch_size = max(1, min(effective_batch_size, 2))
    if effective_batch_size < BATCH_SIZE:
        if free_mb is not None:
            print(f"[info] Adjusting batch size to {effective_batch_size} based on available GPU memory ({free_mb} MB free).")
        else:
            print(f"[info] Adjusting batch size to {effective_batch_size} due to limited dataset size.")

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, effective_batch_size), shuffle=False, num_workers=4, pin_memory=True)

    print("\n[2/6] Initializing 2D model...")
    model = ResNetMultiLabel(backbone=BACKBONE).to(device)
    if torch.cuda.is_available() and len(available_gpus) > 1:
        model = nn.DataParallel(model, device_ids=available_gpus)

    teeth_criterion = nn.BCEWithLogitsLoss()
    jaw_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_jaw_acc': [], 'val_jaw_acc': [],
    }

    print(f"\n[3/6] Starting training for {NUM_EPOCHS} epochs..."); print("=" * 80)
    best_f1 = 0.0
    best_val_preds = None
    best_val_targets = None
    best_val_jaw_preds = None
    best_val_jaw_targets = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, teeth_criterion, jaw_criterion, optimizer, device)
        val_metrics, val_preds, val_targets, val_jaw_preds, val_jaw_targets = validate(
            model, val_loader, teeth_criterion, jaw_criterion, device
        )
        scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_jaw_acc'].append(train_metrics['jaw_accuracy'])
        history['val_jaw_acc'].append(val_metrics['jaw_accuracy'])

        print(
            f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Jaw Acc: {val_metrics['jaw_accuracy']:.4f}"
        )

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / LAST_MODEL_FILENAME)

        if val_metrics['f1'] > best_f1 + MIN_DELTA:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_val_preds = val_preds.clone()
            best_val_targets = val_targets.clone()
            best_val_jaw_preds = val_jaw_preds.clone()
            best_val_jaw_targets = val_jaw_targets.clone()
            torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict()}, Path(OUTPUT_DIR) / BEST_MODEL_FILENAME)
            print(f"        → Best F1 model saved (F1: {best_f1:.4f}) to {BEST_MODEL_FILENAME}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("        → Early stopping triggered.")
                break

    print("\n" + "=" * 80 + f"\n[4/6] Training complete! Best validation F1: {best_f1:.4f} (epoch {best_epoch + 1})")

    if best_val_preds is None:
        print("[warn] No validation improvement recorded; using last epoch outputs.")
        _, best_val_preds, best_val_targets, best_val_jaw_preds, best_val_jaw_targets = validate(
            model, val_loader, teeth_criterion, jaw_criterion, device
        )

    print("\n" + "=" * 80 + "\n[5/6] Calculating per-tooth metrics on best model...")
    per_tooth_metrics, macro_metrics = calculate_per_tooth_metrics(best_val_preds, best_val_targets, best_val_jaw_targets)
    jaw_metrics = calculate_jaw_metrics(best_val_jaw_preds, best_val_jaw_targets)

    print("\n MACRO-AVERAGED METRICS (across all teeth):")
    print("-" * 80)
    print(f"  Macro Precision: {macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {macro_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {macro_metrics['macro_f1']:.4f}")
    print(f"  Macro Accuracy:  {macro_metrics['macro_accuracy']:.4f}")

    print("\n PER-TOOTH METRICS (FDI Notation):")
    print("-" * 80)
    print(f"{'FDI Tooth':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
    print("-" * 80)
    for fdi_label, metrics in per_tooth_metrics.items():
        print(
            f"Tooth {fdi_label:<5} {metrics['precision']:>10.4f}   "
            f"{metrics['recall']:>10.4f}   {metrics['f1']:>10.4f}   "
            f"{metrics['accuracy']:>10.4f}   {metrics['support']:>8}"
        )

    print("\n JAW CLASSIFICATION METRICS:")
    print("-" * 80)
    print(f"  Accuracy: {jaw_metrics['accuracy']:.4f}")
    print(f"  Precision: {jaw_metrics['precision']:.4f}")
    print(f"  Recall:    {jaw_metrics['recall']:.4f}")
    print(f"  F1 Score:  {jaw_metrics['f1']:.4f}")

    metrics_file = Path(OUTPUT_DIR) / METRICS_FILENAME
    with open(metrics_file, 'w') as fp:
        serializable_per_tooth = {str(k): v for k, v in per_tooth_metrics.items()}
        json.dump({
            'macro_metrics': macro_metrics,
            'per_tooth_metrics': serializable_per_tooth,
            'jaw_metrics': jaw_metrics,
            'missing_counts_per_position': dataset.get_missing_counts()
        }, fp, indent=2)
    print(f"\n Detailed metrics saved to: {metrics_file}")

    print("\n[6/6] Generating training plots...")
    plot_training_curves(history, PLOT_DIR, PLOT_FILENAME)
    print("\n All done with retraining!")


if __name__ == "__main__":
    main()
