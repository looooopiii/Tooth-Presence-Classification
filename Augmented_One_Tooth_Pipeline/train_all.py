import argparse, json, random, os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50, resnet18
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

# ---------- Model builder ----------
def build_model(backbone: str, num_classes: int):
    backbone = (backbone or "resnet18").lower()
    if backbone == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2")
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    elif backbone == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1")
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

# ---------- Dataset utils ----------
def infer_jaw(row):
    jaw = str(row.get("jaw", "")).lower()
    if jaw in ("upper","lower"):
        return jaw
    p = (str(row.get("image_path","")) + " " + str(row.get("obj_path","")) + " " + str(row.get("case_id",""))).lower()
    if "_upper" in p: return "upper"
    if "_lower" in p: return "lower"
    return "lower"

def present_set(case_id, jaw, json_root: Path):
    jp = json_root / jaw / case_id / f"{case_id}_{jaw}.json"
    try:
        data = json.loads(jp.read_text())
        labels = data.get("labels", []) or []
    except Exception:
        labels = []
    present = set()
    for v in labels:
        if isinstance(v,int) and v!=0:
            present.add(f"tooth_{v}")
    return present

class TeethDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab_path: Path, json_root: Path, size=512, train=True):
        self.df = df.reset_index(drop=True)
        self.json_root = json_root
        self.idx2tooth = sorted(json.loads(Path(vocab_path).read_text()))
        self.tooth2idx = {t:i for i,t in enumerate(self.idx2tooth)}
        self.dim = len(self.idx2tooth)
        aug=[]
        if train:
            aug += [transforms.ColorJitter(0.08,0.08,0.08,0.02),
                    transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05))]
        self.tf = transforms.Compose(aug + [
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)

        y = torch.zeros(self.dim, dtype=torch.float32)
        removed = [t.strip() for t in str(row.get("removed_teeth","")).split(",") if t.strip().startswith("tooth_")]
        for t in removed:
            if t in self.tooth2idx: y[self.tooth2idx[t]] = 1.0

        jaw = infer_jaw(row)
        pres = present_set(row["case_id"], jaw, self.json_root)
        m = torch.zeros(self.dim, dtype=torch.float32)
        for t in pres:
            if t in self.tooth2idx: m[self.tooth2idx[t]] = 1.0

        return x, y, m

# ---------- Loss ----------
def make_criterion(pos_weight_npy: Path, device):
    pw = torch.tensor(np.load(pos_weight_npy), dtype=torch.float32).to(device)
    base = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    def masked_bce(logits, targets, mask):
        # logits/targets/mask: [B, C]
        per_class = base(logits, targets)         # [B,C]
        masked = per_class * mask
        denom = mask.sum().clamp(min=1.0)
        return masked.sum() / denom
    return masked_bce

# ---------- Split ----------
def split_by_case(df: pd.DataFrame, seed=42, tr=0.7, va=0.15):
    cases = sorted(df["case_id"].unique())
    random.Random(seed).shuffle(cases)
    n = len(cases); ntr = int(tr*n); nva = int(va*n)
    train_ids = set(cases[:ntr])
    val_ids   = set(cases[ntr:ntr+nva])
    test_ids  = set(cases[ntr+nva:])
    return (df[df.case_id.isin(train_ids)].reset_index(drop=True),
            df[df.case_id.isin(val_ids  )].reset_index(drop=True),
            df[df.case_id.isin(test_ids )].reset_index(drop=True))

# ---------- Evaluation ----------
def eval_micro_f1(model, dl, device):
    from sklearn.metrics import f1_score
    model.eval()
    Ys, Ps, Ms = [], [], []
    with torch.no_grad():
        for x,y,m in dl:
            x,y,m = x.to(device), y.to(device), m.to(device)
            p = torch.sigmoid(model(x))
            Ys.append(y.cpu().numpy()); Ps.append(p.cpu().numpy()); Ms.append(m.cpu().numpy())
    import numpy as np
    Y = np.concatenate(Ys); P = np.concatenate(Ps); M = np.concatenate(Ms).astype(bool)
    Yhat = (P>=0.5).astype(int)
    return f1_score(Y[M], Yhat[M], average="micro", zero_division=0)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/home/user/lzhou/week8/data_augmentation/Aug_data/manifest_all.csv"))
    ap.add_argument("--json_root", type=Path, default=Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split"))
    ap.add_argument("--vocab", type=Path, default=Path("/home/user/lzhou/week8/data_augmentation/Aug_data/vocab_all.json"))
    ap.add_argument("--pos_weight", type=Path, default=Path("/home/user/lzhou/week8/data_augmentation/Aug_data/pos_weight_all.npy"))
    ap.add_argument("--outdir", type=Path, default=Path("./runs/all"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if None)")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--amp", action="store_true", help="use mixed precision (fp16 autocast)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # data
    df = pd.read_csv(args.manifest)
    df_train, df_val, df_test = split_by_case(df, seed=args.seed)
    df_train.to_csv(args.outdir/"train_manifest.csv", index=False)
    df_val.to_csv(args.outdir/"val_manifest.csv", index=False)
    df_test.to_csv(args.outdir/"test_manifest.csv", index=False)

    train_ds = TeethDataset(df_train, args.vocab, args.json_root, size=args.size, train=True)
    val_ds   = TeethDataset(df_val,   args.vocab, args.json_root, size=args.size, train=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dim = len(json.loads(Path(args.vocab).read_text()))
    model = build_model(args.backbone, dim).to(device)

    criterion = make_criterion(args.pos_weight, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=(args.amp and device.type=="cuda"))

    best_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for it, (x,y,m) in enumerate(train_dl, 1):
            x,y,m = x.to(device), y.to(device), m.to(device)
            if scaler.is_enabled():
                with autocast(dtype=torch.float16):
                    logit = model(x)
                    loss = criterion(logit, y, m) / max(1, args.accum)
                scaler.scale(loss).backward()
                if it % max(1, args.accum) == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                logit = model(x)
                loss = criterion(logit, y, m) / max(1, args.accum)
                loss.backward()
                if it % max(1, args.accum) == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

        val_f1 = eval_micro_f1(model, val_dl, device)
        print(f"Epoch {epoch:02d}  val_microF1(masked)= {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model": model.state_dict(),
                "vocab": json.loads(Path(args.vocab).read_text()),
                "size": args.size,
            }, args.outdir/"best.pt")
            print(f"  -> saved best.pt (val_microF1={best_f1:.4f})")

if __name__ == "__main__":
    main()