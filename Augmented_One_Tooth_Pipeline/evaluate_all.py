import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms

def _infer_backbone_from_state_dict(sd: dict) -> str:
    """
    Infer resnet backbone variant from a torchvision-style state_dict.
    Returns one of: 'resnet18', 'resnet34', 'resnet50'
    Heuristics:
      - If any key matches 'layer\d\.\d\.conv3.weight' => bottleneck => resnet50 (2048-d head).
      - Else basic block family (conv1/conv2 only):
          * If there exists a block index >=2 in any layer (e.g., 'layer2.2.' or 'layer3.2.') -> resnet34 (3/4/6/3 blocks),
            otherwise -> resnet18 (2/2/2/2 blocks).
    """
    keys = list(sd.keys())
    # normalize possible 'model.' or 'base.' prefixes
    def norm(k):
        for pref in ("model.", "base."):
            if k.startswith(pref):
                return k[len(pref):]
        return k
    nkeys = [norm(k) for k in keys]
    # 1) bottleneck check: conv3 exists in bottleneck blocks (resnet50/101)
    if any((".conv3.weight" in k and k.startswith("layer")) for k in nkeys):
        return "resnet50"
    # 2) basic block family: inspect max block index per layer
    # if any layer has index >=2 (0-based), it's likely resnet34
    import re
    pat = re.compile(r"^layer([1-4])\.(\d+)\.")
    max_idx = {1:-1,2:-1,3:-1,4:-1}
    for k in nkeys:
        m = pat.match(k)
        if m:
            li = int(m.group(1))
            bi = int(m.group(2))
            if bi > max_idx[li]:
                max_idx[li] = bi
    # resnet34 has blocks per layer: [3,4,6,3] -> max indices [2,3,5,2]
    # resnet18 has blocks per layer: [2,2,2,2] -> max indices [1,1,1,1]
    if any(max_idx[l] >= 2 for l in (1,2,3,4)):
        return "resnet34"
    return "resnet18"

def build_model_from_ckpt(ckpt, dim: int, device: torch.device):
    """
    Build a torchvision resnet model that matches the checkpoint backbone,
    replace its fc to `dim`, and load weights.
    """
    sd = ckpt.get("model", ckpt)
    # try to find fc weight to sanity-check feature dims (512 vs 2048)
    # but rely primarily on block-type heuristic above
    backbone = _infer_backbone_from_state_dict(sd)
    if backbone == "resnet18":
        m = resnet18(weights=None)
    elif backbone == "resnet34":
        m = resnet34(weights=None)
    else:
        m = resnet50(weights=None)
    # replace head
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, dim)
    # attempt strict load; if fails, fall back to non-strict
    try:
        m.load_state_dict(sd, strict=True)
        strict_msg = "strict=True"
    except Exception as e:
        m.load_state_dict(sd, strict=False)
        strict_msg = f"strict=False (fallback: {type(e).__name__})"
    print(f"[eval] Detected backbone: {backbone} | fc_in={in_features} | load {strict_msg}")
    return m.to(device).eval()

def infer_jaw(row):
    jaw = str(row.get("jaw","")).lower()
    if jaw in ("upper","lower"): return jaw
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

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab_path: Path, json_root: Path, size=512):
        self.df = df.reset_index(drop=True)
        self.json_root = json_root
        self.idx2tooth = sorted(json.loads(Path(vocab_path).read_text()))
        self.tooth2idx = {t:i for i,t in enumerate(self.idx2tooth)}
        self.dim = len(self.idx2tooth)
        self.tf = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)

        y = np.zeros(self.dim, dtype=np.float32)
        removed = [t.strip() for t in str(row.get("removed_teeth","")).split(",") if t.strip().startswith("tooth_")]
        for t in removed:
            if t in self.tooth2idx: y[self.tooth2idx[t]] = 1.0

        jaw = infer_jaw(row)
        pres = present_set(row["case_id"], jaw, self.json_root)
        m = np.zeros(self.dim, dtype=np.float32)
        for t in pres:
            if t in self.tooth2idx: m[self.tooth2idx[t]] = 1.0

        return x, y, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("./runs/all"))
    ap.add_argument("--json_root", type=Path, default=Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split"))
    ap.add_argument("--vocab", type=Path, default=Path("/home/user/lzhou/week8/data_augmentation/Aug_data/vocab_all.json"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.outdir/"best.pt", map_location="cpu")
    vocab = ckpt["vocab"]; dim = len(vocab)

    # load test split
    test_manifest = args.outdir/"test_manifest.csv"
    df_test = pd.read_csv(test_manifest)

    ds = EvalDataset(df_test, args.vocab, args.json_root, size=args.size)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(ckpt, dim, device)

    Ys, Ps, Ms = [], [], []
    with torch.no_grad():
        for x,y,m in dl:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            Ys.append(y.numpy()); Ps.append(p); Ms.append(m.numpy())
    import numpy as np
    Y = np.concatenate(Ys); P = np.concatenate(Ps); M = np.concatenate(Ms).astype(bool)

    # overall micro-F1 (threshold=0.5, masked)
    from sklearn.metrics import f1_score, precision_score, recall_score
    Yhat = (P>=0.5).astype(int)
    micro_f1  = f1_score(Y[M], Yhat[M], average="micro", zero_division=0)
    micro_prec= precision_score(Y[M], Yhat[M], average="micro", zero_division=0)
    micro_rec = recall_score(Y[M], Yhat[M], average="micro", zero_division=0)
    print(f"Test micro: P={micro_prec:.4f} R={micro_rec:.4f} F1={micro_f1:.4f}")

    # per-class metrics & per-class threshold tuning
    idx2tooth = sorted(vocab)
    rows=[]
    for c, tooth in enumerate(idx2tooth):
        valid = M[:,c]
        if valid.sum()==0:
            rows.append({"tooth": tooth, "P":0,"R":0,"F1":0,"best_F1":0,"best_th":0,"support":0})
            continue
        y, p = Y[valid, c], P[valid, c]
        yhat = (p>=0.5).astype(int)
        Pm = precision_score(y, yhat, zero_division=0)
        Rm = recall_score(y, yhat, zero_division=0)
        Fm = f1_score(y, yhat, zero_division=0)
        # scan thresholds
        bestF, bestT = 0, 0.5
        for th in np.linspace(0.1, 0.9, 17):
            f = f1_score(y, (p>=th).astype(int), zero_division=0)
            if f>bestF: bestF, bestT = f, th
        rows.append({"tooth": tooth, "P":Pm, "R":Rm, "F1":Fm, "best_F1":bestF, "best_th":bestT, "support": int(valid.sum())})

    rep = pd.DataFrame(rows).sort_values("tooth")
    rep_path = args.outdir/"test_report_per_tooth.csv"
    rep.to_csv(rep_path, index=False)
    print(f"Saved per-tooth report -> {rep_path}")

if __name__ == "__main__":
    main()