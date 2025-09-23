import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
import re
import glob
import os

# --- helpers to restrict teeth by jaw (FDI numbering) ---
_TOOTH_RE = re.compile(r"^tooth_(\d{2})$")

def _tooth_code(t: str) -> int | None:
    m = _TOOTH_RE.match(str(t))
    return int(m.group(1)) if m else None

def _is_upper(code: int) -> bool:
    # upper arches: 11-18 (UR) and 21-28 (UL)
    return (11 <= code <= 18) or (21 <= code <= 28)

def _is_lower(code: int) -> bool:
    # lower arches: 31-38 (LL) and 41-48 (LR)
    return (31 <= code <= 38) or (41 <= code <= 48)

def build_jaw_mask(idx2tooth: list[str], jaw: str) -> np.ndarray:
    """Return boolean mask (len = dim) where True means this tooth belongs to the given jaw."""
    jaw = (jaw or "").lower()
    ok = np.zeros(len(idx2tooth), dtype=np.bool_)
    for i, t in enumerate(idx2tooth):
        c = _tooth_code(t)
        if c is None: 
            continue
        if jaw == "upper" and _is_upper(c):
            ok[i] = True
        elif jaw == "lower" and _is_lower(c):
            ok[i] = True
        # if jaw is unknown, keep all False; we will fallback later
    return ok

def resolve_image_path(p: Path, case_id: str, jaw: str, view: str) -> Path | None:
    """
    Try to robustly resolve an image path:
    - exact path
    - try different file extensions
    - try common filename variations (double spaces, PNG vs png)
    - try patterns with optional numeric suffix after Upper/LowerJawScan (e.g., UpperJawScan001)
    - final fallback: glob by case_id + jaw token + view in the same directory
    """
    if p.exists():
        return p

    parent = p.parent
    stem = p.stem  # e.g., "12345_2022-09-14 UpperJawScan_top"
    name = p.name
    base, ext = os.path.splitext(name)
    # 1) try extension variants
    for ext2 in (".png", ".PNG", ".jpg", ".JPG"):
        q = parent / (base + ext2)
        if q.exists():
            return q

    # 2) collapse multiple spaces
    base2 = " ".join(base.split())
    if base2 != base:
        for ext2 in (ext, ".png", ".PNG"):
            q = parent / (base2 + ext2)
            if q.exists():
                return q

    # 3) try with numeric suffix after JawScan (e.g., UpperJawScan001)
    jaw_token = "UpperJawScan" if jaw == "upper" else "LowerJawScan"
    if jaw_token in base:
        for k in range(1, 6):  # try 001..005
            cand = base.replace(jaw_token, f"{jaw_token}{k:03d}")
            q = parent / (cand + ext)
            if q.exists():
                return q
            for ext2 in (".png", ".PNG", ".jpg", ".JPG"):
                qq = parent / (cand + ext2)
                if qq.exists():
                    return qq

    # 4) final glob search in the same directory
    # patterns like: "{case_id} *UpperJawScan*_top.*"
    pattern = f"{case_id} *{jaw_token}*_{view}.*"
    hits = list(parent.glob(pattern))
    if hits:
        # pick the first (or most recent) match
        hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return hits[0]

    return None

def _infer_backbone_from_state_dict(sd: dict) -> str:
    def norm(k):
        for pref in ("model.","base."):
            if k.startswith(pref): return k[len(pref):]
        return k
    nkeys = [norm(k) for k in sd.keys()]
    if any((".conv3.weight" in k and k.startswith("layer")) for k in nkeys):
        return "resnet50"
    pat = re.compile(r"^layer([1-4])\.(\d+)\.")
    max_idx = {1:-1,2:-1,3:-1,4:-1}
    for k in nkeys:
        m = pat.match(k)
        if m:
            li = int(m.group(1)); bi = int(m.group(2))
            max_idx[li] = max(max_idx[li], bi)
    if any(max_idx[l] >= 2 for l in (1,2,3,4)): return "resnet34"
    return "resnet18"

def build_model_from_ckpt(ckpt, dim: int, device: torch.device):
    sd = ckpt.get("model", ckpt)
    bb = _infer_backbone_from_state_dict(sd)
    if bb == "resnet18": m = resnet18(weights=None)
    elif bb == "resnet34": m = resnet34(weights=None)
    else: m = resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, dim)
    try:
        m.load_state_dict(sd, strict=True); msg="strict=True"
    except Exception as e:
        m.load_state_dict(sd, strict=False); msg=f"strict=False ({type(e).__name__})"
    print(f"[eval] backbone={bb} fc_in={in_features} load {msg}")
    return m.to(device).eval()

def present_set_from_json(case_id, jaw, json_root: Path):
    jp = json_root / jaw / case_id / f"{case_id}_{jaw}.json"
    try:
        data = json.loads(jp.read_text()); labels = data.get("labels", []) or []
    except Exception:
        labels = []
    present = set()
    for v in labels:
        if isinstance(v,int) and v!=0:
            present.add(f"tooth_{v}")
    return present

class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv: Path, vocab_path: Path, size=512, json_root: Path|None=None, skip_missing: bool=False):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.idx2tooth = sorted(json.loads(Path(vocab_path).read_text()))
        self.tooth2idx = {t:i for i,t in enumerate(self.idx2tooth)}
        self.dim = len(self.idx2tooth)
        # precompute jaw masks (upper/lower) based on vocab
        self._mask_upper = build_jaw_mask(self.idx2tooth, "upper")
        self._mask_lower = build_jaw_mask(self.idx2tooth, "lower")
        self.json_root = json_root
        self.skip_missing = skip_missing
        self.tf = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        jaw = str(row.get("jaw","")).strip().lower()
        # resolve image path robustly
        raw_p = Path(str(row["image_path"]))
        case_id = str(row.get("case_id", ""))
        view = "top"  # default view name used when building manifests
        p_resolved = resolve_image_path(raw_p, case_id, jaw, view)
        if p_resolved is None:
            if self.skip_missing:
                return None
            raise FileNotFoundError(f"Image not found for case={case_id} jaw={jaw}: {raw_p}")
        img = Image.open(p_resolved).convert("RGB")
        x = self.tf(img)

        y = np.zeros(self.dim, dtype=np.float32)
        removed = [t.strip() for t in str(row.get("removed_teeth","")).split(",") if t.strip().startswith("tooth_")]
        for t in removed:
            if t in self.tooth2idx: y[self.tooth2idx[t]] = 1.0

        # enforce jaw consistency on labels (defensive)
        if jaw == "upper":
            y[~self._mask_upper] = 0.0
        elif jaw == "lower":
            y[~self._mask_lower] = 0.0

        # presence mask: combine jaw mask with optional JSON presence
        if self.json_root is not None:
            case_id = str(row.get("case_id"))
            pres = present_set_from_json(case_id, jaw, self.json_root)
            m = np.zeros(self.dim, dtype=np.float32)
            for t in pres:
                if t in self.tooth2idx:
                    m[self.tooth2idx[t]] = 1.0
        else:
            # start from "all present"
            m = np.ones(self.dim, dtype=np.float32)

        # AND with jaw-consistent mask
        if jaw == "upper":
            m = (m.astype(bool) & self._mask_upper).astype(np.float32)
        elif jaw == "lower":
            m = (m.astype(bool) & self._mask_lower).astype(np.float32)

        return x, y, m

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ms = zip(*batch)
    import numpy as _np
    X = torch.stack(xs, dim=0)
    Y = torch.tensor(_np.stack(ys, axis=0), dtype=torch.float32)
    M = torch.tensor(_np.stack(ms, axis=0), dtype=torch.float32)
    return X, Y, M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="manifest_ios.csv")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Training model： best.pt")
    ap.add_argument("--vocab", type=Path, required=True, help="vocab_all.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=Path("./runs/ios_eval"))
    ap.add_argument("--json_root", type=Path, default=None,
                    help="(Optional) presence JSON root directory; if not provided, mask=all 1")
    ap.add_argument("--skip_missing", action="store_true", help="Skip samples whose image_path cannot be found (instead of raising)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]; dim = len(vocab)

    ds = ExternalDataset(args.manifest, args.vocab, size=args.size, json_root=args.json_root, skip_missing=args.skip_missing)
    print(f"[info] vocab size={ds.dim} | jaw-masked evaluation enabled (upper/lower)")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=safe_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(ckpt, dim, device)

    Ys, Ps, Ms = [], [], []
    with torch.no_grad():
        for batch in dl:
            if batch is None:  # all missing in this batch
                continue
            x,y,m = batch
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            Ys.append(y.numpy()); Ps.append(p); Ms.append(m.numpy())
    Y = np.concatenate(Ys); P = np.concatenate(Ps); M = np.concatenate(Ms).astype(bool)

    from sklearn.metrics import f1_score, precision_score, recall_score
    Yhat = (P>=0.5).astype(int)
    micro_f1  = f1_score(Y[M], Yhat[M], average="micro", zero_division=0)
    micro_pr  = precision_score(Y[M], Yhat[M], average="micro", zero_division=0)
    micro_re  = recall_score(Y[M], Yhat[M], average="micro", zero_division=0)
    print(f"[IOS] micro: P={micro_pr:.4f} R={micro_re:.4f} F1={micro_f1:.4f}")

    # per-tooth report
    rows=[]; idx2tooth = sorted(vocab)
    for c, t in enumerate(idx2tooth):
        valid = M[:,c]
        if valid.sum()==0:
            rows.append({"tooth": t, "P":0,"R":0,"F1":0,"best_F1":0,"best_th":0,"support":0})
            continue
        y, p = Y[valid, c], P[valid, c]
        yhat = (p>=0.5).astype(int)
        Pm = precision_score(y, yhat, zero_division=0)
        Rm = recall_score(y, yhat, zero_division=0)
        Fm = f1_score(y, yhat, zero_division=0)
        bestF, bestT = 0, 0.5
        for th in np.linspace(0.1,0.9,17):
            f = f1_score(y, (p>=th).astype(int), zero_division=0)
            if f>bestF: bestF, bestT = f, th
        rows.append({"tooth": t, "P":Pm, "R":Rm, "F1":Fm, "best_F1":bestF, "best_th":bestT, "support": int(valid.sum())})
    rep = pd.DataFrame(rows).sort_values("tooth")
    rep_path = args.outdir / "ios_report_per_tooth.csv"
    rep.to_csv(rep_path, index=False)
    print(f"[IOS] per-tooth report -> {rep_path}")

if __name__ == "__main__":
    main()