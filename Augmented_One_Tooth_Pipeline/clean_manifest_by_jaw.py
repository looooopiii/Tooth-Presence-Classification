import pandas as pd
from pathlib import Path

UPPER = {f"tooth_{i}" for i in range(11,19)} | {f"tooth_{i}" for i in range(21,29)}
LOWER = {f"tooth_{i}" for i in range(31,39)} | {f"tooth_{i}" for i in range(41,49)}

def main(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    kept, dropped = 0, 0

    def filt(row):
        nonlocal kept, dropped
        jaw = str(row["jaw"]).lower().strip()
        valid = UPPER if jaw=="upper" else LOWER
        items = [t.strip() for t in str(row.get("removed_teeth","")).split(",") if t.strip()]
        filtered = [t for t in items if t in valid]
        if len(filtered) != len(items): dropped += 1
        else: kept += 1
        return ",".join(filtered)

    df["removed_teeth"] = df.apply(filt, axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved -> {out_csv}")
    print(f"[Info] rows unchanged: {kept}, rows had invalid-teeth removed: {dropped}")

if __name__ == "__main__":
    main(
        in_csv="/home/user/lzhou/week8/IOS/manifest_ios.csv",
        out_csv="/home/user/lzhou/week8/IOS/manifest_ios_clean.csv"
    )