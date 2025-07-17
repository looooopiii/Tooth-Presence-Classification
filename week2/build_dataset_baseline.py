import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ===== path =====
JSON_ROOT = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")
IMAGE_ROOT = Path("/home/user/lzhou/blender-5views") 
OUTPUT_CSV = str(Path.home() / "baseline_dataset_front_only.csv")
OUTPUT_JSON = str(Path.home() / "baseline_dataset_front_only.json")

# Fixed perspective, only the front diagram is extracted
TARGET_VIEW = "front"

def extract_32bit_presence(label_array, jaw):
    # Extraction of 32-bit presence vectors from upper and lower teeth (FDI coding to 1~32)
    if jaw == "upper":
        FDI_TO_INDEX = {
            18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
            21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16
        }
    elif jaw == "lower":
        FDI_TO_INDEX = {
            38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
            41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32
        }
    else:
        raise ValueError("Invalid jaw type")

    presence = [0] * 32
    for fdi in label_array:
        if fdi in FDI_TO_INDEX:
            presence[FDI_TO_INDEX[fdi] - 1] = 1
    return presence

def process_one(json_path: Path, jaw: str):
    # Process a single JSON file and return the front image path + label
    with open(json_path, "r") as f:
        data = json.load(f)

    patient_id = data["id_patient"]
    labels = data["labels"]
    label_32 = extract_32bit_presence(labels, jaw)

    obj_name = f"{patient_id}_{jaw}"  # e.g. 0AAQ6BO3_lower
    image_path = IMAGE_ROOT / jaw / patient_id / f"{obj_name}_{TARGET_VIEW}.png"

    if not image_path.exists():
        print(f"Missing: {image_path}")
        return None

    return {
        "id": obj_name,
        "image_path": str(image_path),
        "label": label_32
    }

def main():
    all_rows = []

    for jaw in ["lower", "upper"]:
        json_dir = JSON_ROOT / jaw
        for json_file in tqdm(list(json_dir.rglob("*.json")), desc=f"Processing {jaw}"):
            result = process_one(json_file, jaw)
            if result:
                all_rows.append(result)

    if not all_rows:
        print("No valid front-view samples found.")
        return

    # Output CSV
    df = pd.DataFrame([{"image_path": r["image_path"], "label": r["label"]} for r in all_rows])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved to {OUTPUT_CSV} (Total samples: {len(df)})")

    # Output JSON
    json_output = [{"id": r["id"], "images": [r["image_path"]], "label": r["label"]} for r in all_rows]
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to {OUTPUT_JSON} (Total samples: {len(json_output)})")

if __name__ == "__main__":
    main()

    # === Label distribution statistics ===
    print("\nPer-tooth label distribution:")
    # Use main() to check if the CSV file exists and read it
    try:
        df  # type: ignore
    except NameError:
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV)
    import numpy as np
    labels_array = df["label"].apply(eval).tolist()
    labels_np = np.array(labels_array)
    ones = labels_np.sum(axis=0)
    zeros = labels_np.shape[0] - ones

    for i in range(32):
        print(f"Tooth {i+1:02d}: 1s = {int(ones[i]):4d}, 0s = {int(zeros[i]):4d}")