import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
JSON_ROOT = Path("/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split")  # root dir with upper/lower
IMAGE_ROOT = Path("/home/user/lzhou/blender-5views")  # folder storing 5-view rendered images
OUT_PATH = Path("/home/user/lzhou/5views-json/dataset.json")  # output file

VIEWS = ["top", "bottom", "front", "left", "right"]

def extract_32bit_presence(label_array):
    """Extracts 32-bit label presence from 'labels' array"""
    return [1 if i in label_array else 0 for i in range(1, 33)]

def process_one(json_path: Path):
    # Read JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    patient_id = data["id_patient"]
    jaw = data["jaw"]  # 'upper' or 'lower'
    labels = data["labels"]
    
    obj_name = f"{patient_id}_{jaw}"  # e.g., 0AAQ6BO3_lower
    label_32 = extract_32bit_presence(labels)

    # Locate image files
    image_dir = IMAGE_ROOT / jaw / patient_id
    image_paths = [str(image_dir / f"{obj_name}_{v}.png") for v in VIEWS]

    if not all(Path(p).exists() for p in image_paths):
        print(f"⚠️ Skipping {obj_name} due to missing images")
        return None
    
    return {
        "id": obj_name,
        "images": image_paths,
        "label": label_32
    }

def main():
    all_samples = []

    for jaw in ["upper", "lower"]:
        for json_path in tqdm((JSON_ROOT / jaw).rglob("*.json")):
            result = process_one(json_path)
            if result:
                all_samples.append(result)

    # Save to JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n Done. Total samples: {len(all_samples)}")
    print(f" Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()