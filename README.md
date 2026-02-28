# Tooth-Presence-Classification

## Overview
This repository contains the scripts and outputs for the UZH master project on Tooth Presence Classification. Two main pipelines are implemented:
- `lzhou/` contains the 2D image workflow: jaw meshes into PNGs via Blender, data augmentation, dataset checks, and ResNet scripts for 32-tooth and 16+1 classifiers.
- `tbrighton/` focuses on 3D point-cloud processing: data augmentation that edits OBJ label files, PointNet-style training/evaluation code for 16+1- and 36-neuron heads.

All scripts read from the 3DTeethSeg22 challenge dataset at this path:
```
/local/scratch/datasets/Medical/TeethSeg/3DTeethLand_challenge_train_test_split/
    upper/<CASE>/<CASE>_upper.obj & .json
    lower/<CASE>/<CASE>_lower.obj & .json
```
Update the `JSON_ROOT_*`, `IMG_ROOT_*`, `DATA_PATHS`, `OUTPUT_DIR`, and similar constants at the top of each script if the datasets live elsewhere.

## Repository Layout
| Path | Description |
| --- | --- |
| `lzhou/2D/` | Rendering (`Render2d`), augmentation (`Data_Augmentation`), data quality checks (`check_data`), training/testing scripts for 32-tooth models (`32neuron`) and 16+1 jaw-specific models (`16neuron`). Test label CSVs (`2D/label_flipped.csv`) live here.|
| `tbrighton/Scripts/Training` | PointNet-based 3D model training code for 16+1- and 32-neuron heads (`3D_17n`, `3D_36n`) with BCE, Dynamit, and Dynamit loss with augmented dataset and results. |
| `tbrighton/Scripts/Testing` | Evaluation processes that load checkpoints, fuse multiple rotations, and produce confusion matrices. |
| `tbrighton/Scripts/Augmenting` | OBJ/JSON augmentation scripts (`random_augmentation_fixed.py`, `test_augmentation_fixed.py`). |

## Installation and other requirements:

  **Install Python dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas scikit-learn matplotlib seaborn pillow tqdm
   ```
   - The 2D Render2d scripts must run inside Blender’s bundled Python (`bpy`, `mathutils`, `addon_utils`), like `blender -b -P lzhou/2D/Render2d/render/render_trainset.py`.
   - 3D augmentation/training scripts depend only on standard PyTorch + scientific Python packages. Multiprocessing is used for faster OBJ processing.

  **GPU setup:** Training and testing scripts attempt to auto-select free GPUs using `nvidia-smi`. Ensure CUDA-enabled PyTorch is installed and tweak `get_free_gpus` helpers if running on managed clusters.

## Reproducing Report Results
Follow the sequence below. Adjust paths inside each script as needed.

### 1. Prepare data
- **Render 2D PNGs (lzhou pipeline)**
  ```bash
  blender -b -P lzhou/2D/Render2d/render/render_trainset.py
  blender -b -P lzhou/2D/Render2d/render/render_testset_top.py
  blender -b -P lzhou/2D/Render2d/render/render_test_stl_top.py
  ```
  After data augmentation, run:
  ```bash
  blender -b -P lzhou/2D/Render2d/render/render_aug_random_top.py
  blender -b -P lzhou/2D/Render2d/render/render_aug_test_top.py
  ```
  Optional: `lzhou/2D/Render2d/auto_render_blender_5views.py` renders multi-view packs for data quality checks.
- **Generate augmentation**
  ```bash
  python lzhou/2D/Data_Augmentation/random_augmentation_fixed.py
  python lzhou/2D/Data_Augmentation/test_augmentation_fixed.py
  python tbrighton/Scripts/Augmenting/random_augmentation_fixed.py
  python tbrighton/Scripts/Augmenting/test_augmentation_fixed.py
  ```
  These scripts modify the JSON labels to simulate missing teeth and store CSV summaries (`train_labels_random.csv`, `train_labels_augmented.csv`).
- **Run dataset checks**
  ```bash
  python lzhou/2D/check_data/check_scripts/check_trainset.py
  python lzhou/2D/check_data/check_scripts/check_testset.py
  python lzhou/2D/check_data/check_scripts/Analyze_augmentation_corrected.py
  python tbrighton/Scripts/Augmenting/analyze.py
  ```
  Output goes into each script's `check_output` or analysis folder.

### 2. Train models
- **2D / 32neurons models (lzhou/2D/32neuron)**
  ```bash
  python lzhou/2D/32neuron/scripts/Train/normal/baseline_model_2d_32teeth.py
  python lzhou/2D/32neuron/scripts/Train/dynamit/baseline_dynamit_2d_32teeth.py
  python lzhou/2D/32neuron/scripts/Train/aug_dynamit/augmented_dynamit_2d_32teeth.py
  ```
- **2D / 16+1 jaw-specific models (lzhou/2D/16neuron)**
  ```bash
  python lzhou/2D/16neuron/scripts/Train_rotation/Baseline_16plus1_2d.py
  python lzhou/2D/16neuron/scripts/Train_rotation/Baseline_16plus1_dynamit_2d.py
  python lzhou/2D/16neuron/scripts/Train_rotation/16plus1_teeth_augmented_dynamit.py
  ```
- **3D / PointNet models 32 neuron (tbrighton/Scripts/Training/3D_36n)**
  ```bash
  python tbrighton/Scripts/Training/3D_36n/baseline_model_1.py
  python tbrighton/Scripts/Training/3D_36n/baseline_dynamit_1.py
  python tbrighton/Scripts/Training/3D_36n/augmented_dynamit_1.py
  
- **3D / PointNet models 16+1 neuron (tbrighton/Scripts/Training/3D_17n)**
  python tbrighton/Scripts/Training/3D_17n/baseline_model_17new.py
  python tbrighton/Scripts/Training/3D_17n/dynamit_model_17new.py
  python tbrighton/Scripts/Training/3D_17n/augmented_dynamit_17new.py
  python tbrighton/Scripts/Training/3D_17n/Wbce_model_17new.py
  ```
  Each script saves results under its `trained_models*` directories and plots under `plots*`.

### 3. Evaluate checkpoints
- **2D testing**: The scripts in `lzhou/2D/32neuron/scripts/Test/*` and `lzhou/2D/16neuron/scripts/Test_top` produce confusion matrices, detailed metrics JSON files, and qualitative samples. For the 32-neuron evaluation scripts, run:
  ```bash
  python lzhou/2D/32neuron/scripts/Test/normal/test_baseline_2d_32teeth.py
  python lzhou/2D/32neuron/scripts/Test/dynamit/test_dynamit_2d_32teeth.py
  python lzhou/2D/32neuron/scripts/Test/aug_dynamit/test_augmented_dynamit_2d_32teeth.py
  ```
  For the 16+1 models:
  ```bash
  python lzhou/2D/16neuron/scripts/Test_top/test_2d_bce_rotation.py
  python lzhou/2D/16neuron/scripts/Test_top/test_2d_dynamit_rotation.py
  python lzhou/2D/16neuron/scripts/Test_top/test_2d_Augdynamit_rotation.py
  ```
- **3D testing**: `tbrighton/Scripts/Testing/3D_17n` and `3D_36n` are 3D evaluation scripts, run:
  ```bash
  python tbrighton/Scripts/Testing/3D_17n/test_baseline_new17.py
  python tbrighton/Scripts/Testing/3D_17n/test_dynamit_new.py
  python tbrighton/Scripts/Testing/3D_17n/test_augmented_new17.py
  -----------------------------------------------------------------
  python tbrighton/Scripts/Testing/3D_36n/test_baseline_1.py
  python tbrighton/Scripts/Testing/3D_36n/test_dynamit_1.py
  python tbrighton/Scripts/Testing/3D_36n/test_augment_1.py
  ```
