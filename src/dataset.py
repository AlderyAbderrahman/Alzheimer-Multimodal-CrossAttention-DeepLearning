"""
src/dataset.py
--------------
Single responsibility: load and pair MRI volumes with clinical labels.

What this file does:
  - Scans the NIfTI folder and builds a subject_id → nii_path mapping
  - Joins with the CSV (clinical features + CDR label)
  - Drops subjects that have no matching NIfTI file
  - Preprocesses each MRI volume on-the-fly (resize, normalize, augment)
  - Returns (mri_tensor, clinical_tensor, label) for each subject

Used by train.py:
    dataset = OASISDataset(nii_dir, csv_path, preprocessor, split='train')
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from torch.utils.data import Dataset
from scipy.ndimage import zoom

from src.preprocessing import (
    FEATURE_COLS, LABEL_COL, ID_COL,
    CDR_TO_CLASS, ClinicalPreprocessor
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SHAPE = (96, 96, 96)   # must match BrainMRI3DEncoder input


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — NIfTI FOLDER SCANNER
# Builds a dict: { 'OAS1_0001' : '/path/to/OAS1_0001_MR1_...nii' }
# ─────────────────────────────────────────────────────────────────────────────

def scan_nii_folder(nii_dir: str) -> dict:
    """
    Walk the NIfTI folder and extract subject IDs from filenames.

    Filename format: OAS1_0001_MR1_mpr_n4_anon_sbj_111_normalised.nii
    Subject ID      : OAS1_0001   (first two underscore-separated parts)

    Returns
    -------
    dict { subject_id : full_path_to_nii_file }

    Example
    -------
    {
      'OAS1_0001': '/content/drive/.../OASIS/OAS1_0001_MR1_...nii',
      'OAS1_0003': '/content/drive/.../OASIS/OAS1_0003_MR1_...nii',
      ...
    }
    """
    nii_map = {}

    for root, dirs, files in os.walk(nii_dir):
        for fname in files:
            if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                continue

            # Extract subject ID: take first two parts = OAS1_XXXX
            # e.g. OAS1_0001_MR1_mpr_n4_anon_sbj_111_normalised.nii
            #       ↑ part0  ↑ part1
            parts = fname.split('_')
            if len(parts) < 2:
                continue

            subject_id = f"{parts[0]}_{parts[1]}"   # OAS1_0001

            # If a subject appears twice, keep first found
            if subject_id not in nii_map:
                nii_map[subject_id] = os.path.join(root, fname)

    print(f"NIfTI scanner   : found {len(nii_map)} subjects in {nii_dir}")
    return nii_map


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — CSV + NIfTI JOINER
# Matches each CSV row to a NIfTI file by subject ID
# ─────────────────────────────────────────────────────────────────────────────

def build_subject_list(csv_path: str, nii_dir: str) -> pd.DataFrame:
    """
    Load the CSV, extract subject IDs, join with NIfTI paths.
    Drops subjects that:
      - Have no NIfTI file
      - Have a missing or invalid CDR label

    Returns
    -------
    DataFrame with columns:
        subject_id, nii_path, CDR, Age, M/F, Educ, SES, MMSE, eTIV, nWBV

    This is the master subject list used to build train/val/test splits.
    """
    df = pd.read_csv(csv_path)

    # The CSV ID column looks like 'OAS1_0001_MR1'
    # We extract just 'OAS1_0001' to match filenames
    df['subject_id'] = df[ID_COL].str.extract(r'(OAS1_\d+)')

    # Keep only valid CDR rows
    df = df.dropna(subset=[LABEL_COL])
    df = df[df[LABEL_COL].isin(CDR_TO_CLASS)]

    # One row per subject (drop duplicate scans — keep first)
    df = df.drop_duplicates(subset='subject_id', keep='first')

    # Scan the NIfTI folder
    nii_map = scan_nii_folder(nii_dir)

    # Join: add nii_path column
    df['nii_path'] = df['subject_id'].map(nii_map)

    # Drop subjects with no matching NIfTI
    missing = df['nii_path'].isna().sum()
    if missing > 0:
        print(f"  ⚠ Dropped {missing} subjects with no NIfTI file")
    df = df.dropna(subset=['nii_path'])

    print(f"Subject list    : {len(df)} subjects with both CSV and NIfTI")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — MRI VOLUME PROCESSING
# Loads, resizes, and normalizes a single NIfTI volume
# ─────────────────────────────────────────────────────────────────────────────

def load_nii_volume(nii_path: str,
                    target_shape: tuple = TARGET_SHAPE) -> np.ndarray:
    """
    Load a NIfTI file and resize it to target_shape.

    Steps
    -----
    1. Load .nii with nibabel → raw numpy array
    2. Resize to (96, 96, 96) using scipy zoom (trilinear interpolation)
    3. Clip to [0, 99.5th percentile] to remove outlier voxels
    4. Normalize to [0, 1] (min-max per volume)

    Why normalize per volume?
    -------------------------
    MRI scanners produce arbitrary intensity scales.
    Subject A might have values in [0, 2000], subject B in [0, 800].
    Per-volume normalization puts everyone on the same [0, 1] scale.

    Returns
    -------
    np.ndarray, shape (96, 96, 96), dtype float32
    """
    # Step 1 — Load
    img  = nib.load(nii_path)
    vol  = img.get_fdata(dtype=np.float32)   # raw voxel values

    # Step 2 — Resize to target shape
    if vol.shape != target_shape:
        zoom_factors = (
            target_shape[0] / vol.shape[0],
            target_shape[1] / vol.shape[1],
            target_shape[2] / vol.shape[2],
        )
        # order=1 → trilinear interpolation (fast, smooth, good for MRI)
        vol = zoom(vol, zoom_factors, order=1)

    # Step 3 — Clip outlier voxels (bright artifacts)
    p995 = np.percentile(vol, 99.5)
    vol  = np.clip(vol, 0, p995)

    # Step 4 — Min-max normalize to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    # If vmax == vmin (blank scan), vol stays all zeros — safe

    return vol.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — AUGMENTATION
# Light 3D augmentation applied only during training
# ─────────────────────────────────────────────────────────────────────────────

def augment_volume(vol: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a (96,96,96) float32 volume.

    Augmentations used (all are label-preserving):
    ------------------------------------------------
    1. Random flip along left-right axis (p=0.5)
       Brain is roughly symmetric — flipping doesn't change CDR.

    2. Random brightness shift ± 0.1
       Simulates scanner intensity variation between sites/sessions.

    3. Random Gaussian noise (σ=0.01)
       Simulates scanner thermal noise — improves robustness.

    NOT used (would change the label or distort anatomy):
    - Rotation / elastic deformation   → changes brain shape, hurts 3D CNN
    - Zoom / crop                      → changes brain size features
    - Intensity inversion              → physically meaningless for MRI
    """
    # 1 — Random left-right flip (axis 0 = X = left-right in MNI space)
    if np.random.random() < 0.5:
        vol = np.flip(vol, axis=0).copy()

    # 2 — Random brightness ± 0.1
    brightness = np.random.uniform(-0.1, 0.1)
    vol = np.clip(vol + brightness, 0.0, 1.0)

    # 3 — Gaussian noise σ=0.01
    noise = np.random.normal(0, 0.01, vol.shape).astype(np.float32)
    vol   = np.clip(vol + noise, 0.0, 1.0)

    return vol


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — PYTORCH DATASET
# The class that DataLoader calls to get each (mri, clinical, label) triplet
# ─────────────────────────────────────────────────────────────────────────────

class OASISDataset(Dataset):
    """
    PyTorch Dataset for OASIS-1 multimodal data.

    Returns per __getitem__:
        mri      : torch.FloatTensor  (1, 96, 96, 96)  — the '1' is channel dim
        clinical : torch.FloatTensor  (7,)              — standardized features
        label    : torch.LongTensor   scalar            — 0, 1, or 2

    Usage
    -----
    # In train.py:
    from src.preprocessing import ClinicalPreprocessor
    from src.dataset import OASISDataset, build_subject_list
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    # 1. Build master subject list
    df = build_subject_list(csv_path, nii_dir)

    # 2. Split by subject ID (stratified)
    from sklearn.model_selection import train_test_split
    labels = df['CDR'].map(CDR_TO_CLASS).values
    df_tr, df_va = train_test_split(df, test_size=0.2,
                                    random_state=42, stratify=labels)

    # 3. Fit preprocessor on training clinical data only
    prep = ClinicalPreprocessor()
    prep.fit_transform(df_tr)   # fits scaler/imputer

    # 4. Build datasets
    train_ds = OASISDataset(df_tr, prep, augment=True)
    val_ds   = OASISDataset(df_va, prep, augment=False)

    # 5. Build loaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
    """

    def __init__(self,
                 df:          pd.DataFrame,
                 preprocessor: ClinicalPreprocessor,
                 augment:     bool = False,
                 target_shape: tuple = TARGET_SHAPE):
        """
        Parameters
        ----------
        df           : DataFrame from build_subject_list(), already split
        preprocessor : fitted ClinicalPreprocessor (call fit_transform on
                       training df first, then pass same object here)
        augment      : True for training split, False for val/test
        target_shape : MRI resize target, default (96,96,96)
        """
        self.df           = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.augment      = augment
        self.target_shape = target_shape

        # Pre-compute all clinical features + labels in one go
        # This is fast (pure numpy) so we do it once at init, not per __getitem__
        self.X_clinical, self.y = preprocessor.transform(df)
        # X_clinical : (N, 7)  float32
        # y          : (N,)    int64

        print(f"OASISDataset    : {len(self.df)} subjects  "
              f"| augment={augment}  "
              f"| shape={target_shape}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Called by DataLoader for each sample.

        Returns
        -------
        mri      : FloatTensor (1, 96, 96, 96)
        clinical : FloatTensor (7,)
        label    : LongTensor  scalar
        """
        # ── MRI ──────────────────────────────────────────────────────────────
        nii_path = self.df.loc[idx, 'nii_path']
        vol      = load_nii_volume(nii_path, self.target_shape)
        # vol: numpy (96, 96, 96)  float32

        if self.augment:
            vol = augment_volume(vol)

        # Add channel dim: (96,96,96) → (1,96,96,96)
        # The '1' = grayscale channel (MRI has 1 channel, unlike RGB=3)
        mri = torch.from_numpy(vol).unsqueeze(0)   # (1, 96, 96, 96)

        # ── Clinical ─────────────────────────────────────────────────────────
        clinical = torch.from_numpy(self.X_clinical[idx])   # (7,)

        # ── Label ─────────────────────────────────────────────────────────────
        label = torch.tensor(self.y[idx], dtype=torch.long)  # scalar

        return mri, clinical, label


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from google.colab import drive
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from src.preprocessing import ClinicalPreprocessor, CDR_TO_CLASS

    drive.mount('/content/drive')

    BASE    = '/content/drive/MyDrive/dataset/oasis13d/data'
    CSV     = f'{BASE}/oasis_cross-sectional.csv'
    NII_DIR = f'{BASE}/oasis/OASIS'

    # Step 1 — Build master list
    df = build_subject_list(CSV, NII_DIR)
    print(f"\nColumns: {list(df.columns)}")
    print(df[['subject_id', 'CDR', 'nii_path']].head(3).to_string())

    # Step 2 — Stratified split
    labels   = df['CDR'].map(CDR_TO_CLASS).values
    df_tr, df_va = train_test_split(df, test_size=0.2,
                                    random_state=42, stratify=labels)
    print(f"\nTrain: {len(df_tr)}  Val: {len(df_va)}")

    # Step 3 — Fit preprocessor
    prep = ClinicalPreprocessor()
    prep.fit_transform(df_tr)

    # Step 4 — Build datasets
    train_ds = OASISDataset(df_tr, prep, augment=True)
    val_ds   = OASISDataset(df_va, prep, augment=False)

    # Step 5 — Test one batch
    loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    mri, clinical, label = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"  MRI      : {tuple(mri.shape)}       ← (B, 1, 96, 96, 96)")
    print(f"  Clinical : {tuple(clinical.shape)}          ← (B, 7)")
    print(f"  Label    : {tuple(label.shape)}  values={label.tolist()}")
    print(f"\n✅ dataset.py working correctly")