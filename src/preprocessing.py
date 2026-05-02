from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/dataset/oasis13d/data')
"""
src/preprocessing.py
--------------------
Single responsibility: everything related to the OASIS-1 CSV.
  - Constants shared across the whole project
  - ClinicalPreprocessor  (sklearn imputer + scaler)
  - compute_class_weights (for imbalanced CDR labels)
  - describe_features     (exploration helper)
 
Nothing PyTorch here — no nn.Module, no tensors.
Import these constants in encoder_clinical.py to avoid duplication.
"""
 
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS  (import these everywhere — never redefine them)
# ─────────────────────────────────────────────────────────────────────────────
 
FEATURE_COLS = ['Age', 'M/F', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV']
# Dropped columns and reasons:
#   ID       → identifier, not a feature
#   CDR      → this IS the label, never an input
#   Hand     → near-zero variance (~all right-handed in OASIS-1)
#   ASF      → mathematically derived from eTIV, collinear
#   Delay    → NaN for 95% of subjects (only MR2 repeat scans)
 
LABEL_COL    = 'CDR'
ID_COL       = 'ID'
N_FEATURES   = len(FEATURE_COLS)   # 7
N_CLASSES    = 3
 
# CDR score → integer class index
# CDR 1.0 and 2.0 are merged into class 2 ("Mild/Moderate / confirmed dementia").
# Rationale:
#   - CDR 2.0 has only ~2 subjects in OASIS-1 — not enough for a stable class.
#   - Clinically defensible: CDR ≥ 1.0 marks confirmed dementia vs CDR 0.5
#     which is "questionable". The 3-class split maps cleanly to:
#       0 = No dementia  |  1 = Questionable  |  2 = Confirmed dementia
#   - Merging reduces the max/min class-weight ratio from ~135× to ~12×,
#     yielding a far more stable CrossEntropyLoss gradient signal.
CDR_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}
CLASS_NAMES  = ['Non-demented', 'Very mild', 'Mild/Moderate']
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────
 
def load_oasis_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the OASIS-1 CSV (.csv or .xlsx) and print a quick sanity check.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
 
    df = pd.read_excel(csv_path) if csv_path.endswith('.xlsx') \
         else pd.read_csv(csv_path)
 
    print(f"Loaded  : {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns : {list(df.columns)}\n")
 
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("Missing values per column:")
        print(missing.to_string())
    else:
        print("No missing values found.")
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
 
class ClinicalPreprocessor:
    """
    Fits on training data only, then transforms val/test consistently.
 
    Usage
    -----
    prep = ClinicalPreprocessor()
    X_train, y_train = prep.fit_transform(df_train)   # fit HERE only
    X_val,   y_val   = prep.transform(df_val)          # reuse fit
    X_test,  y_test  = prep.transform(df_test)
    prep.save('checkpoints/preprocessor/')
 
    # Later, in inference / evaluation:
    prep = ClinicalPreprocessor.load('checkpoints/preprocessor/')
    X_new, y_new = prep.transform(df_new)
    """
 
    def __init__(self):
        # Median imputation: robust to outliers, correct for ordinal scales
        self.imputer = SimpleImputer(strategy='median')
        # StandardScaler: zero mean, unit variance
        # Needed because Age (~70) and eTIV (~1,500,000) are wildly different scales
        self.scaler  = StandardScaler()
        self._fitted = False
 
    # ── internal helpers ──────────────────────────────────────────────────────
 
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select valid rows, encode sex."""
        df = df.copy()
        df = df.dropna(subset=[LABEL_COL])
        df = df[df[LABEL_COL].isin(CDR_TO_CLASS)]
        # Encode sex: Female=0, Male=1
        df['M/F'] = (df['M/F'].astype(str)
                               .str.strip().str.upper()
                               .map({'F': 0.0, 'M': 1.0}))
        return df
 
    def _to_labels(self, df: pd.DataFrame) -> np.ndarray:
        return df[LABEL_COL].map(CDR_TO_CLASS).values.astype(np.int64)
 
    # ── public API ────────────────────────────────────────────────────────────
 
    def fit_transform(self, df: pd.DataFrame
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Fit imputer+scaler on training data. Returns (X float32, y int64)."""
        df    = self._clean(df)
        y     = self._to_labels(df)
        X_raw = df[FEATURE_COLS].values.astype(np.float64)
 
        X = self.imputer.fit_transform(X_raw)
        X = self.scaler.fit_transform(X).astype(np.float32)
        self._fitted = True
 
        self._report(X, y)
        return X, y
 
    def transform(self, df: pd.DataFrame
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Transform val or test split using training statistics. Never refits."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        df    = self._clean(df)
        y     = self._to_labels(df)
        X_raw = df[FEATURE_COLS].values.astype(np.float64)
 
        X = self.imputer.transform(X_raw)
        X = self.scaler.transform(X).astype(np.float32)
        return X, y
 
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.imputer, f"{directory}/imputer.pkl")
        joblib.dump(self.scaler,  f"{directory}/scaler.pkl")
        print(f"Preprocessor saved → {directory}/")
 
    @classmethod
    def load(cls, directory: str) -> 'ClinicalPreprocessor':
        prep = cls()
        prep.imputer = joblib.load(f"{directory}/imputer.pkl")
        prep.scaler  = joblib.load(f"{directory}/scaler.pkl")
        prep._fitted = True
        return prep
 
    def _report(self, X: np.ndarray, y: np.ndarray):
        print(f"Feature matrix : {X.shape}  (N_samples × {N_FEATURES} features)")
        print(f"Features       : {FEATURE_COLS}")
        print(f"Value range    : [{X.min():.2f}, {X.max():.2f}]  (after StandardScaler)")
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for cls_id, cnt in zip(unique, counts):
            bar = '█' * cnt
            print(f"  class {cls_id} "
                  f"({CLASS_NAMES[cls_id]}): {bar[:60]} {cnt}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS  (used in CrossEntropyLoss to handle CDR imbalance)
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_class_weights(y_train: np.ndarray,
                           n_classes: int = N_CLASSES) -> np.ndarray:
    """
    Returns inverse-frequency weights as a float32 numpy array.
    Pass to nn.CrossEntropyLoss(weight=torch.tensor(w).to(device)).
 
    Example
    -------
    w = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    """
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = (weights / weights.sum() * n_classes).astype(np.float32)
 
    print("Class weights for CrossEntropyLoss:")
    for i, (w, c) in enumerate(zip(weights, counts)):
        print(f"  class {i} ({CLASS_NAMES[i]:<18}): "
              f"weight={w:.3f}  (n={int(c)})")
    return weights
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EXPLORATION HELPER  (run in notebook 01_data_exploration.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
 
def describe_features(df: pd.DataFrame):
    """
    Print feature statistics and CDR correlation.
    Run once in your exploration notebook — output goes straight
    into your thesis as Table 1 / Figure 1.
    """
    df = df.copy()
    df = df.dropna(subset=[LABEL_COL])
    df = df[df[LABEL_COL].isin(CDR_TO_CLASS)]
    df['M/F'] = df['M/F'].astype(str).str.upper().map({'F': 0.0, 'M': 1.0})
 
    print("=== Descriptive statistics ===")
    print(df[FEATURE_COLS + [LABEL_COL]].describe().round(3).to_string())
 
    print("\n=== Pearson correlation with CDR label ===")
    corr = (df[FEATURE_COLS + [LABEL_COL]]
            .corr()['CDR']
            .drop('CDR')
            .sort_values(key=abs, ascending=False))
    for feat, val in corr.items():
        bar  = '█' * int(abs(val) * 20)
        sign = '+' if val > 0 else '-'
        print(f"  {feat:<8} {sign}{bar:<20} {val:+.3f}")
 
    print("\n=== Class balance (3-class, CDR 1+2 merged) ===")
    # Map raw CDR values to merged class labels for display
    df['_cls'] = df[LABEL_COL].map(CDR_TO_CLASS)
    for cls_id, name in enumerate(CLASS_NAMES):
        cnt = (df['_cls'] == cls_id).sum()
        pct = cnt / len(df) * 100
        bar = '█' * cnt
        print(f"  class {cls_id} {name:<18}: {bar[:50]} {cnt:3d} ({pct:.1f}%)")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split
 
    from google.colab import drive

    drive.mount('/content/drive')
    BASE_PATH = "/content/drive/MyDrive/dataset/oasis13d/data"
    csv_path = f"{BASE_PATH}/oasis_cross-sectional.csv"
    df = pd.read_csv(csv_path)
    describe_features(df)
 
    # Subject-level stratified split
    # Stratify ensures all 3 classes appear in both train and val —
    # critical for class 2 (Mild/Moderate) which has only ~23 subjects.
    df['_base'] = df[ID_COL].str.extract(r'(OAS1_\d+)')
    # Derive a per-subject label for stratification (take first scan's CDR)
    subject_label = (df.groupby('_base')[LABEL_COL]
                       .first()
                       .map(CDR_TO_CLASS))
    subjects = subject_label.index.values
    labels   = subject_label.values
 
    tr, va = train_test_split(
        subjects,
        test_size=0.2,
        random_state=42,
        stratify=labels          # ← keeps class 2 in both splits
    )
 
    prep = ClinicalPreprocessor()
    X_tr, y_tr = prep.fit_transform(df[df['_base'].isin(tr)])
    X_va, y_va = prep.transform(df[df['_base'].isin(va)])
    print(f"\nTrain X: {X_tr.shape}  y: {y_tr.shape}")
    print(f"Val   X: {X_va.shape}  y: {y_va.shape}")
 
    w = compute_class_weights(y_tr)
    print(f"\nWeights array: {w}")