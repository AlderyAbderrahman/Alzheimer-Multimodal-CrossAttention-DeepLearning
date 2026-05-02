from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/dataset/oasis13d')  # ← fixed

import torch
import torch.nn as nn
from src.preprocessing import N_FEATURES, FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-WISE ATTENTION GATE
# ─────────────────────────────────────────────────────────────────────────────

class FeatureWiseAttention(nn.Module):
    """
    Learns which of the 7 clinical features matter most for the prediction.

    Mechanism: output = x * sigmoid(W*x)
      - When sigmoid output is near 1 → feature passes through fully
      - When sigmoid output is near 0 → feature is suppressed

    Why this matters for AD:
      MMSE and nWBV should be heavily weighted; M/F and eTIV much less so.
      Instead of hard-coding this, we let the model learn it from data.

    Parameters added: only 7*7 + 7 = 56 — essentially free.
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.gate = nn.Linear(n_features, n_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x      : (B, 7)
        # weights: (B, 7)  each in [0, 1]
        weights = torch.sigmoid(self.gate(x))
        return x * weights                      # (B, 7)  element-wise gate

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns gate weights without applying them.
        Use at inference to see which features the model relied on.
        Shape: (B, 7)
        """
        with torch.no_grad():
            return torch.sigmoid(self.gate(x))  # (B, 7)


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL MLP ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalMLPEncoder(nn.Module):
    """
    Encodes 7 standardized clinical features into a single query token
    for cross-attention with the MRI spatial patches.

    Input  : (B, 7)              standardized clinical vector
    Output : (B, 1, feature_dim) query token — the '1' makes it a
                                 sequence of length 1, matching the
                                 transformer attention API

    Architecture
    ------------
    FeatureWiseAttention(7)          learns per-feature importance
    Linear(7  → 64)  + LN + GELU + Dropout
    Linear(64 → 128) + LN + GELU + Dropout
    Linear(128 → D)  + LN           D must match MRI encoder feature_dim
    unsqueeze(1)     → (B, 1, D)

    Design choices
    --------------
    GELU not ReLU   : smoother gradient for small networks; avoids dead neurons
    LayerNorm not BN: output feeds into transformer attention which expects LN
    3 layers        : enough capacity for 7→256 without overfitting on 436 subjects
    Dropout 0.2     : light regularisation; clinical data is clean, not noisy images
    """

    def __init__(self,
                 in_features: int = N_FEATURES,   # 7
                 feature_dim: int = 256,
                 dropout:     float = 0.2):
        super().__init__()

        self.feature_attention = FeatureWiseAttention(in_features)

        self.mlp = nn.Sequential(

            # Layer 1 — initial expansion  7 → 64
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2 — intermediate       64 → 128
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 3 — project to D       128 → feature_dim
            # Must match BrainMRI3DEncoder's feature_dim
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 7)  batch of standardized clinical feature vectors

        Returns
        -------
        token : (B, 1, feature_dim)
            Used as the Query in cross-attention:
              Q = clinical token  (B,   1, D)
              K = MRI patches     (B, 216, D)
              V = MRI patches     (B, 216, D)
        """
        x = self.feature_attention(x)   # (B, 7)  gated
        x = self.mlp(x)                 # (B, feature_dim)
        return x.unsqueeze(1)           # (B, 1, feature_dim)

    def get_feature_importance(self, x: torch.Tensor) -> dict:
        """
        Returns a dict {feature_name: mean_gate_weight} for a batch.
        Use in your thesis to show the model learned MMSE > SES etc.

        Example output:
            {'Age': 0.61, 'M/F': 0.41, 'MMSE': 0.97, 'nWBV': 0.89, ...}
        """
        weights = self.feature_attention.get_weights(x)  # (B, 7)
        mean_w  = weights.mean(dim=0)                     # (7,)
        return {feat: round(mean_w[i].item(), 3)
                for i, feat in enumerate(FEATURE_COLS)}


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ClinicalMLPEncoder(in_features=7, feature_dim=256).to(device)

    # Simulate a batch of 4 patients
    dummy = torch.randn(4, 7, device=device)
    token = encoder(dummy)

    print(f"Input  : {tuple(dummy.shape)}")   # (4, 7)
    print(f"Output : {tuple(token.shape)}")   # (4, 1, 256)

    # Show learned feature importance (random weights before training)
    importance = encoder.get_feature_importance(dummy)
    print("\nFeature gate weights (random — will be meaningful after training):")
    for feat, w in sorted(importance.items(), key=lambda x: -x[1]):
        bar = '█' * int(w * 20)
        print(f"  {feat:<8} {bar:<20} {w:.3f}")

    # Confirm alignment with MRI encoder output
    mri_patches = torch.randn(4, 216, 256, device=device)
    print(f"\nAlignment check:")
    print(f"  MRI patches    : {tuple(mri_patches.shape)}")
    print(f"  Clinical token : {tuple(token.shape)}")
    print(f"  → Q={tuple(token.shape)}  K=V={tuple(mri_patches.shape)}")
    print(f"  → Ready for cross-attention fusion ✓")

    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nClinical encoder parameters: {n_params:,}")
    # Expected: ~35,000 — tiny compared to MRI encoder's ~3.5M