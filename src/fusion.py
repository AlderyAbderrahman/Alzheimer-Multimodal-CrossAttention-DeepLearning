"""
src/fusion.py
-------------
Cross-modal attention fusion module.

Takes:
  patches     (B, 216, D)  — spatial MRI tokens from BrainMRI3DEncoder
  global_vec  (B, D)       — global MRI summary from BrainMRI3DEncoder
  clin_token  (B, 1,  D)   — clinical query token from ClinicalMLPEncoder

Returns:
  fused       (B, D)       — single vector for the classifier head

Novelty vs plain concatenation (most prior work):
  The clinical token acts as a QUERY over the 216 spatial MRI patches.
  The model learns WHICH brain regions are most relevant given the
  patient's clinical profile — not just that both modalities exist.
  The attention map is also interpretable (which regions lit up for
  a given CDR score) — a key thesis result figure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ATTENTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    One cross-attention block.

    Query  = clinical token   (B,   1, D)
    Key    = MRI patches      (B, 216, D)
    Value  = MRI patches      (B, 216, D)

    The clinical token asks: "given this patient's age/MMSE/nWBV,
    which of the 216 brain regions should I pay attention to?"

    Output: (B, 1, D) — attended MRI context, shaped by clinical profile
    """

    def __init__(self, feature_dim: int = 256,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        assert feature_dim % n_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})"

        self.attn = nn.MultiheadAttention(
            embed_dim   = feature_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,   # expects (B, seq, dim) — matches our shapes
        )
        self.norm    = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor,
                      context: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        query   : (B,   1, D)  clinical token
        context : (B, 216, D)  MRI spatial patches

        Returns
        -------
        out          : (B, 1, D)    attended output
        attn_weights : (B, 1, 216)  attention map — save for visualization
        """
        attended, attn_weights = self.attn(
            query   = query,      # Q: clinical
            key     = context,    # K: MRI patches
            value   = context,    # V: MRI patches
            need_weights = True,
            average_attn_weights = False,  # keep per-head weights
        )
        # Residual + norm (standard transformer pattern)
        out = self.norm(query + self.dropout(attended))
        return out, attn_weights  # (B,1,D), (B, n_heads, 1, 216)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION LAYER  (from Alorf 2025, Eq. 16-17 — adapted for structural MRI)
# ─────────────────────────────────────────────────────────────────────────────

class ModalityInteraction(nn.Module):
    """
    Computes a non-linear interaction between MRI and clinical features
    BEFORE final concatenation.

    From Alorf (2025) Eq. 16:
        X_interacted = ReLU( (Ox @ W1) * (Ot @ W2) )

    This element-wise product forces the two modalities to jointly
    activate — a feature only matters if BOTH the MRI AND clinical
    streams agree on it. This is stronger than plain concatenation.

    Eq. 17 weighted fusion:
        O_concat = λ*(Ox ⊕ Ot) + (1-λ)*X_interacted
    where λ is a learnable scalar.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()

        self.W1 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W2 = nn.Linear(feature_dim, feature_dim, bias=False)

        # λ: learnable scalar in [0,1] controlling concat vs interaction
        # Initialized at 0.5 — equal weight to start
        self.lam = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(feature_dim * 2)

    def forward(self, mri_vec: torch.Tensor,
                      clin_vec: torch.Tensor
                ) -> torch.Tensor:
        """
        Args
        ----
        mri_vec  : (B, D)  — attended MRI vector
        clin_vec : (B, D)  — clinical vector

        Returns
        -------
        fused : (B, 2D)   — interaction-aware fused representation
        """
        # Eq. 16 — element-wise interaction
        x_interact = F.relu(self.W1(mri_vec) * self.W2(clin_vec))  # (B, D)

        # Eq. 17 — weighted combination of direct concat + interaction
        lam       = torch.sigmoid(self.lam)                          # scalar in [0,1]
        direct    = torch.cat([mri_vec, clin_vec], dim=-1)           # (B, 2D)
        interact2 = torch.cat([x_interact, x_interact], dim=-1)      # (B, 2D)

        fused = lam * direct + (1 - lam) * interact2                 # (B, 2D)
        return self.norm(fused)                                       # (B, 2D)


# ─────────────────────────────────────────────────────────────────────────────
# FULL FUSION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Complete fusion pipeline:

      1. CrossModalAttention
           clinical token queries MRI patches
           → attended_mri  (B, D)
           → attn_weights  (B, n_heads, 1, 216) — for visualization

      2. ModalityInteraction
           interaction between attended_mri and clinical vector
           → fused  (B, 2D)

      3. Projection
           (B, 2D) → (B, D)  — back to single feature_dim

    The output (B, D) goes directly into the classifier head.
    """

    def __init__(self, feature_dim: int = 256,
                 n_heads:     int   = 8,
                 dropout:     float = 0.1):
        super().__init__()

        self.cross_attn  = CrossModalAttention(feature_dim, n_heads, dropout)
        self.interaction = ModalityInteraction(feature_dim)

        # Project 2D → D after interaction concat
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self,
                patches:    torch.Tensor,    # (B, 216, D)  MRI spatial tokens
                global_vec: torch.Tensor,    # (B, D)       MRI global summary
                clin_token: torch.Tensor,    # (B, 1,   D)  clinical query token
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        fused        : (B, D)               → feed to classifier
        attn_weights : (B, n_heads, 1, 216) → save for attention map viz
        """
        # ── Step 1: clinical token queries MRI patches ──────────────────────
        attended, attn_weights = self.cross_attn(
            query   = clin_token,   # (B,   1, D)
            context = patches,      # (B, 216, D)
        )
        attended_mri = attended.squeeze(1)      # (B, D)  remove seq dim
        clin_vec     = clin_token.squeeze(1)    # (B, D)  remove seq dim

        # ── Step 2: optionally blend attended MRI with global summary ───────
        # global_vec captures the whole brain; attended focuses on specific
        # regions. Averaging them keeps both signals.
        mri_vec = (attended_mri + global_vec) * 0.5   # (B, D)

        # ── Step 3: modality interaction ────────────────────────────────────
        fused_2d = self.interaction(mri_vec, clin_vec)  # (B, 2D)

        # ── Step 4: project back to D ────────────────────────────────────────
        fused = self.proj(fused_2d)                     # (B, D)

        return fused, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION MAP VISUALIZATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def attention_to_volume(attn_weights: torch.Tensor,
                         grid_size: int = 6) -> torch.Tensor:
    """
    Converts the flat 216-dim attention vector back to a 3D volume
    for visualization as a brain heatmap.

    Args
    ----
    attn_weights : (B, n_heads, 1, 216)  from CrossModalFusion.forward()
    grid_size    : int = 6               must match encoder's spatial grid

    Returns
    -------
    attn_vol : (B, 6, 6, 6)  — attention map in MNI space
               upscale to (96,96,96) with F.interpolate for overlay on MRI
    """
    # Average across heads, remove seq dim
    attn = attn_weights.mean(dim=1)          # (B, 1, 216)
    attn = attn.squeeze(1)                   # (B, 216)

    B = attn.shape[0]
    attn_vol = attn.view(B, grid_size, grid_size, grid_size)  # (B,6,6,6)
    return attn_vol


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    D = 256
    B = 4   # batch size

    # Simulate encoder outputs
    patches    = torch.randn(B, 216, D, device=device)
    global_vec = torch.randn(B,       D, device=device)
    clin_token = torch.randn(B,   1,  D, device=device)

    fusion = CrossModalFusion(feature_dim=D, n_heads=8).to(device)

    with torch.no_grad():
        fused, attn_w = fusion(patches, global_vec, clin_token)

    print(f"patches    : {tuple(patches.shape)}")
    print(f"global_vec : {tuple(global_vec.shape)}")
    print(f"clin_token : {tuple(clin_token.shape)}")
    print(f"─────────────────────────────────")
    print(f"fused      : {tuple(fused.shape)}")       # (4, 256) ← to classifier
    print(f"attn_weights:{tuple(attn_w.shape)}")      # (4, 8, 1, 216)

    # Reshape attention to 3D brain volume
    attn_vol = attention_to_volume(attn_w)
    print(f"attn_vol   : {tuple(attn_vol.shape)}")    # (4, 6, 6, 6)

    # Parameter count
    n = sum(p.numel() for p in fusion.parameters())
    print(f"\nFusion parameters: {n:,}")
    # Expected: ~400,000
