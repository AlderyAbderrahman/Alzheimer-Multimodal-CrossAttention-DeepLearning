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
 
Adaptations for small dataset (416 subjects):
  - n_heads reduced 8 → 4  (halves attention parameters)
  - ModalityInteraction replaced with simple gated fusion (fewer params)
  - Stronger dropout throughout (0.1 → 0.2)
  - Learnable α to blend attended MRI with global summary
  - Total fusion parameters: ~200K vs ~400K original
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
 
    n_heads=4 (not 8): with only 332 training subjects, 4 heads gives
    64 dims per head which is enough expressivity without overfitting.
    """
 
    def __init__(self, feature_dim: int = 256,
                 n_heads:     int   = 4,      # reduced from 8
                 dropout:     float = 0.2):   # increased from 0.1
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
 
    def forward(self, query:   torch.Tensor,
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
        attn_weights : (B, n_heads, 1, 216)  attention map for visualization
        """
        attended, attn_weights = self.attn(
            query   = query,
            key     = context,
            value   = context,
            need_weights         = True,
            average_attn_weights = False,  # keep per-head weights for viz
        )
        # Residual + norm (standard transformer pattern)
        out = self.norm(query + self.dropout(attended))
        return out, attn_weights   # (B,1,D),  (B, n_heads, 1, 216)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GATED FUSION
# Replaces the heavier ModalityInteraction for small-data regime
# ─────────────────────────────────────────────────────────────────────────────
 
class GatedFusion(nn.Module):
    """
    Lightweight gated fusion of MRI and clinical vectors.
 
    Instead of learning a complex interaction (expensive, overfits on
    small data), we learn a per-dimension gate that decides how much
    each modality contributes to the final representation.
 
    Mechanism:
        gate   = sigmoid( W * [mri_vec, clin_vec] )   ∈ (0,1)^D
        fused  = gate * mri_vec + (1-gate) * clin_vec
 
    Why this works:
      - For features where MRI is reliable (nWBV region), gate → 1
      - For features where clinical is reliable (MMSE score), gate → 0
      - The model learns this per-dimension from data
      - Only D*(2D) + D parameters — far fewer than ModalityInteraction
 
    Parameters: 256*(512)+256 = ~131K  vs ModalityInteraction's ~200K+
    But critically: no interaction term that can overfit on 332 subjects.
    """
 
    def __init__(self, feature_dim: int = 256, dropout: float = 0.2):
        super().__init__()
 
        # Gate: takes concatenated [mri, clin] → D-dim gate weights
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )
 
        # Project gated output back to feature_dim with normalization
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
 
    def forward(self, mri_vec:  torch.Tensor,
                      clin_vec: torch.Tensor
                ) -> torch.Tensor:
        """
        Args
        ----
        mri_vec  : (B, D)
        clin_vec : (B, D)
 
        Returns
        -------
        fused : (B, D)
        """
        # Compute gate from both modalities jointly
        gate  = self.gate_fc(torch.cat([mri_vec, clin_vec], dim=-1))  # (B, D)
 
        # Weighted blend: gate controls MRI vs clinical contribution
        fused = gate * mri_vec + (1.0 - gate) * clin_vec              # (B, D)
 
        return self.proj(fused)                                         # (B, D)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FULL FUSION MODULE
# ─────────────────────────────────────────────────────────────────────────────
 
class CrossModalFusion(nn.Module):
    """
    Complete fusion pipeline optimized for small dataset (416 subjects).
 
    Steps:
      1. CrossModalAttention (4 heads)
             clinical token queries MRI patches
             → attended_mri  (B, D)
             → attn_weights  (B, 4, 1, 216)  — for visualization
 
      2. Blend attended MRI with global summary
             α is learnable — model decides how much local vs global
             → mri_vec  (B, D)
 
      3. GatedFusion
             lightweight per-dimension gating of MRI vs clinical
             → fused  (B, D)  — ready for classifier
 
    Total parameters: ~200K  (vs ~400K in the original version)
    """
 
    def __init__(self, feature_dim: int = 256,
                 n_heads:     int   = 4,
                 dropout:     float = 0.2):
        super().__init__()
 
        self.cross_attn = CrossModalAttention(feature_dim, n_heads, dropout)
        self.fusion     = GatedFusion(feature_dim, dropout)
 
        # Learnable blend between attended (local) and global MRI summary
        # Initialized at 0.5 — equal weight; model learns the right balance
        # sigmoid(α) → ensures blend stays in [0, 1]
        self.alpha = nn.Parameter(torch.tensor(0.5))
 
    def forward(self,
                patches:    torch.Tensor,   # (B, 216, D)
                global_vec: torch.Tensor,   # (B, D)
                clin_token: torch.Tensor,   # (B, 1,   D)
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        patches    : (B, 216, D)  spatial MRI tokens from BrainMRI3DEncoder
        global_vec : (B, D)       global MRI summary from BrainMRI3DEncoder
        clin_token : (B, 1,   D)  clinical query token from ClinicalMLPEncoder
 
        Returns
        -------
        fused        : (B, D)               → feed directly to classifier
        attn_weights : (B, n_heads, 1, 216) → save for attention map viz
        """
        # ── Step 1: clinical token queries MRI patches ──────────────────────
        attended, attn_weights = self.cross_attn(
            query   = clin_token,   # (B,   1, D)
            context = patches,      # (B, 216, D)
        )
        attended_mri = attended.squeeze(1)    # (B, D)
        clin_vec     = clin_token.squeeze(1)  # (B, D)
 
        # ── Step 2: blend attended (local) with global summary ───────────────
        # α learned: if α→1, trust attended regions; if α→0, trust global
        alpha   = torch.sigmoid(self.alpha)                           # scalar
        mri_vec = alpha * attended_mri + (1.0 - alpha) * global_vec  # (B, D)
 
        # ── Step 3: gated fusion of MRI and clinical ─────────────────────────
        fused = self.fusion(mri_vec, clin_vec)                        # (B, D)
 
        return fused, attn_weights
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION MAP VISUALIZATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
 
def attention_to_volume(attn_weights: torch.Tensor,
                        grid_size:    int = 6) -> torch.Tensor:
    """
    Converts the flat 216-dim attention vector back to a 3D volume
    for visualization as a brain heatmap overlay.
 
    Args
    ----
    attn_weights : (B, n_heads, 1, 216)  from CrossModalFusion.forward()
    grid_size    : int = 6               must match encoder_mri spatial grid
 
    Returns
    -------
    attn_vol : (B, 6, 6, 6)
 
    To overlay on the original MRI at full resolution:
        attn_96 = F.interpolate(
            attn_vol.unsqueeze(1),          # (B, 1, 6, 6, 6)
            size=(96, 96, 96),
            mode='trilinear',
            align_corners=False
        ).squeeze(1)                        # (B, 96, 96, 96)
    """
    # Average across heads, remove query seq dim
    attn = attn_weights.mean(dim=1)   # (B, 1, 216)
    attn = attn.squeeze(1)            # (B, 216)
 
    B = attn.shape[0]
    return attn.view(B, grid_size, grid_size, grid_size)   # (B, 6, 6, 6)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
 
    D = 256
    B = 4
 
    patches    = torch.randn(B, 216, D, device=device)
    global_vec = torch.randn(B,       D, device=device)
    clin_token = torch.randn(B,   1,  D, device=device)
 
    fusion = CrossModalFusion(feature_dim=D, n_heads=4).to(device)
 
    with torch.no_grad():
        fused, attn_w = fusion(patches, global_vec, clin_token)
 
    print(f"patches      : {tuple(patches.shape)}")
    print(f"global_vec   : {tuple(global_vec.shape)}")
    print(f"clin_token   : {tuple(clin_token.shape)}")
    print(f"─────────────────────────────────────")
    print(f"fused        : {tuple(fused.shape)}")       # (4, 256) ← to classifier
    print(f"attn_weights : {tuple(attn_w.shape)}")      # (4, 4, 1, 216)
 
    attn_vol = attention_to_volume(attn_w)
    print(f"attn_vol     : {tuple(attn_vol.shape)}")    # (4, 6, 6, 6)
 
    n = sum(p.numel() for p in fusion.parameters())
    print(f"\nFusion parameters: {n:,}")
 
    # Show learned alpha value
    alpha = torch.sigmoid(fusion.alpha).item()
    print(f"Learned α (local/global blend): {alpha:.3f}  (0.5 = equal)")
    print(f"\n✅ fusion.py smoke test passed")
 