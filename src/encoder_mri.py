import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.pool  = nn.MaxPool3d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.pool(x)


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(channels)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(
                      F.relu(self.bn1(self.conv1(x)), inplace=True)
                      )) + x, inplace=True)


class BrainMRI3DEncoder(nn.Module):
    """
    Returns BOTH the spatial patch sequence (for cross-attention)
    and the global vector (for simple fusion / ablation baseline).

    Spatial output:  (B, N_patches, feature_dim)  where N_patches = 6*6*6 = 216
    Global output:   (B, feature_dim)
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.stage1 = ConvBlock3D(1,    32,  pool=True)
        self.stage2 = ConvBlock3D(32,   64,  pool=True)
        self.stage3 = ConvBlock3D(64,   128, pool=True)
        self.res    = ResBlock3D(128)
        self.stage4 = ConvBlock3D(128,  256, pool=True)
        # After stage4: (B, 256, 6, 6, 6)

        # ── Projection: 256 → feature_dim (applied per spatial location) ──
        # Conv1x1x1 is equivalent to a Linear applied to each of the 216 patches
        self.patch_proj = nn.Conv3d(256, feature_dim,
                                    kernel_size=1, bias=False)
        self.patch_norm = nn.BatchNorm3d(feature_dim)

        # ── Positional encoding (learnable, one per spatial location) ──
        # Shape: (1, feature_dim, 6, 6, 6) — broadcast over batch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feature_dim, 6, 6, 6)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Global branch ──
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.global_proj = nn.Linear(feature_dim, feature_dim)
        self.global_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, 96, 96, 96)

        Returns:
            patches: (B, 216, feature_dim)  ← for cross-attention
            global_vec: (B, feature_dim)    ← for simple fusion / CLS token
        """
        # ── Shared CNN backbone ──
        x = self.stage1(x)      # (B,  32, 48, 48, 48)
        x = self.stage2(x)      # (B,  64, 24, 24, 24)
        x = self.stage3(x)      # (B, 128, 12, 12, 12)
        x = self.res(x)         # (B, 128, 12, 12, 12)
        x = self.stage4(x)      # (B, 256,  6,  6,  6)

        # ── Project each spatial location to feature_dim ──
        x = F.relu(self.patch_norm(self.patch_proj(x)), inplace=True)
        # x: (B, feature_dim, 6, 6, 6)

        # ── Add positional encoding ──
        x = x + self.pos_embed
        # Still: (B, feature_dim, 6, 6, 6)

        # ── Flatten spatial dims → token sequence ──
        # (B, feature_dim, 6, 6, 6)
        #   → (B, feature_dim, 216)   [merge D*H*W]
        #   → (B, 216, feature_dim)   [swap: tokens first, features second]
        B, C, D, H, W = x.shape
        patches = x.view(B, C, D * H * W)   # (B, feature_dim, 216)
        patches = patches.permute(0, 2, 1)  # (B, 216, feature_dim)
        # Each of the 216 rows = one spatial brain region's feature vector

        # ── Global vector (for ablation / simple fusion) ──
        global_vec = self.gap(x)            # (B, feature_dim, 1, 1, 1)
        global_vec = self.flatten(global_vec)        # (B, feature_dim)
        global_vec = self.dropout(global_vec)
        global_vec = self.global_norm(self.global_proj(global_vec))
        # (B, feature_dim)

        return patches, global_vec


# ── HOW CROSS-ATTENTION WILL USE THESE OUTPUTS ───────────────────────────────
#
#  clinical_token: (B, 1, feature_dim)      ← Query  (1 token)
#  patches:        (B, 216, feature_dim)    ← Keys and Values
#
#  attention_output = CrossAttention(
#      query = clinical_token,   # "what am I looking for?"
#      key   = patches,          # "what is in each brain region?"
#      value = patches           # "what information do I extract?"
#  )
#  # output: (B, 1, feature_dim) → squeeze → (B, feature_dim)
#  # This vector encodes: "the brain regions most relevant to this patient's
#  #                        clinical profile"


# ── SHAPE VERIFICATION ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = BrainMRI3DEncoder(feature_dim=256).to(device)
    dummy   = torch.randn(2, 1, 96, 96, 96, device=device)

    with torch.no_grad():
        patches, global_vec = encoder(dummy)

    print(f"Input shape      : {tuple(dummy.shape)}")
    print(f"Patches shape    : {tuple(patches.shape)}")
    #  → (2, 216, 256)   B=2, 216 spatial tokens, 256 features each
    print(f"Global vec shape : {tuple(global_vec.shape)}")
    #  → (2, 256)

    # Confirm the 216 comes from 6*6*6
    print(f"\n6×6×6 = {6*6*6} spatial locations")
    print(f"Each encodes a ~16mm³ region of the brain (96/6 × 2mm voxels)")

    # Memory check
    param_mb = sum(p.numel() * 4 for p in encoder.parameters()) / 1e6
    print(f"\nEncoder weights : {param_mb:.1f} MB")