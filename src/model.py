"""
src/model.py
------------
AlzheimerFusionModel — complete model, adapted for small dataset (332 train subjects).
 
Key small-dataset adaptations:
  - n_heads = 4 (not 8) in fusion
  - Staged training support: freeze MRI encoder → train fusion → fine-tune all
  - Stronger dropout in classifier head (0.5)
  - Xavier weight initialization for classifier output layer
 
Ablation modes for thesis table:
  'full'          → complete cross-attention fusion  (proposed architecture)
  'mri_only'      → MRI global_vec → classifier
  'clinical_only' → clinical token → classifier
  'concat_only'   → naive concat → linear → classifier  (no attention)
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from src.encoder_mri      import BrainMRI3DEncoder
from src.encoder_clinical import ClinicalMLPEncoder
from src.fusion           import CrossModalFusion
from src.preprocessing    import N_FEATURES, N_CLASSES
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER HEAD
# ─────────────────────────────────────────────────────────────────────────────
 
class ClassifierHead(nn.Module):
    """
    Maps fused representation → class logits.
 
    Input:  (B, in_dim)    → (B, n_classes)
 
    No softmax — nn.CrossEntropyLoss expects raw logits.
    Adding softmax here would cause numerical instability during training.
    Only use softmax at inference when you need probabilities.
 
    Dropout is 0.5 (higher than encoders) because this is the
    most overfit-prone layer with small data — it sits right before
    the output and can memorize training labels easily.
    """
 
    def __init__(self, in_dim:    int,
                       n_classes: int   = N_CLASSES,
                       dropout:   float = 0.5):
        super().__init__()
 
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),   # raw logits — no activation
        )
 
        # Initialize final layer with small weights.
        # Prevents overconfident predictions at the start of training,
        # which would produce large losses and unstable gradients.
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # (B, n_classes)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
 
class AlzheimerFusionModel(nn.Module):
    """
    Complete multimodal Alzheimer classification model.
 
    Inputs:
        mri      : (B, 1, 96, 96, 96)  — normalized T1w MRI volume
        clinical : (B, 7)              — standardized clinical features
 
    Outputs:
        logits       : (B, n_classes)         — for CrossEntropyLoss
        attn_weights : (B, 4, 1, 216) or None — attention map for visualization
 
    Ablation modes:
        'full'          → complete cross-attention fusion  (your proposed model)
        'mri_only'      → MRI global_vec → classifier only
        'clinical_only' → clinical token → classifier only
        'concat_only'   → concat(global_vec, clin_vec) → linear → classifier
    """
 
    VALID_MODES = ('full', 'mri_only', 'clinical_only', 'concat_only')
 
    def __init__(self,
                 feature_dim:  int   = 256,
                 n_heads:      int   = 4,       # 4 for small dataset (not 8)
                 n_classes:    int   = N_CLASSES,
                 dropout_enc:  float = 0.3,
                 dropout_head: float = 0.5,     # higher dropout for classifier
                 mode:         str   = 'full'):
 
        super().__init__()
        assert mode in self.VALID_MODES, \
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"
 
        self.mode        = mode
        self.feature_dim = feature_dim
 
        # ── Encoders ──────────────────────────────────────────────────────────
        # Both always built — reused across all ablation modes
        self.mri_encoder = BrainMRI3DEncoder(
            feature_dim = feature_dim,
            dropout     = dropout_enc,
        )
        self.clin_encoder = ClinicalMLPEncoder(
            in_features = N_FEATURES,
            feature_dim = feature_dim,
            dropout     = dropout_enc,
        )
 
        # ── Fusion (only in 'full' mode) ──────────────────────────────────────
        if mode == 'full':
            self.fusion = CrossModalFusion(
                feature_dim = feature_dim,
                n_heads     = n_heads,
                dropout     = 0.2,
            )
 
        # ── Concat projection (only in 'concat_only' ablation) ───────────────
        if mode == 'concat_only':
            # Concat gives 2×feature_dim → project back to feature_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout_enc),
            )
 
        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = ClassifierHead(
            in_dim    = feature_dim,
            n_classes = n_classes,
            dropout   = dropout_head,
        )
 
    # ── Forward ───────────────────────────────────────────────────────────────
 
    def forward(self, mri:      torch.Tensor,
                      clinical: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args
        ----
        mri      : (B, 1, 96, 96, 96)
        clinical : (B, 7)
 
        Returns
        -------
        logits       : (B, n_classes)
        attn_weights : (B, n_heads, 1, 216)  or None for ablation modes
        """
        # ── MRI stream ────────────────────────────────────────────────────────
        patches, global_vec = self.mri_encoder(mri)
        # patches:    (B, 216, feature_dim)
        # global_vec: (B, feature_dim)
 
        # ── Clinical stream ───────────────────────────────────────────────────
        clin_token = self.clin_encoder(clinical)
        # clin_token: (B, 1, feature_dim)
 
        attn_weights = None   # overwritten only in 'full' mode
 
        # ── Mode-specific fusion ──────────────────────────────────────────────
 
        if self.mode == 'full':
            # Clinical token queries MRI patches via cross-attention
            fused, attn_weights = self.fusion(patches, global_vec, clin_token)
            # fused: (B, feature_dim)
 
        elif self.mode == 'mri_only':
            # Ignore clinical entirely — baseline
            fused = global_vec
            # fused: (B, feature_dim)
 
        elif self.mode == 'clinical_only':
            # Ignore MRI entirely — baseline
            fused = clin_token.squeeze(1)
            # fused: (B, feature_dim)
 
        elif self.mode == 'concat_only':
            # Naive fusion — no attention, no interaction — ablation
            clin_vec = clin_token.squeeze(1)                          # (B, D)
            fused    = self.concat_proj(
                torch.cat([global_vec, clin_vec], dim=-1)             # (B, 2D)
            )
            # fused: (B, feature_dim)
 
        # ── Classifier ────────────────────────────────────────────────────────
        logits = self.classifier(fused)   # (B, n_classes)
 
        return logits, attn_weights
 
    # ── Inference helpers ─────────────────────────────────────────────────────
 
    def predict_proba(self, mri:      torch.Tensor,
                            clinical: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities. Use at inference, NOT during training.
        Returns: (B, n_classes) in [0,1], sums to 1 per row.
        """
        logits, _ = self.forward(mri, clinical)
        return F.softmax(logits, dim=-1)
 
    def predict(self, mri:      torch.Tensor,
                      clinical: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted class index.
        Returns: (B,) int64
        """
        logits, _ = self.forward(mri, clinical)
        return logits.argmax(dim=-1)
 
    # ── Staged training helpers ───────────────────────────────────────────────
 
    def freeze_mri_encoder(self):
        """
        Stage 1 of training: freeze the MRI encoder entirely.
 
        Why: The MRI encoder has ~3.5M parameters. With only 332 training
        subjects it will overfit badly if trained from scratch end-to-end.
        Freezing it forces the model to first learn meaningful fusion and
        classification using fixed MRI features.
 
        Call this BEFORE creating the optimizer so frozen params are
        excluded from the parameter groups.
 
        Training flow:
            model.freeze_mri_encoder()
            optimizer = AdamW(model.parameters(), lr=1e-4)
            # trains: clinical encoder + fusion + classifier only
        """
        for param in self.mri_encoder.parameters():
            param.requires_grad = False
        print("MRI encoder frozen — training fusion + classifier only")
        self._print_trainable()
 
    def unfreeze_mri_encoder(self):
        """
        Stage 2 of training: unfreeze MRI encoder for fine-tuning.
 
        Call this when validation loss plateaus in Stage 1.
        Use a much lower learning rate for the MRI encoder to avoid
        destroying the features it already learned.
 
        Training flow:
            model.unfreeze_mri_encoder()
            optimizer = AdamW([
                {'params': model.mri_encoder.parameters(), 'lr': 1e-5},
                {'params': model.other_params(),           'lr': 1e-4},
            ])
        """
        for param in self.mri_encoder.parameters():
            param.requires_grad = True
        print("MRI encoder unfrozen — fine-tuning with differential lr")
        self._print_trainable()
 
    def other_params(self):
        """
        Returns all parameters EXCEPT the MRI encoder.
        Used to set differential learning rates in Stage 2.
 
        Example:
            optimizer = AdamW([
                {'params': model.mri_encoder.parameters(), 'lr': 1e-5},
                {'params': model.other_params(),           'lr': 1e-4},
            ])
        """
        mri_ids = set(id(p) for p in self.mri_encoder.parameters())
        return [p for p in self.parameters() if id(p) not in mri_ids]
 
    def _print_trainable(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen    = total - trainable
        print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}  "
              f"|  Total: {total:,}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
 
def model_summary(model: nn.Module):
    """Print parameter count per submodule, showing frozen status."""
    print(f"\n{'Component':<30} {'Parameters':>12}  {'Trainable':>12}")
    print('─' * 58)
    total = trainable_total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        t = sum(p.numel() for p in module.parameters()
                if p.requires_grad)
        total           += n
        trainable_total += t
        frozen_mark = '  ❄' if t == 0 else ''
        print(f"  {name:<28} {n:>12,}  {t:>12,}{frozen_mark}")
    print('─' * 58)
    print(f"  {'TOTAL':<28} {total:>12,}  {trainable_total:>12,}\n")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
 
    B = 2
    dummy_mri  = torch.randn(B, 1, 96, 96, 96, device=device)
    dummy_clin = torch.randn(B, 7,           device=device)
 
    # ── Test all 4 modes ──────────────────────────────────────────────────────
    print("=== Mode check ===")
    for mode in AlzheimerFusionModel.VALID_MODES:
        m = AlzheimerFusionModel(mode=mode).to(device)
        m.eval()
        with torch.no_grad():
            logits, attn_w = m(dummy_mri, dummy_clin)
        attn_str = str(tuple(attn_w.shape)) if attn_w is not None else 'None'
        print(f"  {mode:<16} logits={tuple(logits.shape)}  "
              f"attn={attn_str}")
 
    # ── Full model parameter summary ──────────────────────────────────────────
    print("\n=== Full model — all trainable ===")
    full = AlzheimerFusionModel(mode='full').to(device)
    model_summary(full)
 
    # ── Stage 1: freeze MRI encoder ──────────────────────────────────────────
    print("=== Stage 1 — freeze MRI encoder ===")
    full.freeze_mri_encoder()
    model_summary(full)
 
    # ── Stage 2: unfreeze for fine-tuning ────────────────────────────────────
    print("=== Stage 2 — unfreeze for fine-tuning ===")
    full.unfreeze_mri_encoder()
    model_summary(full)
 
    # ── End-to-end gradient check ─────────────────────────────────────────────
    print("=== Gradient check ===")
    full.train()
    y    = torch.tensor([0, 1], dtype=torch.long, device=device)
    w    = torch.tensor([0.179, 0.846, 1.975], device=device)
    loss = nn.CrossEntropyLoss(weight=w)(
        full(dummy_mri, dummy_clin)[0], y
    )
    loss.backward()
    print(f"  Loss: {loss.item():.4f}  ✅ gradients flow end-to-end")