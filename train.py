"""
train.py
--------
Training script for AlzheimerFusionModel.

Staged training strategy (for 332 training subjects):
  Stage 1 — freeze MRI encoder, train fusion + classifier (~30 epochs)
             Goal: learn meaningful fusion without overfitting the 3.5M-param CNN
  Stage 2 — unfreeze MRI encoder, fine-tune everything with differential lr (~20 epochs)
             Goal: fine-tune MRI features now that fusion is stable

Usage (in Colab):
    !python train.py

Or import and call directly in a notebook:
    from train import train
    train()
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/MyDrive/dataset/oasis13d')

# ── Project imports ───────────────────────────────────────────────────────────
from src.preprocessing import (
    ClinicalPreprocessor, compute_class_weights, CDR_TO_CLASS
)
from src.dataset  import build_subject_list, OASISDataset
from src.model    import AlzheimerFusionModel, model_summary
from src.fusion   import attention_to_volume


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# All hyperparameters in one place — change here, never inside functions
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    'base_path'   : '/content/drive/MyDrive/dataset/oasis13d/data',
    'csv_path'    : '/content/drive/MyDrive/dataset/oasis13d/data/oasis_cross-sectional.csv',
    'nii_dir'     : '/content/drive/MyDrive/dataset/oasis13d/data/oasis/OASIS',
    'ckpt_dir'    : '/content/drive/MyDrive/dataset/oasis13d/checkpoints',

    # ── Model ──────────────────────────────────────────────────────────────
    'feature_dim' : 256,
    'n_heads'     : 4,
    'dropout_enc' : 0.3,
    'dropout_head': 0.5,

    # ── Stage 1 (frozen MRI encoder) ───────────────────────────────────────
    'stage1_epochs'  : 30,
    'stage1_lr'      : 1e-4,
    'stage1_patience': 8,      # early stopping patience

    # ── Stage 2 (fine-tune everything) ─────────────────────────────────────
    'stage2_epochs'  : 20,
    'stage2_lr'      : 1e-4,   # for clinical enc + fusion + classifier
    'stage2_lr_mri'  : 1e-5,   # much lower for MRI encoder (fine-tune)
    'stage2_patience': 8,

    # ── DataLoader ─────────────────────────────────────────────────────────
    'batch_size'  : 4,         # small batch — limited GPU memory + 416 subjects
    'num_workers' : 2,
    'val_split'   : 0.2,
    'random_seed' : 42,

    # ── Regularization ─────────────────────────────────────────────────────
    'weight_decay': 1e-4,
}


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor,
                    labels: torch.Tensor,
                    n_classes: int = 3) -> dict:
    """
    Compute accuracy and per-class accuracy from a batch.

    Args
    ----
    logits : (B, n_classes)  raw model output
    labels : (B,)            ground truth class indices

    Returns
    -------
    dict with 'acc' and 'per_class_acc'
    """
    preds   = logits.argmax(dim=-1)                    # (B,)
    correct = (preds == labels).float()

    acc = correct.mean().item()

    per_class = {}
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            per_class[c] = correct[mask].mean().item()
        else:
            per_class[c] = float('nan')

    return {'acc': acc, 'per_class_acc': per_class}


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model:     nn.Module,
              loader:    DataLoader,
              criterion: nn.Module,
              optimizer: torch.optim.Optimizer | None,
              device:    torch.device,
              train:     bool) -> dict:
    """
    Run one full epoch (train or val).

    Args
    ----
    model     : AlzheimerFusionModel
    loader    : DataLoader
    criterion : CrossEntropyLoss with class weights
    optimizer : AdamW (None during validation)
    device    : cuda or cpu
    train     : True = training mode, False = eval mode

    Returns
    -------
    dict with 'loss' and 'acc' averaged over the epoch
    """
    model.train(train)

    total_loss = 0.0
    all_logits = []
    all_labels = []

    for mri, clinical, labels in loader:
        mri      = mri.to(device)
        clinical = clinical.to(device)
        labels   = labels.to(device)

        if train:
            optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(train):
            logits, _ = model(mri, clinical)
            loss      = criterion(logits, labels)

        if train:
            loss.backward()
            # Gradient clipping — prevents exploding gradients
            # especially important with cross-attention on small data
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Aggregate metrics over full epoch
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(loader)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model:     nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch:     int,
                    val_acc:   float,
                    stage:     int,
                    cfg:       dict):
    """Save model + optimizer state to Drive."""
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    path = os.path.join(cfg['ckpt_dir'], f'best_stage{stage}.pt')
    torch.save({
        'epoch'      : epoch,
        'val_acc'    : val_acc,
        'stage'      : stage,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'cfg'        : cfg,
    }, path)
    print(f"  ✅ Checkpoint saved → {path}  (val_acc={val_acc:.4f})")


def load_checkpoint(model:     nn.Module,
                    optimizer: torch.optim.Optimizer | None,
                    path:      str,
                    device:    torch.device) -> dict:
    """Load checkpoint. Returns the saved dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STAGE
# ─────────────────────────────────────────────────────────────────────────────

def train_stage(model:      nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                criterion:    nn.Module,
                optimizer:    torch.optim.Optimizer,
                scheduler:    torch.optim.lr_scheduler._LRScheduler,
                device:       torch.device,
                n_epochs:     int,
                patience:     int,
                stage:        int,
                cfg:          dict) -> float:
    """
    Train for one stage (fixed optimizer + scheduler).

    Returns
    -------
    best_val_acc : float  — best validation accuracy achieved in this stage
    """
    best_val_acc  = 0.0
    patience_ctr  = 0

    print(f"\n{'='*60}")
    print(f"  STAGE {stage}  —  {n_epochs} epochs  |  patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        tr = run_epoch(model, train_loader, criterion,
                       optimizer, device, train=True)

        # ── Validate ─────────────────────────────────────────────────────────
        va = run_epoch(model, val_loader, criterion,
                       None, device, train=False)

        scheduler.step(va['loss'])

        elapsed = time.time() - t0

        # ── Log ──────────────────────────────────────────────────────────────
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"| tr_loss={tr['loss']:.4f}  tr_acc={tr['acc']:.4f}  "
              f"| va_loss={va['loss']:.4f}  va_acc={va['acc']:.4f}  "
              f"| {elapsed:.1f}s")

        # ── Per-class val accuracy (helps spot class 2 collapse) ─────────────
        pca = va['per_class_acc']
        print(f"           val per-class: "
              f"Non-dem={pca.get(0, float('nan')):.3f}  "
              f"VeryMild={pca.get(1, float('nan')):.3f}  "
              f"Mild/Mod={pca.get(2, float('nan')):.3f}")

        # ── Save best ────────────────────────────────────────────────────────
        if va['acc'] > best_val_acc:
            best_val_acc = va['acc']
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch,
                            best_val_acc, stage, cfg)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping — no improvement for "
                      f"{patience} epochs")
                break

    print(f"\n  Stage {stage} complete — best val_acc: {best_val_acc:.4f}")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict = CFG):
    """
    Full staged training pipeline.

    Stage 1: freeze MRI encoder → train fusion + classifier
    Stage 2: unfreeze MRI encoder → fine-tune everything
    """

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n── Loading data ──")
    df = build_subject_list(cfg['csv_path'], cfg['nii_dir'])

    labels   = df['CDR'].map(CDR_TO_CLASS).values
    df_tr, df_va = train_test_split(
        df,
        test_size   = cfg['val_split'],
        random_state= cfg['random_seed'],
        stratify    = labels,
    )
    print(f"Train: {len(df_tr)}  Val: {len(df_va)}")

    # ── Preprocessor ─────────────────────────────────────────────────────────
    prep = ClinicalPreprocessor()
    prep.fit_transform(df_tr)   # fit on training data only

    # ── Datasets & Loaders ───────────────────────────────────────────────────
    train_ds = OASISDataset(df_tr, prep, augment=True)
    val_ds   = OASISDataset(df_va, prep, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = cfg['num_workers'],
        pin_memory  = device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers'],
        pin_memory  = device.type == 'cuda',
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    # Recompute from training labels after split
    _, y_tr = prep.transform(df_tr)
    weights  = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    model = AlzheimerFusionModel(
        feature_dim   = cfg['feature_dim'],
        n_heads       = cfg['n_heads'],
        dropout_enc   = cfg['dropout_enc'],
        dropout_head  = cfg['dropout_head'],
        mode          = 'full',
    ).to(device)
    model_summary(model)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — frozen MRI encoder
    # Train: clinical encoder + fusion + classifier
    # MRI encoder: frozen (no gradients, no updates)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Stage 1: freeze MRI encoder ──")
    model.freeze_mri_encoder()

    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg['stage1_lr'],
        weight_decay = cfg['weight_decay'],
    )
    scheduler_s1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s1,
        mode     = 'min',
        factor   = 0.5,
        patience = 4,
    )

    best_s1 = train_stage(
        model, train_loader, val_loader,
        criterion, optimizer_s1, scheduler_s1,
        device,
        n_epochs = cfg['stage1_epochs'],
        patience = cfg['stage1_patience'],
        stage    = 1,
        cfg      = cfg,
    )

    # Load best stage 1 weights before starting stage 2
    best_s1_path = os.path.join(cfg['ckpt_dir'], 'best_stage1.pt')
    load_checkpoint(model, None, best_s1_path, device)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — unfreeze MRI encoder, fine-tune with differential lr
    # MRI encoder : lr = 1e-5  (10× smaller — don't destroy learned features)
    # Everything else : lr = 1e-4
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Stage 2: unfreeze MRI encoder ──")
    model.unfreeze_mri_encoder()

    optimizer_s2 = torch.optim.AdamW([
        {'params': model.mri_encoder.parameters(),
         'lr'    : cfg['stage2_lr_mri'],        # 1e-5 — fine-tune carefully
         'weight_decay': cfg['weight_decay']},
        {'params': model.other_params(),
         'lr'    : cfg['stage2_lr'],             # 1e-4 — normal lr
         'weight_decay': cfg['weight_decay']},
    ])
    scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2,
        mode     = 'min',
        factor   = 0.5,
        patience = 4,
    )

    best_s2 = train_stage(
        model, train_loader, val_loader,
        criterion, optimizer_s2, scheduler_s2,
        device,
        n_epochs = cfg['stage2_epochs'],
        patience = cfg['stage2_patience'],
        stage    = 2,
        cfg      = cfg,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Stage 1 best val_acc : {best_s1:.4f}")
    print(f"  Stage 2 best val_acc : {best_s2:.4f}")
    print(f"  Best checkpoint      : {cfg['ckpt_dir']}/best_stage2.pt")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
