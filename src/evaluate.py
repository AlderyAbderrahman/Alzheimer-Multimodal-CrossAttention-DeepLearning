"""
evaluate.py
-----------
Evaluation script for AlzheimerFusionModel.

What this file produces:
  1. Overall accuracy, weighted F1, Cohen's Kappa
  2. Per-class precision, recall, F1
  3. Confusion matrix (printed + saved as .png)
  4. Attention map visualization for sample subjects
  5. Feature importance from clinical encoder
  6. Ablation study — compare all 4 modes on val set

Usage (in Colab):
    %run /content/drive/MyDrive/dataset/oasis13d/evaluate.py

Or call specific functions:
    from evaluate import evaluate_model, ablation_study
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/MyDrive/dataset/oasis13d')

# ── Project imports ───────────────────────────────────────────────────────────
from src.preprocessing import (
    ClinicalPreprocessor, compute_class_weights,
    CDR_TO_CLASS, CLASS_NAMES
)
from src.dataset  import build_subject_list, OASISDataset
from src.model    import AlzheimerFusionModel, model_summary
from src.fusion   import attention_to_volume


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    'csv_path'    : '/content/drive/MyDrive/dataset/oasis13d/data/oasis_cross-sectional.csv',
    'nii_dir'     : '/content/mri_cache',
    'ckpt_dir'    : '/content/drive/MyDrive/dataset/oasis13d/checkpoints',
    'output_dir'  : '/content/drive/MyDrive/dataset/oasis13d/results',
    'feature_dim' : 256,
    'n_heads'     : 4,
    'dropout_enc' : 0.3,
    'dropout_head': 0.5,
    'batch_size'  : 4,
    'num_workers' : 2,
    'val_split'   : 0.2,
    'random_seed' : 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA SETUP  (same split as train.py — same random_seed guarantees same val set)
# ─────────────────────────────────────────────────────────────────────────────

def get_val_loader(cfg: dict):
    """
    Rebuild the exact same val split used during training.
    Same random_seed + same stratify = identical split every time.
    Returns (val_loader, prep, df_va)
    """
    df     = build_subject_list(cfg['csv_path'], cfg['nii_dir'])
    labels = df['CDR'].map(CDR_TO_CLASS).values

    df_tr, df_va = train_test_split(
        df,
        test_size    = cfg['val_split'],
        random_state = cfg['random_seed'],
        stratify     = labels,
    )

    prep = ClinicalPreprocessor()
    prep.fit_transform(df_tr)   # fit on train only — same as during training

    val_ds = OASISDataset(df_va, prep, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers'],
    )
    return val_loader, prep, df_va


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL FROM CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str,
               cfg:       dict,
               device:    torch.device,
               mode:      str = 'full') -> AlzheimerFusionModel:
    """
    Build model and load weights from checkpoint.
    """
    model = AlzheimerFusionModel(
        feature_dim   = cfg['feature_dim'],
        n_heads       = cfg['n_heads'],
        dropout_enc   = cfg['dropout_enc'],
        dropout_head  = cfg['dropout_head'],
        mode          = mode,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Load only keys that match current mode
    # (ablation models are subsets of the full model)
    state = ckpt['model_state']
    model.load_state_dict(state, strict=False)
    print(f"Loaded: {ckpt_path}  (epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model:      AlzheimerFusionModel,
                   val_loader: DataLoader,
                   device:     torch.device,
                   cfg:        dict) -> dict:
    """
    Run full evaluation on the validation set.

    Computes:
      - Overall accuracy
      - Weighted F1 score
      - Cohen's Kappa (agreement beyond chance — key metric for medical AI)
      - Per-class precision, recall, F1
      - Confusion matrix

    Returns dict of all metrics.
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for mri, clinical, labels in val_loader:
            mri      = mri.to(device)
            clinical = clinical.to(device)

            logits, _ = model(mri, clinical)
            probs     = torch.softmax(logits, dim=-1)
            preds     = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc   = (all_preds == all_labels).mean()
    f1    = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm    = confusion_matrix(all_labels, all_preds)
    report= classification_report(
                all_labels, all_preds,
                target_names=CLASS_NAMES,
                digits=3
            )

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted F1    : {f1:.4f}")
    print(f"  Cohen's Kappa  : {kappa:.4f}  ", end='')
    # Kappa interpretation for thesis
    if   kappa >= 0.81: print("(Almost perfect agreement)")
    elif kappa >= 0.61: print("(Substantial agreement)")
    elif kappa >= 0.41: print("(Moderate agreement)")
    else:               print("(Fair agreement)")
    print(f"\n{report}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(cm, cfg)

    return {
        'acc'      : acc,
        'f1'       : f1,
        'kappa'    : kappa,
        'cm'       : cm,
        'preds'    : all_preds,
        'labels'   : all_labels,
        'probs'    : all_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm:  np.ndarray,
                          cfg: dict,
                          title: str = 'Confusion Matrix'):
    """
    Plot and save a normalized confusion matrix.
    Normalized = each row sums to 1 (shows recall per class).
    """
    os.makedirs(cfg['output_dir'], exist_ok=True)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, t in zip(
        axes,
        [cm, cm_norm],
        ['Raw counts', 'Normalized (recall)']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{title} — {t}', fontsize=13)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = f'{data[i,j]:.2f}' if t == 'Normalized (recall)' \
                      else str(int(data[i,j]))
                ax.text(j, i, val,
                        ha='center', va='center',
                        color='white' if data[i,j] > thresh else 'black',
                        fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(cfg['output_dir'], 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — ATTENTION MAP VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_attention(model:      AlzheimerFusionModel,
                        val_loader: DataLoader,
                        df_va:      object,
                        device:     torch.device,
                        cfg:        dict,
                        n_subjects: int = 3):
    """
    For n_subjects from each class, visualize which brain regions
    the model attended to when making its prediction.

    Saves one figure per subject showing:
      - Middle axial MRI slice
      - Attention heatmap overlaid on the MRI
      - True label and predicted label
    """
    os.makedirs(cfg['output_dir'], exist_ok=True)
    model.eval()

    shown = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for mri, clinical, labels in val_loader:
            mri_d    = mri.to(device)
            clin_d   = clinical.to(device)

            logits, attn_w = model(mri_d, clin_d)
            preds          = logits.argmax(dim=-1)

            if attn_w is None:
                print("Attention maps not available in this mode.")
                return

            # Reshape attention to 3D volume (B, 6, 6, 6)
            attn_vol = attention_to_volume(attn_w)   # (B, 6, 6, 6)

            for i in range(mri.shape[0]):
                label = labels[i].item()
                pred  = preds[i].item()

                if shown[label] >= n_subjects:
                    continue

                shown[label] += 1

                # ── MRI middle axial slice ────────────────────────────────────
                mri_np  = mri[i, 0].numpy()          # (96,96,96)
                mid_z   = mri_np.shape[2] // 2
                mri_slice = mri_np[:, :, mid_z]      # (96,96)

                # ── Upsample attention to MRI size ────────────────────────────
                attn_np = attn_vol[i].unsqueeze(0).unsqueeze(0)  # (1,1,6,6,6)
                attn_up = torch.nn.functional.interpolate(
                    attn_np.cpu(),
                    size   = (96, 96, 96),
                    mode   = 'trilinear',
                    align_corners = False,
                ).squeeze().numpy()                  # (96,96,96)
                attn_slice = attn_up[:, :, mid_z]    # (96,96)

                # ── Plot ──────────────────────────────────────────────────────
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                axes[0].imshow(mri_slice.T, cmap='gray', origin='lower')
                axes[0].set_title('MRI (axial slice)')
                axes[0].axis('off')

                axes[1].imshow(attn_slice.T, cmap='hot', origin='lower')
                axes[1].set_title('Attention map')
                axes[1].axis('off')

                axes[2].imshow(mri_slice.T,  cmap='gray',
                               origin='lower', alpha=0.7)
                axes[2].imshow(attn_slice.T, cmap='hot',
                               origin='lower', alpha=0.5)
                axes[2].set_title(
                    f'Overlay\nTrue: {CLASS_NAMES[label]}  '
                    f'Pred: {CLASS_NAMES[pred]}'
                )
                axes[2].axis('off')

                correct_str = '[CORRECT]' if label == pred else '[WRONG]'
                fig.suptitle(
                    f'Subject attention map  {correct_str}  '
                    f'True={CLASS_NAMES[label]}  '
                    f'Pred={CLASS_NAMES[pred]}',
                    fontsize=13
                )
                plt.tight_layout()

                save_path = os.path.join(
                    cfg['output_dir'],
                    f'attn_class{label}_subj{shown[label]}.png'
                )
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Attention map saved → {save_path}")

            if all(v >= n_subjects for v in shown.values()):
                break


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model:      AlzheimerFusionModel,
                       val_loader: DataLoader,
                       device:     torch.device,
                       cfg:        dict):
    """
    Show which clinical features the model learned to rely on most.
    Uses the FeatureWiseAttention gate weights from ClinicalMLPEncoder.

    This is a key thesis figure — it should show MMSE and nWBV
    ranked highest, confirming the model learned clinically meaningful
    feature weighting.
    """
    from src.preprocessing import FEATURE_COLS

    model.eval()
    all_weights = []

    with torch.no_grad():
        for _, clinical, _ in val_loader:
            clinical = clinical.to(device)
            # Get gate weights from clinical encoder
            w = model.clin_encoder.get_feature_importance(clinical)
            all_weights.append(
                np.array([w[f] for f in FEATURE_COLS])
            )

    mean_weights = np.mean(all_weights, axis=0)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  CLINICAL FEATURE IMPORTANCE")
    print(f"  (gate weights from FeatureWiseAttention)")
    print(f"{'='*45}")
    sorted_idx = np.argsort(mean_weights)[::-1]
    for i in sorted_idx:
        bar = '█' * int(mean_weights[i] * 30)
        print(f"  {FEATURE_COLS[i]:<8} {bar:<30} {mean_weights[i]:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg['output_dir'], exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2196F3' if i == sorted_idx[0] else
              '#64B5F6' if i == sorted_idx[1] else
              '#BBDEFB' for i in range(len(FEATURE_COLS))]

    sorted_feats   = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_weights = [mean_weights[i]  for i in sorted_idx]

    bars = ax.barh(sorted_feats, sorted_weights, color=colors)
    ax.set_xlabel('Mean gate weight', fontsize=12)
    ax.set_title('Clinical feature importance\n'
                 '(learned by FeatureWiseAttention)', fontsize=13)
    ax.set_xlim(0, 1.0)

    for bar, w in zip(bars, sorted_weights):
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(cfg['output_dir'], 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance saved → {save_path}")

    return dict(zip(FEATURE_COLS, mean_weights))


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def ablation_study(val_loader: DataLoader,
                   device:     torch.device,
                   cfg:        dict):
    """
    Compare all 4 model modes on the same val set.
    This is Table 2 in your thesis.

    Modes compared:
        clinical_only  → only clinical features
        mri_only       → only MRI
        concat_only    → MRI + clinical, naive concatenation
        full           → MRI + clinical, cross-attention fusion (proposed)
    """
    ckpt_path = os.path.join(cfg['ckpt_dir'], 'best_stage1.pt')

    results = {}
    modes   = ['clinical_only', 'mri_only', 'concat_only', 'full']

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY")
    print(f"{'='*65}")
    print(f"  {'Mode':<18} {'Accuracy':>10} {'F1 (weighted)':>15} "
          f"{'Kappa':>10}")
    print(f"  {'-'*55}")

    for mode in modes:
        model = load_model(ckpt_path, cfg, device, mode=mode)
        model.eval()

        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for mri, clinical, labels in val_loader:
                mri      = mri.to(device)
                clinical = clinical.to(device)
                logits, _= model(mri, clinical)
                preds    = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc   = (all_preds == all_labels).mean()
        f1    = f1_score(all_labels, all_preds, average='weighted')
        kappa = cohen_kappa_score(all_labels, all_preds)

        marker = ' ← proposed' if mode == 'full' else ''
        print(f"  {mode:<18} {acc:>10.4f} {f1:>15.4f} "
              f"{kappa:>10.4f}{marker}")

        results[mode] = {'acc': acc, 'f1': f1, 'kappa': kappa}

    print(f"{'='*65}\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    os.makedirs(cfg['output_dir'], exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics   = ['acc', 'f1', 'kappa']
    titles    = ['Accuracy', 'Weighted F1', "Cohen's Kappa"]
    colors    = ['#BBDEFB', '#BBDEFB', '#BBDEFB', '#1565C0']  # last = proposed

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results[m][metric] for m in modes]
        bars = ax.bar(modes, vals, color=colors)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(modes, rotation=20, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.3f}', ha='center', fontsize=9)

    plt.suptitle('Ablation Study — Model Component Contributions',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(cfg['output_dir'], 'ablation_study.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Ablation chart saved → {save_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(CFG['output_dir'], exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading validation data ──")
    val_loader, prep, df_va = get_val_loader(CFG)

    # ── Load best model (Stage 1 — 90.48%) ───────────────────────────────────
    print("\n── Loading best model ──")
    ckpt_path = os.path.join(CFG['ckpt_dir'], 'best_stage1.pt')
    model     = load_model(ckpt_path, CFG, device, mode='full')

    # ── Part 1: Full evaluation ───────────────────────────────────────────────
    print("\n── Part 1: Full evaluation ──")
    metrics = evaluate_model(model, val_loader, device, CFG)

    # ── Part 2: Feature importance ────────────────────────────────────────────
    print("\n── Part 2: Feature importance ──")
    feat_imp = feature_importance(model, val_loader, device, CFG)

    # ── Part 3: Attention maps ────────────────────────────────────────────────
    print("\n── Part 3: Attention maps ──")
    visualize_attention(model, val_loader, df_va, device, CFG, n_subjects=2)

    # ── Part 4: Ablation study ────────────────────────────────────────────────
    print("\n── Part 4: Ablation study ──")
    ablation_results = ablation_study(val_loader, device, CFG)

    print("\n✅ Evaluation complete — all figures saved to:")
    print(f"   {CFG['output_dir']}")