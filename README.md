alzheimer-multimodal/
│
├── data/
│   ├── raw/                    # .nii files (Kaggle dataset input)
│   └── oasis_clinical.csv      #  CSV file
│
├── src/                        # all reusable Python modules
│   ├── __init__.py
│   ├── dataset.py              # Dataset class, data loading, CSV join
│   ├── preprocessing.py        # normalize, resize, augmentation
│   ├── encoder_mri.py          # BrainMRI3DEncoder  ← what we just wrote
│   ├── encoder_clinical.py     # Clinical MLP encoder  ← next step
│   ├── fusion.py               # Cross-attention fusion module
│   ├── model.py                # Full model combining all streams
│   └── utils.py                # metrics, visualization, Grad-CAM
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # look at your NIfTI files + CSV
│   ├── 02_preprocessing_check.ipynb  # verify shapes, plot slices
│   ├── 03_baseline_experiments.ipynb # CNN only, MLP only baselines
│   └── 04_fusion_model.ipynb         # full model training + results
│
├── train.py                    # main training script (runs on Kaggle)
├── evaluate.py                 # evaluation + confusion matrix
└── requirements.txt
