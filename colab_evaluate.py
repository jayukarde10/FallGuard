# ============================================================
#  FallGuard AI — Google Colab Evaluation Script
#  C3D Spatiotemporal Model × GMDCSA24 Human Fall Dataset
#
#  HOW TO USE:
#    1. Upload files to Google Drive (see SECTION 0 below).
#    2. Open this file in Colab:
#         File → Upload notebook → select colab_evaluate.py
#       OR copy-paste each CELL section into Colab code cells.
#    3. Run cells top-to-bottom.
#    4. GPU runtime recommended (Runtime → Change runtime type → T4 GPU)
#       but CPU works fine for inference/evaluation.
# ============================================================

# ────────────────────────────────────────────────────────────
# SECTION 0  ·  WHAT TO UPLOAD TO GOOGLE DRIVE
# ────────────────────────────────────────────────────────────
# Place the following inside a single Drive folder, e.g.:
#   MyDrive/FallGuardAI/
#
#   FallGuardAI/
#   ├── weights.h5                         <- C3D Keras weights (305 MB)
#   └── dataset/                           <- GMDCSA24 dataset root
#       ├── Subject 1/
#       │   ├── Fall.csv
#       │   ├── ADL.csv
#       │   ├── Fall/   (*.mp4 videos)
#       │   └── ADL/    (*.mp4 videos)
#       ├── Subject 2/  (same structure)
#       ├── Subject 3/  (same structure)
#       └── Subject 4/  (same structure)
#
# After running CELL 5 (feature extraction), the file
#   models/dataset_features.npz
# will be saved to your Drive so you can skip re-extraction
# on future runs.
# ────────────────────────────────────────────────────────────


# ============================================================
# CELL 1 — Install Dependencies
# ============================================================

# !pip install -q torch torchvision h5py opencv-python-headless \
#              scikit-learn pandas matplotlib seaborn plotly joblib

# --- uncomment and run the line above in Colab ---


# ============================================================
# CELL 2 — Mount Google Drive & Set Paths
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os

# ── EDIT THIS PATH to match your Drive folder ──────────────
DRIVE_ROOT = "/content/drive/MyDrive/FallGuardAI"
# ───────────────────────────────────────────────────────────

WEIGHTS_PATH  = os.path.join(DRIVE_ROOT, "weights.h5")
DATASET_ROOT  = os.path.join(DRIVE_ROOT, "dataset")
MODELS_DIR    = os.path.join(DRIVE_ROOT, "models")
FEATURES_PATH = os.path.join(MODELS_DIR, "dataset_features.npz")
HEAD_PATH     = os.path.join(MODELS_DIR, "fall_head.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

assert os.path.exists(WEIGHTS_PATH), \
    f"ERROR: weights.h5 not found at {WEIGHTS_PATH}. Check Drive layout."
assert os.path.isdir(DATASET_ROOT), \
    f"ERROR: Dataset folder not found at {DATASET_ROOT}. Check Drive layout."

print("Drive mounted and paths verified.")
print(f"  weights.h5 : {WEIGHTS_PATH}")
print(f"  dataset/   : {DATASET_ROOT}")


# ============================================================
# CELL 3 — C3D Model Definition & Keras H5 Weight Loader
# ============================================================

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class C3D(nn.Module):
    """
    C3D (Convolutional 3D Network) - Tran et al. 2015.
    Pre-trained on Sports-1M (487 action classes).
    Used here as a spatiotemporal feature extractor (FC7 -> 4096-d).

    Input  : (B, 3, 16, 112, 112)  -- B clips x RGB x T x H x W
    Output : logits (B, 487)  +  fc7 features (B, 4096)
    """
    def __init__(self, num_classes=487):
        super().__init__()
        self.conv1  = nn.Conv3d(3,   64,  kernel_size=3, padding=1)
        self.pool1  = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2  = nn.Conv3d(64,  128, kernel_size=3, padding=1)
        self.pool2  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.relu(self.conv1(x));  h = self.pool1(h)
        h = F.relu(self.conv2(h));  h = self.pool2(h)
        h = F.relu(self.conv3a(h)); h = F.relu(self.conv3b(h)); h = self.pool3(h)
        h = F.relu(self.conv4a(h)); h = F.relu(self.conv4b(h)); h = self.pool4(h)
        h = F.relu(self.conv5a(h)); h = F.relu(self.conv5b(h))
        h = F.pad(h, (0, 1, 0, 1, 0, 0))  # right/bottom spatial pad
        h = self.pool5(h)
        h = h.permute(0, 2, 3, 4, 1).contiguous().view(h.size(0), -1)
        f6 = F.relu(self.fc6(h))
        f7 = F.relu(self.fc7(f6))
        logits = self.fc8(f7)
        return logits, f7

    def extract_features(self, x):
        """Returns only the 4096-d FC7 spatiotemporal embeddings."""
        _, f7 = self.forward(x)
        return f7


def load_c3d_from_keras_h5(weights_path, device="cpu"):
    """
    Loads C3D and transfers Keras-format weights from an HDF5 file.
    Handles the (D,H,W,Cin,Cout) -> (Cout,Cin,D,H,W) permutation for Conv3D
    and the (in,out) -> (out,in) transpose for Linear layers.
    """
    model = C3D(num_classes=487)
    print(f"  Loading weights from {weights_path} ...", end="", flush=True)
    with h5py.File(weights_path, 'r') as f:
        conv_names = ['conv1', 'conv2', 'conv3a', 'conv3b',
                      'conv4a', 'conv4b', 'conv5a', 'conv5b']
        for ln in conv_names:
            w = f[ln][f'{ln}/kernel:0'][:]
            b = f[ln][f'{ln}/bias:0'][:]
            getattr(model, ln).weight.data.copy_(
                torch.from_numpy(w).permute(4, 3, 0, 1, 2))
            getattr(model, ln).bias.data.copy_(torch.from_numpy(b))
        for ln in ['fc6', 'fc7', 'fc8']:
            w = f[ln][f'{ln}/kernel:0'][:]
            b = f[ln][f'{ln}/bias:0'][:]
            getattr(model, ln).weight.data.copy_(torch.from_numpy(w).t())
            getattr(model, ln).bias.data.copy_(torch.from_numpy(b))
    model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f" done. ({total_params / 1e6:.1f}M parameters on {device})")
    return model


# ============================================================
# CELL 4 — Dataset Helpers: CSV Parsing & Video Clip Extraction
# ============================================================

import re
import cv2
import pandas as pd

# C3D mean pixel values (Sports-1M training statistics)
_MEAN = np.array([124.0, 117.0, 104.0], dtype=np.float32)


def parse_class_intervals(class_str):
    """
    Parses GMDCSA24 CSV 'Classes' column.
    E.g.  'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]'
    Returns [{'action': str, 'start': float, 'end': float, 'is_fall': bool}]
    """
    results = []
    if not isinstance(class_str, str):
        return results
    for part in [p.strip() for p in class_str.split(';') if p.strip()]:
        m = re.match(r'^(.*?)\s*\[\s*([\d\.]+)\s+to\s+([\d\.]+)\s*\]', part)
        if m:
            act = m.group(1).strip()
            results.append({
                'action':  act,
                'start':   float(m.group(2)),
                'end':     float(m.group(3)),
                'is_fall': 'fall' in act.lower()
            })
    return results


def get_video_clip(video_path, target_sec, num_frames=16):
    """
    Extracts a 16-frame clip centred near target_sec.
    Returns a normalised (1, 3, 16, 112, 112) float32 Tensor, or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, min(int(target_sec * fps), total_frames - num_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (112, 112)))
    cap.release()
    if len(frames) < num_frames:
        return None
    arr = (np.array(frames, dtype=np.float32) - _MEAN)  # (T,H,W,3)
    arr = np.transpose(arr, (3, 0, 1, 2))                # (3,T,H,W)
    return torch.from_numpy(arr).unsqueeze(0).float()    # (1,3,T,H,W)


def video_duration(video_path):
    """Returns duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


# ============================================================
# CELL 5 — Feature Extraction  (skips automatically if cached)
# ============================================================

def extract_all_features(dataset_root, weights_path, features_save_path, device="cpu"):
    """
    Iterates over all Subject 1-4 Fall/ADL clips, extracts 4096-d
    C3D FC7 features, and saves them as an NPZ cache file.
    On subsequent runs, it loads from cache instead of re-extracting.
    """
    if os.path.exists(features_save_path):
        print(f"Feature cache found. Loading from {features_save_path}")
        print("  (Delete this file to force re-extraction)")
        data = np.load(features_save_path, allow_pickle=True)
        return (data['X'], data['y'], data['subjects'], data['categories'],
                data['filenames'], data['durations'], data['descriptions'],
                data['fall_directions'], data['adl_actions'])

    model = load_c3d_from_keras_h5(weights_path, device=device)

    # Fall direction keyword mapping
    DIRECTION_MAP = {
        'SW': 'Sideways', 'SL': 'Sideways-Left', 'SR': 'Sideways-Right',
        'FW': 'Forward',  'BW': 'Backward',       'SD': 'Sideways'
    }

    X_list, y_list       = [], []
    subj_list, cat_list  = [], []
    fname_list, dur_list = [], []
    desc_list            = []
    fdir_list, adl_list  = [], []

    for subj in ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']:
        subj_dir = os.path.join(dataset_root, subj)
        if not os.path.isdir(subj_dir):
            print(f"  Warning: {subj} not found, skipping.")
            continue
        print(f"\n  Processing {subj} ...")

        # --- Falls ---
        fall_csv = os.path.join(subj_dir, 'Fall.csv')
        if os.path.exists(fall_csv):
            df_fall = pd.read_csv(fall_csv)
            for _, row in df_fall.iterrows():
                fname = str(row.get('File Name', '')).strip()
                vpath = os.path.join(subj_dir, 'Fall', fname)
                if not os.path.exists(vpath):
                    continue
                intervals = parse_class_intervals(row.get(' Classes', ''))
                fall_ints = [i for i in intervals if i['is_fall']]
                if fall_ints:
                    mid_sec = (fall_ints[0]['start'] + fall_ints[0]['end']) / 2.0
                    desc    = fall_ints[0]['action']
                else:
                    mid_sec = 2.5
                    desc    = 'Fall (unspecified)'
                clip = get_video_clip(vpath, mid_sec)
                if clip is None:
                    print(f"    Skipped {fname} (clip extraction failed)")
                    continue
                with torch.no_grad():
                    feat = model.extract_features(clip.to(device)).cpu().numpy()[0]
                direction = 'Unknown'
                for abbr, full in DIRECTION_MAP.items():
                    if f'({abbr})' in desc or abbr in desc:
                        direction = full
                        break
                X_list.append(feat);  y_list.append(1)
                subj_list.append(subj); cat_list.append('Fall')
                fname_list.append(fname); dur_list.append(video_duration(vpath))
                desc_list.append(desc); fdir_list.append(direction)
                adl_list.append('N/A')

        # --- ADL ---
        adl_csv = os.path.join(subj_dir, 'ADL.csv')
        if os.path.exists(adl_csv):
            df_adl = pd.read_csv(adl_csv)
            for _, row in df_adl.iterrows():
                fname = str(row.get('File Name', '')).strip()
                vpath = os.path.join(subj_dir, 'ADL', fname)
                if not os.path.exists(vpath):
                    continue
                intervals = parse_class_intervals(row.get(' Classes', ''))
                desc = intervals[0]['action'] if intervals else 'ADL'
                clip = get_video_clip(vpath, 2.0)
                if clip is None:
                    print(f"    Skipped {fname} (clip extraction failed)")
                    continue
                with torch.no_grad():
                    feat = model.extract_features(clip.to(device)).cpu().numpy()[0]
                al = desc.lower()
                if 'sit' in al:             adl_action = 'Sitting'
                elif 'walk' in al:          adl_action = 'Walking'
                elif 'sleep' in al or 'lie' in al: adl_action = 'Sleeping/Lying'
                elif 'read' in al:          adl_action = 'Reading'
                elif 'stand' in al:         adl_action = 'Standing'
                else:                       adl_action = 'Other ADL'
                X_list.append(feat);  y_list.append(0)
                subj_list.append(subj); cat_list.append('ADL')
                fname_list.append(fname); dur_list.append(video_duration(vpath))
                desc_list.append(desc); fdir_list.append('N/A')
                adl_list.append(adl_action)

    X           = np.array(X_list,    dtype=np.float32)
    y           = np.array(y_list,    dtype=np.int32)
    subjects    = np.array(subj_list)
    categories  = np.array(cat_list)
    filenames   = np.array(fname_list)
    durations   = np.array(dur_list,  dtype=np.float32)
    descriptions     = np.array(desc_list)
    fall_directions  = np.array(fdir_list)
    adl_actions      = np.array(adl_list)

    np.savez_compressed(
        features_save_path,
        X=X, y=y, subjects=subjects, categories=categories,
        filenames=filenames, durations=durations, descriptions=descriptions,
        fall_directions=fall_directions, adl_actions=adl_actions
    )
    print(f"\nFeatures extracted and saved to {features_save_path}")
    print(f"  Total: {len(y)} clips | Falls: {sum(y==1)} | ADL: {sum(y==0)}")
    return (X, y, subjects, categories, filenames, durations,
            descriptions, fall_directions, adl_actions)


# Determine compute device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Run extraction (or load from cache)
(X, y, subjects, categories, filenames, durations,
 descriptions, fall_directions, adl_actions) = extract_all_features(
    DATASET_ROOT, WEIGHTS_PATH, FEATURES_PATH, device=DEVICE
)

# Build metadata DataFrame
df_meta = pd.DataFrame({
    'subject':        subjects,
    'category':       categories,
    'file_name':      filenames,
    'duration':       durations,
    'description':    descriptions,
    'fall_direction': fall_directions,
    'adl_action':     adl_actions,
    'label':          y
})
print("\nClass distribution:")
print(df_meta['category'].value_counts().to_string())


# ============================================================
# CELL 6 — Train Logistic Regression Classification Head
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

if os.path.exists(HEAD_PATH):
    print(f"Classifier head found. Loading from {HEAD_PATH}")
    clf = joblib.load(HEAD_PATH)
else:
    print("Training calibrated Logistic Regression head on C3D features ...")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=0.5, max_iter=1000,
                                      random_state=42, class_weight='balanced'))
    ])
    clf.fit(X, y)
    joblib.dump(clf, HEAD_PATH)
    print(f"Classifier saved to {HEAD_PATH}")

y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:, 1]

print(f"\nOverall Accuracy: {accuracy_score(y, y_pred)*100:.2f}%")
print(classification_report(y, y_pred, target_names=['ADL (0)', 'Fall (1)']))


# ============================================================
# CELL 7 — Comprehensive 28-Metric Evaluation Engine
# ============================================================

from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    f1_score, fbeta_score,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from IPython.display import display


def compute_all_metrics(y_true, y_pred, y_prob, threshold=0.5):
    """
    Computes every single binary classification metric (28 total).
    Covers: confusion matrix cells, rate metrics, sensitivity/specificity family,
    harmonic scores, correlation coefficients, and probabilistic metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = int(tp + tn + fp + fn)

    acc     = (tp + tn) / total
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    err     = 1.0 - acc
    sens    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv     = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fdr     = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    f_omit  = fn / (fn + tn) if (fn + tn) > 0 else 0.0
    csi     = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

    f1  = f1_score(y_true, y_pred, zero_division=0)
    f2  = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    yj  = sens + spec - 1.0
    mark = prec + npv - 1.0
    dor  = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('nan')

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)
    lloss   = log_loss(y_true, y_prob)

    return {
        "True Positives (TP)":              int(tp),
        "True Negatives (TN)":              int(tn),
        "False Positives (FP)":             int(fp),
        "False Negatives (FN)":             int(fn),
        "Accuracy":                         acc,
        "Balanced Accuracy":                bal_acc,
        "Error Rate":                       err,
        "Sensitivity / Recall (TPR)":       sens,
        "Specificity / Selectivity (TNR)":  spec,
        "Precision / PPV":                  prec,
        "Negative Predictive Value (NPV)":  npv,
        "False Positive Rate (FPR)":        fpr_val,
        "False Negative Rate (FNR)":        fnr_val,
        "False Discovery Rate (FDR)":       fdr,
        "False Omission Rate (FOR)":        f_omit,
        "Threat Score (CSI / Jaccard)":     csi,
        "F1-Score":                         f1,
        "F2-Score (Recall-Weighted)":       f2,
        "F0.5-Score (Precision-Weighted)":  f05,
        "Matthews Correlation (MCC)":       mcc,
        "Cohen's Kappa":                    kap,
        "Youden's J Statistic":             yj,
        "Markedness":                       mark,
        "Diagnostic Odds Ratio (DOR)":      dor,
        "ROC-AUC":                          roc_auc,
        "PR-AUC (Avg Precision)":           pr_auc,
        "Brier Score Loss":                 brier,
        "Log Loss / Cross-Entropy":         lloss,
    }


all_metrics = compute_all_metrics(y, y_pred, y_prob, threshold=0.5)

metrics_df = pd.DataFrame(list(all_metrics.items()), columns=["Metric", "Value"])
metrics_df["Score"] = metrics_df["Value"].apply(
    lambda v: f"{v:d}"   if isinstance(v, (int, np.integer))
    else (f"{v:.6f}"     if isinstance(v, (float, np.floating)) and not np.isnan(v)
          else "NaN (0 denominators)")
)
print("\n" + "=" * 70)
print("            EXHAUSTIVE PERFORMANCE EVALUATION SUMMARY")
print("=" * 70)
display(metrics_df[["Metric", "Score"]])


# ============================================================
# CELL 8 — Dual Confusion Matrix (Raw Counts + Normalized %)
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

labels = ['ADL (Class 0)', 'Fall (Class 1)']
cm = confusion_matrix(y, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            xticklabels=labels, yticklabels=labels)
axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=14, fontweight='bold', pad=12)
axes[0].set_ylabel("Ground Truth Label"); axes[0].set_xlabel("Predicted Label")

# Row-normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            xticklabels=labels, yticklabels=labels)
axes[1].set_title("Confusion Matrix — Row-Normalized (%)", fontsize=14, fontweight='bold', pad=12)
axes[1].set_ylabel("Ground Truth Label"); axes[1].set_xlabel("Predicted Label")

plt.suptitle("FallGuard AI - C3D Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 9 — ROC Curve + Precision-Recall Curve
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ROC curve
fpr_arr, tpr_arr, roc_thresh = roc_curve(y, y_prob)
roc_val = auc(fpr_arr, tpr_arr)
youden_idx  = int(np.argmax(tpr_arr - fpr_arr))
best_thresh = roc_thresh[youden_idx]

axes[0].plot(fpr_arr, tpr_arr, color='#2563eb', lw=3,
             label=f'C3D (AUC = {roc_val:.4f})')
axes[0].plot([0,1],[0,1], color='#94a3b8', lw=1.5, linestyle='--',
             label='Random Chance')
axes[0].scatter(fpr_arr[youden_idx], tpr_arr[youden_idx],
                color='#ef4444', s=120, zorder=5,
                label=f'Youden Cutoff (t={best_thresh:.2f})')
axes[0].set(xlim=[-0.02,1.02], ylim=[-0.02,1.05],
            xlabel='False Positive Rate', ylabel='True Positive Rate')
axes[0].set_title("ROC Curve", fontsize=14, fontweight='bold', pad=12)
axes[0].legend(loc='lower right')

# PR curve
prec_arr, rec_arr, _ = precision_recall_curve(y, y_prob)
pr_auc_val = average_precision_score(y, y_prob)
base = sum(y) / len(y)

axes[1].plot(rec_arr, prec_arr, color='#10b981', lw=3,
             label=f'PR Curve (AP={pr_auc_val:.4f})')
axes[1].axhline(base, color='#94a3b8', lw=1.5, linestyle='--',
                label=f'Prevalence Baseline ({base:.2f})')
axes[1].set(xlim=[-0.02,1.02], ylim=[-0.02,1.05],
            xlabel='Recall (Sensitivity)', ylabel='Precision (PPV)')
axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight='bold', pad=12)
axes[1].legend(loc='lower left')

plt.suptitle("FallGuard AI - Diagnostic Curves", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "roc_pr_curves.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 10 — Calibration Curve + Probability Density Distribution
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Calibration (reliability diagram)
prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=8, strategy='uniform')
axes[0].plot(prob_pred, prob_true, marker='o', lw=2.5, color='#8b5cf6',
             label='C3D Calibration')
axes[0].plot([0,1],[0,1], linestyle='--', color='#94a3b8', label='Perfect')
axes[0].set(xlabel='Mean Predicted Probability',
            ylabel='Empirical Fraction of Positives')
axes[0].set_title(
    f"Reliability Diagram  (Brier = {all_metrics['Brier Score Loss']:.4f})",
    fontsize=14, fontweight='bold', pad=12)
axes[0].legend(loc='upper left')

# Probability density by class
df_prob = pd.DataFrame({
    'Probability': y_prob,
    'Class': ['Fall' if v == 1 else 'ADL' for v in y]
})
sns.kdeplot(data=df_prob, x='Probability', hue='Class', fill=True,
            common_norm=False, palette={'Fall':'#ef4444','ADL':'#22c55e'},
            alpha=0.4, linewidth=2, ax=axes[1])
axes[1].axvline(0.5, color='#475569', linestyle=':', linewidth=2,
                label='Threshold = 0.50')
axes[1].set_title("Predicted Probability Density", fontsize=14, fontweight='bold', pad=12)
axes[1].set(xlabel='Predicted Fall Probability', ylabel='Density')
axes[1].legend(title='True Label')

plt.suptitle("FallGuard AI - Calibration & Confidence", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "calibration_density.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 11 — Threshold Sensitivity Tradeoff (F1/F2/MCC/Sens/Spec vs Threshold)
# ============================================================

thresholds = np.linspace(0.01, 0.99, 150)
f1_list, f2_list, prec_list, rec_list, spec_list, mcc_list = [], [], [], [], [], []

for t in thresholds:
    p_t = (y_prob >= t).astype(int)
    f1_list.append(f1_score(y, p_t, zero_division=0))
    f2_list.append(fbeta_score(y, p_t, beta=2.0, zero_division=0))
    from sklearn.metrics import precision_score, recall_score
    prec_list.append(precision_score(y, p_t, zero_division=0))
    rec_list.append(recall_score(y, p_t, zero_division=0))
    mcc_list.append(matthews_corrcoef(y, p_t))
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, p_t).ravel()
    spec_list.append(tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0)

best_f2_idx = int(np.argmax(f2_list))

plt.figure(figsize=(11, 5.5))
plt.plot(thresholds, f1_list,   label='F1-Score',            color='#3b82f6', lw=2.5)
plt.plot(thresholds, f2_list,   label='F2 (Recall Focus)',   color='#f59e0b', lw=2.5)
plt.plot(thresholds, mcc_list,  label='MCC',                 color='#a855f7', lw=2.5)
plt.plot(thresholds, prec_list, label='Precision',           color='#10b981', lw=2, linestyle='--')
plt.plot(thresholds, rec_list,  label='Sensitivity',         color='#ef4444', lw=2, linestyle='--')
plt.plot(thresholds, spec_list, label='Specificity',         color='#6366f1', lw=2, linestyle=':')
plt.axvline(0.50, color='#64748b', alpha=0.6, lw=1.5, label='Default Threshold=0.50')
plt.axvline(thresholds[best_f2_idx], color='#f59e0b', linestyle='--', alpha=0.8, lw=1.2,
            label=f'Best F2 Threshold={thresholds[best_f2_idx]:.2f}')
plt.title("Performance Metrics vs. Decision Threshold", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Decision Threshold"); plt.ylabel("Metric Score")
plt.ylim([-0.05, 1.08])
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "threshold_tradeoff.png"), bbox_inches='tight', dpi=150)
plt.show()

print(f"Optimal F2 threshold: {thresholds[best_f2_idx]:.3f}  |  "
      f"F2 = {f2_list[best_f2_idx]:.4f}")


# ============================================================
# CELL 12 — Leave-One-Subject-Out Cross-Validation (LOSO-CV)
# ============================================================

from sklearn.linear_model import LogisticRegression as LR

unique_subjects = sorted(np.unique(subjects))
loso_results = []

print("=" * 65)
print("   Leave-One-Subject-Out Cross-Validation (LOSO-CV)")
print("=" * 65)

for test_subj in unique_subjects:
    test_idx  = (subjects == test_subj)
    train_idx = ~test_idx

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Fit scaler only on training fold (no data leakage)
    sc = StandardScaler().fit(X_train)
    X_tr_s = sc.transform(X_train)
    X_te_s = sc.transform(X_test)

    fold_clf = LR(C=0.5, max_iter=1000, random_state=42, class_weight='balanced')
    fold_clf.fit(X_tr_s, y_train)

    te_prob = fold_clf.predict_proba(X_te_s)[:, 1]
    te_pred = fold_clf.predict(X_te_s)

    fm = compute_all_metrics(y_test, te_pred, te_prob, threshold=0.5)
    fm['Subject']    = test_subj
    fm['Test Clips'] = int(len(y_test))
    fm['Train Clips']= int(len(y_train))
    loso_results.append(fm)

    print(f"  [{test_subj}]  N={len(y_test):3d} | "
          f"Acc={fm['Accuracy']*100:5.1f}% | "
          f"F1={fm['F1-Score']:.4f} | "
          f"AUC={fm['ROC-AUC']:.4f} | "
          f"Sens={fm['Sensitivity / Recall (TPR)']:.4f} | "
          f"Spec={fm['Specificity / Selectivity (TNR)']:.4f}")

loso_df = pd.DataFrame(loso_results)
key_cols = ['Subject', 'Train Clips', 'Test Clips',
            'Accuracy', 'Balanced Accuracy',
            'Sensitivity / Recall (TPR)', 'Specificity / Selectivity (TNR)',
            'Precision / PPV', 'F1-Score', 'F2-Score (Recall-Weighted)',
            'Matthews Correlation (MCC)', 'ROC-AUC', 'PR-AUC (Avg Precision)']

print("\nLOSO-CV Full Results:")
display(loso_df[key_cols].round(4).set_index('Subject'))

agg_cols = ['Accuracy', 'F1-Score', 'ROC-AUC',
            'Sensitivity / Recall (TPR)', 'Specificity / Selectivity (TNR)']
print("\nLOSO-CV Aggregate (Mean +/- Std):")
display(loso_df[agg_cols].agg(['mean', 'std']).round(4))


# ============================================================
# CELL 13 — LOSO Per-Subject Bar Chart
# ============================================================

subj_labels = [s.replace('Subject ', 'S') for s in loso_df['Subject']]
x = np.arange(len(subj_labels))
bw = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy & F1
b1 = axes[0].bar(x - bw/2, loso_df['Accuracy'],  bw, label='Accuracy', color='#3b82f6', alpha=0.85)
b2 = axes[0].bar(x + bw/2, loso_df['F1-Score'],  bw, label='F1-Score', color='#10b981', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(subj_labels)
axes[0].set_ylim(0, 1.18); axes[0].set_ylabel("Score"); axes[0].legend()
axes[0].set_title("LOSO-CV: Accuracy & F1 per Subject", fontsize=13, fontweight='bold', pad=10)
for b in [*b1, *b2]:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)

# Sensitivity & Specificity
b3 = axes[1].bar(x - bw/2, loso_df['Sensitivity / Recall (TPR)'],  bw, label='Sensitivity', color='#ef4444', alpha=0.85)
b4 = axes[1].bar(x + bw/2, loso_df['Specificity / Selectivity (TNR)'], bw, label='Specificity', color='#8b5cf6', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(subj_labels)
axes[1].set_ylim(0, 1.18); axes[1].set_ylabel("Score"); axes[1].legend()
axes[1].set_title("LOSO-CV: Sensitivity & Specificity per Subject", fontsize=13, fontweight='bold', pad=10)
for b in [*b3, *b4]:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle("FallGuard AI - LOSO-CV Subject Performance", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "loso_cv_results.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 14 — Fine-Grained: Fall Direction & ADL Activity Breakdown
# ============================================================

df_analysis = df_meta.copy()
df_analysis['predicted']   = y_pred
df_analysis['probability'] = y_prob
df_analysis['correct']     = (df_analysis['label'] == df_analysis['predicted'])

# Fall direction
fall_df = df_analysis[df_analysis['category'] == 'Fall'].copy()
fall_dir_summary = fall_df.groupby('fall_direction').agg(
    Total_Clips         = ('label',       'count'),
    Detected_Falls      = ('predicted',   'sum'),
    Detection_Rate_pct  = ('correct',     lambda c: f"{c.mean()*100:.1f}%"),
    Avg_Fall_Probability= ('probability', lambda p: f"{p.mean()*100:.1f}%"),
    Avg_Duration_s      = ('duration',    lambda d: round(d.mean(), 2))
).reset_index()
print("FALL DETECTION BY DIRECTION:")
display(fall_dir_summary)

# ADL activity
adl_df = df_analysis[df_analysis['category'] == 'ADL'].copy()
adl_summary = adl_df.groupby('adl_action').agg(
    Total_Clips          = ('label',       'count'),
    Correctly_Classified = ('correct',     'sum'),
    False_Alarms         = ('predicted',   lambda p: int((p == 1).sum())),
    Specificity_pct      = ('correct',     lambda c: f"{c.mean()*100:.1f}%"),
    Avg_Fall_Prob_pct    = ('probability', lambda p: f"{p.mean()*100:.1f}%")
).reset_index()
print("\nADL ACTIVITY FALSE ALARM ANALYSIS:")
display(adl_summary)


# ============================================================
# CELL 15 — Action-Level Bar Charts
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fdir_plot = fall_df.groupby('fall_direction')['correct'].mean().reset_index()
fdir_plot.columns = ['Direction', 'Rate']
colors_f = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#22d3ee']
bars1 = axes[0].bar(fdir_plot['Direction'], fdir_plot['Rate'],
                    color=colors_f[:len(fdir_plot)], alpha=0.85, edgecolor='white', lw=1.2)
axes[0].set_ylim(0, 1.2)
axes[0].set_title("Fall Detection Rate by Direction", fontsize=13, fontweight='bold', pad=10)
axes[0].set_ylabel("Detection Rate"); axes[0].set_xlabel("Fall Direction")
for b in bars1:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.03,
                 f'{b.get_height()*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

adl_plot = adl_df.groupby('adl_action').apply(
    lambda g: (g['predicted'] == 1).mean()
).reset_index()
adl_plot.columns = ['Activity', 'FA_Rate']
colors_a = ['#6366f1', '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b']
bars2 = axes[1].bar(adl_plot['Activity'], adl_plot['FA_Rate'],
                    color=colors_a[:len(adl_plot)], alpha=0.85, edgecolor='white', lw=1.2)
axes[1].set_ylim(0, 1.2)
axes[1].set_title("ADL False Alarm Rate by Activity", fontsize=13, fontweight='bold', pad=10)
axes[1].set_ylabel("False Alarm Rate (lower = better)"); axes[1].set_xlabel("ADL Activity")
axes[1].tick_params(axis='x', rotation=20)
for b in bars2:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.03,
                 f'{b.get_height()*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("FallGuard AI - Action-Level Analysis", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "action_breakdown.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 16 — Stratified 5-Fold CV + Plotly Radar Chart
# ============================================================

from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_records = []

print("Stratified 5-Fold Cross-Validation:")
for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    sc = StandardScaler().fit(X_tr)
    fold_clf = LR(C=0.5, max_iter=1000, random_state=42, class_weight='balanced')
    fold_clf.fit(sc.transform(X_tr), y_tr)
    te_prob = fold_clf.predict_proba(sc.transform(X_te))[:, 1]
    te_pred = fold_clf.predict(sc.transform(X_te))
    fm = compute_all_metrics(y_te, te_pred, te_prob)
    fm['Fold'] = f"Fold {fold_idx + 1}"
    fold_records.append(fm)
    print(f"  Fold {fold_idx+1}: "
          f"Acc={fm['Accuracy']*100:.1f}% | "
          f"F1={fm['F1-Score']:.4f} | "
          f"AUC={fm['ROC-AUC']:.4f}")

kfold_df = pd.DataFrame(fold_records)
kcols = ['Fold', 'Accuracy', 'Sensitivity / Recall (TPR)',
         'Specificity / Selectivity (TNR)', 'F1-Score',
         'F2-Score (Recall-Weighted)', 'Matthews Correlation (MCC)', 'ROC-AUC']
display(kfold_df[kcols].round(4).set_index('Fold'))

# Radar chart
radar_metrics = ['Accuracy', 'Sensitivity / Recall (TPR)',
                 'Specificity / Selectivity (TNR)', 'F1-Score',
                 'ROC-AUC', 'PR-AUC (Avg Precision)', 'Matthews Correlation (MCC)']
radar_labels  = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'ROC-AUC', 'PR-AUC', 'MCC']
colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6']

fig_radar = go.Figure()
for i, row in kfold_df.iterrows():
    vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
    lbls = radar_labels + [radar_labels[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=lbls, fill='toself',
        name=row['Fold'], line=dict(color=colors[i], width=2), opacity=0.5
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="5-Fold CV Performance Radar Chart",
    showlegend=True, height=500
)
fig_radar.show()


# ============================================================
# CELL 17 — Subject-Level Metrics Heatmap (LOSO-CV)
# ============================================================

heatmap_metrics = [
    'Accuracy', 'Sensitivity / Recall (TPR)',
    'Specificity / Selectivity (TNR)', 'Precision / PPV',
    'F1-Score', 'F2-Score (Recall-Weighted)',
    'Matthews Correlation (MCC)', 'ROC-AUC',
    "Cohen's Kappa", "Youden's J Statistic"
]
heatmap_labels = [
    'Accuracy', 'Sensitivity', 'Specificity', 'Precision',
    'F1', 'F2', 'MCC', 'ROC-AUC', 'Kappa', "Youden's J"
]
heatmap_data = loso_df[heatmap_metrics].values.astype(float)
subj_short   = [s.replace('Subject ', 'Subj ') for s in loso_df['Subject']]

plt.figure(figsize=(13, 4.5))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, linewidths=0.5, linecolor='white',
            xticklabels=heatmap_labels, yticklabels=subj_short,
            annot_kws={'size': 10})
plt.title("LOSO-CV: Comprehensive Metrics Heatmap (per Subject)",
          fontsize=14, fontweight='bold', pad=14)
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "loso_heatmap.png"), bbox_inches='tight', dpi=150)
plt.show()


# ============================================================
# CELL 18 — Final Executive Summary Report
# ============================================================

loso_mean_acc = loso_df['Accuracy'].mean()
loso_mean_f1  = loso_df['F1-Score'].mean()
loso_mean_auc = loso_df['ROC-AUC'].mean()

print()
print("=" * 70)
print("   FALLGUARD AI -- FINAL EVALUATION REPORT")
print("=" * 70)
print(f"  Dataset      : GMDCSA24 (Alam et al., 2024)")
print(f"  Total Clips  : {len(y)}  (Falls: {sum(y==1)} | ADL: {sum(y==0)})")
print(f"  Backbone     : C3D (Tran et al., Sports-1M, 78.4M params)")
print(f"  Classifier   : Logistic Regression on 4096-d FC7 embeddings")
print("-" * 70)
print(f"  Accuracy     : {all_metrics['Accuracy']*100:.2f}%")
print(f"  Balanced Acc : {all_metrics['Balanced Accuracy']*100:.2f}%")
print(f"  Sensitivity  : {all_metrics['Sensitivity / Recall (TPR)']*100:.2f}%")
print(f"  Specificity  : {all_metrics['Specificity / Selectivity (TNR)']*100:.2f}%")
print(f"  F1-Score     : {all_metrics['F1-Score']:.4f}")
print(f"  F2-Score     : {all_metrics['F2-Score (Recall-Weighted)']:.4f}")
print(f"  ROC-AUC      : {all_metrics['ROC-AUC']:.4f}")
print(f"  PR-AUC       : {all_metrics['PR-AUC (Avg Precision)']:.4f}")
print(f"  MCC          : {all_metrics['Matthews Correlation (MCC)']:.4f}")
print(f"  Kappa        : {all_metrics[\"Cohen's Kappa\"]:.4f}")
print(f"  Brier Score  : {all_metrics['Brier Score Loss']:.4f}")
print(f"  Log-Loss     : {all_metrics['Log Loss / Cross-Entropy']:.4f}")
print("-" * 70)
print(f"  LOSO-CV Acc  : {loso_mean_acc*100:.2f}% (mean across 4 subjects)")
print(f"  LOSO-CV F1   : {loso_mean_f1:.4f}")
print(f"  LOSO-CV AUC  : {loso_mean_auc:.4f}")
print("-" * 70)
print(f"  Saved plots  : {MODELS_DIR}")
print("=" * 70)
