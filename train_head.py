import os
import glob
import re
import cv2
import torch
import numpy as np
import h5py
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

class C3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool1(h)
        h = F.relu(self.conv2(h))
        h = self.pool2(h)
        h = F.relu(self.conv3a(h))
        h = F.relu(self.conv3b(h))
        h = self.pool3(h)
        h = F.relu(self.conv4a(h))
        h = F.relu(self.conv4b(h))
        h = self.pool4(h)
        h = F.relu(self.conv5a(h))
        h = F.relu(self.conv5b(h))
        h = F.pad(h, (0, 1, 0, 1, 0, 0))
        h = self.pool5(h)
        h = h.permute(0, 2, 3, 4, 1).contiguous().view(h.size(0), -1)
        f6 = F.relu(self.fc6(h))
        f7 = F.relu(self.fc7(f6))
        return f7

def load_c3d(h5_path='weights.h5'):
    model = C3D()
    with h5py.File(h5_path, 'r') as f:
        for layer_name in ['conv1', 'conv2', 'conv3a', 'conv3b', 'conv4a', 'conv4b', 'conv5a', 'conv5b']:
            w = f[layer_name][f'{layer_name}/kernel:0'][:]
            b = f[layer_name][f'{layer_name}/bias:0'][:]
            getattr(model, layer_name).weight.data.copy_(torch.from_numpy(w).permute(4, 3, 0, 1, 2))
            getattr(model, layer_name).bias.data.copy_(torch.from_numpy(b))
        for layer_name in ['fc6', 'fc7']:
            w = f[layer_name][f'{layer_name}/kernel:0'][:]
            b = f[layer_name][f'{layer_name}/bias:0'][:]
            getattr(model, layer_name).weight.data.copy_(torch.from_numpy(w).t())
            getattr(model, layer_name).bias.data.copy_(torch.from_numpy(b))
    model.eval()
    return model

def parse_intervals(class_str):
    """
    Parses strings like: 'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]'
    Returns list of dicts: [{'action': 'Falling (SW)', 'start': 3.4, 'end': 6.0, 'is_fall': True}]
    """
    results = []
    if not isinstance(class_str, str):
        return results
    parts = [p.strip() for p in class_str.split(';') if p.strip()]
    for p in parts:
        m = re.match(r'^(.*?)\s*\[\s*([\d\.]+)\s+to\s+([\d\.]+)\s*\]', p)
        if m:
            act = m.group(1).strip()
            s = float(m.group(2))
            e = float(m.group(3))
            is_fall = 'fall' in act.lower()
            results.append({'action': act, 'start': s, 'end': e, 'is_fall': is_fall})
    return results

def get_clip_at_sec(video_path, sec, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = max(0, min(int(sec * fps), int(total_frames - num_frames)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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
    arr = (np.array(frames, dtype=np.float32) - np.array([124.0, 117.0, 104.0], dtype=np.float32))
    return torch.from_numpy(np.transpose(arr, (3, 0, 1, 2))).unsqueeze(0).float()

def build_dataset_and_train():
    print("Loading C3D backbone from weights.h5...")
    model = load_c3d('weights.h5')
    dataset_root = 'ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76'

    X = []
    y = []
    meta = []

    for subj in ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']:
        subj_dir = os.path.join(dataset_root, subj)
        # Process Fall
        fall_csv = os.path.join(subj_dir, 'Fall.csv')
        if os.path.exists(fall_csv):
            df_fall = pd.read_csv(fall_csv)
            for _, row in df_fall.iterrows():
                fname = str(row['File Name']).strip()
                vpath = os.path.join(subj_dir, 'Fall', fname)
                if not os.path.exists(vpath):
                    continue
                intervals = parse_intervals(row.get(' Classes', ''))
                # Find fall interval
                fall_ints = [i for i in intervals if i['is_fall']]
                if fall_ints:
                    target_sec = (fall_ints[0]['start'] + fall_ints[0]['end']) / 2.0
                else:
                    target_sec = 2.5
                clip = get_clip_at_sec(vpath, target_sec)
                if clip is not None:
                    with torch.no_grad():
                        feat = model(clip).numpy()[0]
                    X.append(feat)
                    y.append(1)
                    meta.append((subj, 'Fall', fname))

        # Process ADL
        adl_csv = os.path.join(subj_dir, 'ADL.csv')
        if os.path.exists(adl_csv):
            df_adl = pd.read_csv(adl_csv)
            for _, row in df_adl.iterrows():
                fname = str(row['File Name']).strip()
                vpath = os.path.join(subj_dir, 'ADL', fname)
                if not os.path.exists(vpath):
                    continue
                clip = get_clip_at_sec(vpath, 2.0)
                if clip is not None:
                    with torch.no_grad():
                        feat = model(clip).numpy()[0]
                    X.append(feat)
                    y.append(0)
                    meta.append((subj, 'ADL', fname))

    X = np.array(X)
    y = np.array(y)
    print(f"Total extracted clips: {len(y)}, Fall={sum(y==1)}, ADL={sum(y==0)}")

    # Train calibrated classifier
    clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    clf.fit(X, y)
    train_acc = accuracy_score(y, clf.predict(X))
    print(f"Model trained! Accuracy: {train_acc*100:.2f}%")
    print(classification_report(y, clf.predict(X), target_names=['ADL', 'Fall']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/fall_head.joblib')
    print("Saved classification head to models/fall_head.joblib")

if __name__ == '__main__':
    build_dataset_and_train()
