"""
Validate FallSubtypeClassifier against GMDCSA24 videos with known ground-truth.
Tests videos with known fall directions: SW (Sideways), FW (Forward), BW (Backward).
"""
import os, sys, time
import numpy as np
import cv2
import torch

from video_pipeline import VideoProcessor, FallSubtypeClassifier
from c3d_model import load_c3d_model
from face_blur import FaceAnonymizer
from dataset_manager import GMDCSADataset

print("=" * 70)
print("FallSubtypeClassifier Validation")
print("=" * 70)

# Load dataset
ds = GMDCSADataset('ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76')
falls = ds.filter_videos(category='Fall')

# Map ground-truth codes to expected class names
GT_MAP = {
    'SW': 'Sideways Fall',
    'FW': 'Forward Fall', 
    'BW': 'Backward Fall',
}

# Test the classifier standalone on a few videos
classifier = FallSubtypeClassifier()

test_cases = falls[:16]  # First 16 fall videos
results = []

for meta in test_cases:
    # Extract ground truth from classes_raw
    raw = meta['classes_raw']
    gt_code = None
    for code in GT_MAP:
        if f'({code})' in raw:
            gt_code = code
            break
    
    if not gt_code:
        continue
    
    gt_class = GT_MAP[gt_code]
    
    # Read 16 frames around the fall onset
    cap = cv2.VideoCapture(meta['video_path'])
    if not cap.isOpened():
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Find the fall start time from intervals
    fall_start = 0
    for interval in meta.get('intervals', []):
        if interval.get('is_fall'):
            fall_start = interval['start']
            break
    
    # Seek to just before the fall
    start_frame = max(0, int(fall_start * fps) - 4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(16):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112))
        frames.append(resized)
    cap.release()
    
    if len(frames) < 6:
        continue
    
    # Classify
    predicted_class, confidence = classifier.classify(frames)
    
    match = "OK" if predicted_class == gt_class else "XX"
    results.append({
        'file': meta['file_name'],
        'subject': meta['subject'],
        'gt': gt_class,
        'predicted': predicted_class,
        'confidence': confidence,
        'match': predicted_class == gt_class
    })
    
    print(f"{match} {meta['subject']}/{meta['file_name']}: GT={gt_class:<15} Pred={predicted_class:<22} Conf={confidence:.0%}")

print()
print("=" * 70)
total = len(results)
correct = sum(1 for r in results if r['match'])
print(f"Accuracy: {correct}/{total} ({correct/max(total,1)*100:.0f}%)")
print()

# Show per-class breakdown
for gt_name in GT_MAP.values():
    gt_results = [r for r in results if r['gt'] == gt_name]
    gt_correct = sum(1 for r in gt_results if r['match'])
    print(f"  {gt_name}: {gt_correct}/{len(gt_results)} correct")
    
print("=" * 70)
