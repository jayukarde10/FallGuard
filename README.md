# 🛡️ FallGuard AI: Spatiotemporal Fall Detection & Privacy Shield

FallGuard AI is a video-based human fall detection and privacy preservation platform designed for elderly care, assisted living, and healthcare monitoring.

It integrates:
1. **The C3D Deep Learning Backbone** (`weights.h5`, 78.4M parameters) for 3D spatiotemporal video action understanding.
2. **The GMDCSA24 Dataset** (*A Dataset for Human Fall Detection in Videos*, Alam et al., 2024) across 4 subjects and 160 video clips.
3. **Automated Facial Anonymization** using OpenCV YuNet Deep Neural Network to obscure facial identities with multiple privacy filters (Gaussian, Mosaic, Blackout, Frosted).
4. **Interactive Streamlit Web Dashboard** with real-time HUD telemetry, confidence timelines, and compliance auditing.

---

## 🌟 Key Features

### 1. Spatiotemporal Fall Detection (C3D + Calibrated Head)
- Loads pre-trained spatiotemporal convolution weights directly from [`weights.h5`](file:///home/soham/project/weights.h5).
- Evaluates 16-frame sliding windows ($112 \times 112$ RGB) with configurable temporal strides.
- Calibrated binary classification head trained across GMDCSA24 video clips to discriminate between Falls (Forward, Backward, Sideways) and Activities of Daily Living (ADL: Sitting, Walking, Sleeping, Reading).

### 2. Multi-Filter Facial Anonymization
- Detects faces at non-standard camera angles, tilts, and fallen postures using OpenCV YuNet DNN (`models/face_detection_yunet.onnx`).
- Supported privacy blurring algorithms:
  - **Gaussian Blur**: Smooth natural blur with adjustable kernel strength.
  - **Pixelate (Mosaic)**: Retro pixelation grid effect.
  - **Solid Blackout Box**: High-security sensor bars with detection labels.
  - **Frosted Glass**: Bilateral edge-preserving filtering combined with heavy blurring.
- Configurable boundary margin padding (10% to 50%) to ensure hair, ears, and chin are completely anonymized.

### 3. GMDCSA24 Dataset Integration
- Built-in dataset catalog with full metadata integration across Subjects 1–4.
- Parses ground truth activity intervals (e.g., `Sitting [0 to 3.4s]`, `Falling [3.4 to 6s]`).
- Visualizes actual ground truth action intervals alongside the model's predictive curve.

### 4. Interactive Streamlit UI
- **Tab 1: Video Analysis & Inference**: Side-by-side comparison of original and anonymized video, HUD telemetry, emergency fall alert banners, and interactive Plotly confidence timelines.
- **Tab 2: Face Anonymization Lab**: Single-frame inspection workbench with 4-way visual comparison of all blur techniques and face bounding box coordinates.
- **Tab 3: C3D Model Architecture**: Layer-by-layer inspection of 78.4M parameters, tensor dimensions, and interactive weight distribution histograms.
- **Tab 4: GMDCSA24 Dataset Explorer**: Statistical distribution charts and filterable catalog across all 160 dataset videos.
- **Tab 5: Incident & Compliance Audit**: Exportable GDPR/HIPAA verification certificates with timestamps, peak risk scores, and anonymization counts.

---

## 🚀 Running the Application

To start the Streamlit dashboard:

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Streamlit
streamlit run app.py
```

The application is accessible in your web browser at:
👉 **`http://localhost:8501`**

---

## 📁 Repository Structure

```
├── app.py                  # Main Streamlit UI application
├── c3d_model.py            # PyTorch C3D architecture & weights.h5 loader
├── face_blur.py            # OpenCV YuNet face detection & privacy filters
├── dataset_manager.py      # GMDCSA24 dataset catalog & CSV annotation parser
├── video_pipeline.py       # Sliding-window inference, HUD overlay & H.264 video rendering
├── train_head.py           # Calibration script for fall classification head
├── weights.h5              # Pretrained C3D model weights (78.4M parameters)
├── models/
│   ├── face_detection_yunet.onnx   # OpenCV YuNet deep face detector
│   └── fall_head.joblib            # Calibrated fall classifier head
└── ekramalam-GMDCSA24-.../         # GMDCSA24 video dataset (Subjects 1–4, ADL & Fall)
```
# aimlproject
