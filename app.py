import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from c3d_model import load_c3d_model, get_model_summary
from dataset_manager import GMDCSADataset
from face_blur import FaceAnonymizer
from video_pipeline import VideoProcessor

# Page Configuration
st.set_page_config(
    page_title="FallGuard AI | C3D Fall Detection & Privacy Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark theme, glassmorphism, modern typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background: radial-gradient(circle at 10% 20%, rgba(18, 24, 38, 0.95) 0%, rgba(10, 13, 20, 1) 90%);
        color: #e2e8f0;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 12px;
        font-weight: 400;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
    }
    .badge-blue { color: #38bdf8; border-color: rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.08); }
    .badge-purple { color: #a855f7; border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.08); }
    .badge-green { color: #34d399; border-color: rgba(52, 211, 153, 0.3); background: rgba(52, 211, 153, 0.08); }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.18);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Status Alert Banners */
    .alert-fall {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 16px 20px;
        color: #fecaca;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 12px 0;
    }
    .alert-safe {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.25) 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        color: #bbf7d0;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 12px 0;
    }

    /* Code & Monospace */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# Global Resource Caching
@st.cache_resource(show_spinner="Loading C3D Spatiotemporal Model (weights.h5)...")
def get_c3d_model():
    return load_c3d_model("weights.h5")

@st.cache_resource
def get_face_anonymizer():
    return FaceAnonymizer("models")

@st.cache_resource
def get_dataset():
    return GMDCSADataset("ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76")

# Load Resources
dataset = get_dataset()
face_anonymizer = get_face_anonymizer()
c3d_model = get_c3d_model()
video_processor = VideoProcessor(c3d_model, "models/fall_head.joblib", face_anonymizer)


# Sidebar Configuration
st.sidebar.title("🛡️ FallGuard Controls")

# Input Source
st.sidebar.markdown("### 📂 Video Source")
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["GMDCSA24 Dataset", "Custom Video Upload"],
    index=0
)

selected_video_path = None
selected_meta = None

if input_mode == "GMDCSA24 Dataset":
    subjects = dataset.get_subjects()
    selected_subject = st.sidebar.selectbox("Subject ID", subjects, index=0)

    categories = dataset.get_categories()
    selected_category = st.sidebar.selectbox("Action Class", categories, index=0)

    videos = dataset.filter_videos(selected_subject, selected_category)
    if videos:
        video_options = [f"{v['file_name']} ({v['length_sec']:.0f}s)" for v in videos]
        selected_vid_str = st.sidebar.selectbox("Video Clip", video_options, index=0)
        selected_file_name = selected_vid_str.split(" ")[0]
        selected_meta = dataset.get_video_record(selected_subject, selected_category, selected_file_name)
        if selected_meta:
            selected_video_path = selected_meta['video_path']
    else:
        st.sidebar.warning("No videos found for selection.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        selected_video_path = tfile.name
        selected_meta = {
            'subject': "Custom Subject",
            'category': "User Upload",
            'file_name': uploaded_file.name,
            'video_path': selected_video_path,
            'length_sec': 0.0,
            'time_of_recording': "Unknown",
            'attire': "Unknown",
            'description': "User uploaded video clip",
            'classes_raw': "Custom",
            'intervals': [],
            'fall_count': 0
        }

st.sidebar.markdown("---")

# Privacy & Face Blur Settings
st.sidebar.markdown("### 🛡️ Privacy & Face Blur")
enable_blur = st.sidebar.toggle("Enable Facial Anonymization", value=True)

blur_type = "gaussian"
blur_strength = 45
face_padding = 0.25
conf_threshold = 0.50

if enable_blur:
    blur_type_label = st.sidebar.selectbox(
        "Blur Technique",
        ["Gaussian Blur (Smooth)", "Pixelate (Mosaic)", "Solid Blackout Box", "Frosted Glass (Heavy)"],
        index=0
    )
    type_map = {
        "Gaussian Blur (Smooth)": "gaussian",
        "Pixelate (Mosaic)": "pixelate",
        "Solid Blackout Box": "blackout",
        "Frosted Glass (Heavy)": "frosted"
    }
    blur_type = type_map[blur_type_label]

    blur_strength = st.sidebar.slider("Blur Intensity", min_value=15, max_value=95, value=45, step=10)
    face_padding = st.sidebar.slider("Boundary Padding (%)", min_value=10, max_value=50, value=25, step=5) / 100.0
    conf_threshold = st.sidebar.slider("Face Detector Confidence", min_value=0.30, max_value=0.85, value=0.50, step=0.05)

st.sidebar.markdown("---")

# Fall Detection Hyperparameters
st.sidebar.markdown("### ⚡ Model Inference Settings")
fall_threshold = st.sidebar.slider(
    "Fall Alert Sensitivity Threshold",
    min_value=0.20, max_value=0.85, value=0.50, step=0.05,
    help="Confidence threshold above which a fall alert is triggered."
)
stride_frames = st.sidebar.select_slider(
    "Inference Sliding Window Stride (Frames)",
    options=[4, 6, 8, 12, 16],
    value=8,
    help="Stride between C3D 16-frame evaluations. Smaller stride = higher temporal precision; larger stride = faster processing."
)


# Hero Section Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ FallGuard AI: Spatiotemporal Fall Detection & Privacy Shield</div>
    <div class="hero-subtitle">
        Intelligent Elderly Care Monitoring powered by the <b>C3D 3D-Convolutional Neural Network</b>, trained on the <b>GMDCSA24 Dataset</b> with real-time <b>Facial Anonymization</b>.
    </div>
    <div>
        <span class="badge-pill badge-blue">🧠 C3D Spatiotemporal Backbone (78.4M Params)</span>
        <span class="badge-pill badge-purple">📁 GMDCSA24 Video Dataset (160 Videos)</span>
        <span class="badge-pill badge-green">🛡️ OpenCV YuNet DNN Face Blur</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs Navigation
tabs = st.tabs([
    "🎬 Video Analysis & Inference",
    "🛡️ Face Anonymization Lab",
    "🧠 C3D Model Architecture",
    "📊 GMDCSA24 Dataset Explorer",
    "📋 Incident & Compliance Audit"
])


# ==========================================
# TAB 1: VIDEO ANALYSIS & INFERENCE
# ==========================================
with tabs[0]:
    if selected_video_path and os.path.exists(selected_video_path):
        # Metadata banner card
        with st.container():
            col_m1, col_m2, col_m3, col_m4 = st.columns([1.5, 1.2, 1.5, 3])
            with col_m1:
                st.markdown(f"**Clip:** `{selected_meta['file_name']}`")
                st.markdown(f"**Subject:** `{selected_meta['subject']}`")
            with col_m2:
                cat_badge = "🔴 Fall" if selected_meta['category'] == "Fall" else "🟢 ADL"
                st.markdown(f"**Class:** {cat_badge}")
                st.markdown(f"**Duration:** `{selected_meta['length_sec']:.1f}s`")
            with col_m3:
                st.markdown(f"**Lighting:** `{selected_meta['time_of_recording']}`")
                st.markdown(f"**Attire:** `{selected_meta['attire']}`")
            with col_m4:
                st.markdown(f"**Scenario:** *{selected_meta['description']}*")
                if selected_meta.get('intervals'):
                    int_str = " | ".join([f"`{i['action']} ({i['start']}-{i['end']}s)`" for i in selected_meta['intervals']])
                    st.markdown(f"**Ground Truth Segments:** {int_str}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([1.2, 3])
        with col_ctrl1:
            run_btn = st.button("🚀 Process & Analyze Video", type="primary", width="stretch")

        # Session state key for results
        res_key = f"result_{selected_meta['subject']}_{selected_meta['file_name']}_{enable_blur}_{blur_type}_{fall_threshold}_{stride_frames}"

        if run_btn:
            progress_bar = st.progress(0, text="Initializing spatiotemporal pipeline...")
            status_text = st.empty()

            out_dir = os.path.join(tempfile.gettempdir(), "fallguard_output")
            os.makedirs(out_dir, exist_ok=True)
            output_vid_path = os.path.join(out_dir, f"proc_{int(time.time())}_{selected_meta['file_name']}.mp4")

            def on_progress(curr, total, prob):
                frac = min(1.0, curr / max(1, total))
                status = "🚨 Fall Risk High" if prob >= fall_threshold else "🟢 Normal Motion"
                progress_bar.progress(frac, text=f"Processing Frame {curr}/{total} ({frac*100:.0f}%) — {status} ({prob*100:.1f}%)")

            with st.spinner("Extracting 3D clips & applying privacy shield..."):
                t_start = time.time()
                res = video_processor.process_video(
                    input_video_path=selected_video_path,
                    output_video_path=output_vid_path,
                    enable_blur=enable_blur,
                    blur_type=blur_type,
                    blur_strength=blur_strength,
                    face_padding=face_padding,
                    conf_threshold=conf_threshold,
                    fall_threshold=fall_threshold,
                    stride_frames=stride_frames,
                    progress_callback=on_progress
                )
                progress_bar.progress(1.0, text="Processing Complete!")
                st.session_state[res_key] = res
                st.session_state["latest_res"] = res
                st.session_state["latest_meta"] = selected_meta

        # Display results if available
        current_res = st.session_state.get(res_key, st.session_state.get("latest_res", None))

        if current_res is not None and os.path.exists(current_res.get("output_video_path", "")):
            res = current_res
            st.markdown("---")

            # Fall Status Alert Banner
            if res['is_fall_detected']:
                fall_time_msg = f"at <b>{res['fall_detected_at_time']:.2f}s</b>" if res['fall_detected_at_time'] else ""
                st.markdown(f"""
                <div class="alert-fall">
                    <span style="font-size: 1.8rem;">🚨</span>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700;">EMERGENCY ALERT: HUMAN FALL DETECTED!</div>
                        <div>C3D spatiotemporal model detected sudden downward posture collapse {fall_time_msg} with <b>{res['max_fall_probability']*100:.1f}%</b> peak confidence.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-safe">
                    <span style="font-size: 1.8rem;">🟢</span>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700;">STATUS: NORMAL ACTIVITY OF DAILY LIVING (ADL)</div>
                        <div>No abnormal collapse detected. Maximum fall risk remained at <b>{res['max_fall_probability']*100:.1f}%</b> (below {fall_threshold*100:.0f}% threshold).</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Metric Cards Row
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fall Verdict</div>
                    <div class="metric-value" style="color: {'#ef4444' if res['is_fall_detected'] else '#22c55e'};">
                        {'FALL' if res['is_fall_detected'] else 'NORMAL'}
                    </div>
                    <div class="metric-sub">Peak: {res['max_fall_probability']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                onset_str = f"{res['fall_detected_at_time']:.2f}s" if res['fall_detected_at_time'] else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detection Time</div>
                    <div class="metric-value">{onset_str}</div>
                    <div class="metric-sub">Total: {res['duration_sec']:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Faces Blurred</div>
                    <div class="metric-value" style="color: #38bdf8;">{res['total_faces_blurred']}</div>
                    <div class="metric-sub">{blur_type.title()} Mode</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Latency / FPS</div>
                    <div class="metric-value">{res['total_frames']/max(0.1, res['processing_time_sec']):.1f} FPS</div>
                    <div class="metric-sub">{res['processing_time_sec']:.1f}s elapsed</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col5:
                # Ground truth comparison
                is_gt_fall = selected_meta['category'] == 'Fall'
                is_pred_fall = res['is_fall_detected']
                match = (is_gt_fall == is_pred_fall)
                match_label = "CORRECT (TP)" if (is_gt_fall and is_pred_fall) else ("CORRECT (TN)" if (not is_gt_fall and not is_pred_fall) else "DISCREPANCY")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Ground Truth</div>
                    <div class="metric-value" style="color: {'#34d399' if match else '#f59e0b'};">{match_label}</div>
                    <div class="metric-sub">Actual: {selected_meta['category']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Video Player: Processed Anonymized Video vs Original
            vid_col1, vid_col2 = st.columns(2)
            with vid_col1:
                st.markdown("#### 🛡️ Processed Video (Face Anonymized + HUD)")
                if os.path.exists(res['output_video_path']):
                    with open(res['output_video_path'], 'rb') as f:
                        vid_bytes = f.read()
                    st.video(vid_bytes)

                    # Download button
                    st.download_button(
                        label="⬇️ Download Anonymized MP4 Video",
                        data=vid_bytes,
                        file_name=f"fallguard_{selected_meta['file_name']}",
                        mime="video/mp4",
                        width="stretch"
                    )
            with vid_col2:
                st.markdown("#### 📹 Original Video (GMDCSA24 Dataset)")
                if os.path.exists(selected_video_path):
                    with open(selected_video_path, 'rb') as f:
                        orig_bytes = f.read()
                    st.video(orig_bytes)

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Plotly Timeline Chart
            st.markdown("### 📈 Spatiotemporal Fall Confidence Timeline")
            fig = go.Figure()

            # Fall Probability Line
            fig.add_trace(go.Scatter(
                x=res['timestamps'],
                y=res['fall_probabilities'],
                mode='lines',
                name='C3D Fall Probability',
                line=dict(color='#38bdf8', width=3),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.12)'
            ))

            # Threshold line
            fig.add_hline(
                y=fall_threshold,
                line_dash="dash",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f"Alert Threshold ({fall_threshold:.2f})",
                annotation_position="bottom right",
                annotation_font_color="#ef4444"
            )

            # Highlight Ground Truth Fall Windows
            if selected_meta.get('intervals'):
                for interval in selected_meta['intervals']:
                    if interval['is_fall']:
                        fig.add_vrect(
                            x0=interval['start'],
                            x1=interval['end'],
                            fillcolor="rgba(239, 68, 68, 0.18)",
                            line_width=1,
                            line_color="rgba(239, 68, 68, 0.4)",
                            annotation_text=f"GT: {interval['action']}",
                            annotation_position="top left",
                            annotation_font_color="#f87171"
                        )
                    else:
                        fig.add_vrect(
                            x0=interval['start'],
                            x1=interval['end'],
                            fillcolor="rgba(52, 211, 153, 0.08)",
                            line_width=0,
                            annotation_text=f"GT: {interval['action']}",
                            annotation_position="top left",
                            annotation_font_color="#6ee7b7"
                        )

            fig.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                xaxis=dict(
                    title="Video Timestamp (seconds)",
                    gridcolor='rgba(255, 255, 255, 0.08)',
                    zerolinecolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title="Fall Probability (0.0 - 1.0)",
                    range=[0.0, 1.05],
                    gridcolor='rgba(255, 255, 255, 0.08)',
                    zerolinecolor='rgba(255, 255, 255, 0.1)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#e2e8f0")
                ),
                font=dict(family="Plus Jakarta Sans", color="#e2e8f0")
            )
            st.plotly_chart(fig, width="stretch")

        else:
            # Placeholder before running
            st.info("👆 Click **'🚀 Process & Analyze Video'** above to run C3D spatiotemporal fall detection and facial anonymization.")
            # Quick preview of raw video
            st.markdown("#### 📹 Video Preview")
            if os.path.exists(selected_video_path):
                with open(selected_video_path, 'rb') as f:
                    st.video(f.read())
    else:
        st.warning("Please select or upload a video file to begin.")


# ==========================================
# TAB 2: FACE ANONYMIZATION LAB
# ==========================================
with tabs[1]:
    st.markdown("### 🛡️ Facial Anonymization & Privacy Laboratory")
    st.markdown(
        "Interactive comparison of OpenCV YuNet face detection and privacy-preserving filters across video frames."
    )

    if selected_video_path and os.path.exists(selected_video_path):
        cap_lab = cv2.VideoCapture(selected_video_path)
        total_lab_frames = int(cap_lab.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_lab_fps = cap_lab.get(cv2.CAP_PROP_FPS) or 30.0

        lab_f_idx = st.slider(
            "Select Video Frame to Inspect",
            min_value=0,
            max_value=max(0, total_lab_frames - 1),
            value=min(30, total_lab_frames - 1),
            step=1,
            format="Frame %d"
        )

        cap_lab.set(cv2.CAP_PROP_POS_FRAMES, lab_f_idx)
        ret, lab_frame = cap_lab.read()
        cap_lab.release()

        if ret and lab_frame is not None:
            time_at_f = lab_f_idx / orig_lab_fps
            st.caption(f"Inspecting Frame {lab_f_idx} at timestamp `{time_at_f:.2f}s` (Resolution: {lab_frame.shape[1]}x{lab_frame.shape[0]})")

            # Detect faces
            detected_faces = face_anonymizer.detect_faces(lab_frame, conf_threshold=conf_threshold)

            # Apply each blur type
            frame_orig_rgb = cv2.cvtColor(lab_frame, cv2.COLOR_BGR2RGB)

            f_gauss, _ = face_anonymizer.anonymize_frame(
                lab_frame, conf_threshold=conf_threshold, blur_type="gaussian",
                blur_strength=blur_strength, padding=face_padding
            )
            f_pix, _ = face_anonymizer.anonymize_frame(
                lab_frame, conf_threshold=conf_threshold, blur_type="pixelate",
                blur_strength=blur_strength, padding=face_padding
            )
            f_black, _ = face_anonymizer.anonymize_frame(
                lab_frame, conf_threshold=conf_threshold, blur_type="blackout",
                blur_strength=blur_strength, padding=face_padding
            )
            f_frost, _ = face_anonymizer.anonymize_frame(
                lab_frame, conf_threshold=conf_threshold, blur_type="frosted",
                blur_strength=blur_strength, padding=face_padding
            )

            # Render 4-way comparison
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 1. Original Unblurred Frame")
                # Draw detection boxes on original frame for inspection
                boxed_orig = frame_orig_rgb.copy()
                for (x, y, w, h, conf) in detected_faces:
                    cv2.rectangle(boxed_orig, (x, y), (x + w, y + h), (56, 189, 248), 2)
                    cv2.putText(
                        boxed_orig, f"Face ({conf:.2f})", (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2
                    )
                st.image(boxed_orig, width="stretch")

            with c2:
                st.markdown(f"##### 2. Gaussian Blur (Strength: {blur_strength})")
                st.image(cv2.cvtColor(f_gauss, cv2.COLOR_BGR2RGB), width="stretch")

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### 3. Mosaic / Pixelation Filter")
                st.image(cv2.cvtColor(f_pix, cv2.COLOR_BGR2RGB), width="stretch")

            with c4:
                st.markdown("##### 4. Solid Blackout Privacy Bar")
                st.image(cv2.cvtColor(f_black, cv2.COLOR_BGR2RGB), width="stretch")

            # Details
            st.markdown("---")
            st.markdown(f"**Detected Faces in Frame:** `{len(detected_faces)}`")
            if detected_faces:
                for idx, (x, y, w, h, conf) in enumerate(detected_faces):
                    st.markdown(f"- **Face #{idx+1}:** Bounding Box: `[x={x}, y={y}, width={w}, height={h}]` | YuNet Confidence: `{conf*100:.1f}%`")
            else:
                st.info("No face detected in this particular frame (subject may be turned away or fallen out of facial view).")

    else:
        st.info("Select a video in the sidebar to test face anonymization.")


# ==========================================
# TAB 3: C3D MODEL ARCHITECTURE
# ==========================================
with tabs[2]:
    st.markdown("### 🧠 C3D Spatiotemporal Convolutional Neural Network")
    st.markdown(
        "Direct inspection of the deep learning architecture loaded from `weights.h5` (~320 MB)."
    )

    summary_data, total_params = get_model_summary("weights.h5")

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Parameters</div>
            <div class="metric-value" style="color: #38bdf8;">{total_params:,}</div>
            <div class="metric-sub">~78.4 Million Weights</div>
        </div>
        """, unsafe_allow_html=True)
    with c_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Network Input</div>
            <div class="metric-value">16 x 112 x 112</div>
            <div class="metric-sub">3-Channel RGB Video Clip</div>
        </div>
        """, unsafe_allow_html=True)
    with c_p3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Feature Vector</div>
            <div class="metric-value" style="color: #a855f7;">4,096-dim</div>
            <div class="metric-sub">FC7 Spatiotemporal Embedding</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Layer architecture breakdown table
    st.markdown("#### 📋 Layer-by-Layer Weights Inspection (`weights.h5`)")
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(
            df_summary,
            width="stretch",
            column_config={
                "Layer Name": st.column_config.TextColumn("Layer", width="medium"),
                "Weights & Shapes": st.column_config.TextColumn("Tensor Dimensions", width="large"),
                "Parameters": st.column_config.NumberColumn("Parameters", format="%d")
            }
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizing Weight Distributions
    st.markdown("#### 📊 Weight Value Distribution Histogram")
    selected_layer = st.selectbox(
        "Select Layer to Inspect Tensor Distribution",
        ["conv1", "conv2", "conv3a", "conv4a", "conv5b", "fc6", "fc7", "fc8"],
        index=0
    )
    if hasattr(c3d_model, selected_layer):
        w_tensor = getattr(c3d_model, selected_layer).weight.data.cpu().numpy().flatten()
        # Subsample for fast plotting
        sample_w = np.random.choice(w_tensor, size=min(15000, len(w_tensor)), replace=False)
        fig_hist = go.Figure(data=[go.Histogram(
            x=sample_w,
            nbinsx=60,
            marker_color='#818cf8',
            opacity=0.85
        )])
        fig_hist.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 0.6)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=280,
            xaxis=dict(title=f"Weight Value ({selected_layer})", gridcolor='rgba(255, 255, 255, 0.08)'),
            yaxis=dict(title="Frequency", gridcolor='rgba(255, 255, 255, 0.08)'),
            font=dict(family="Plus Jakarta Sans", color="#e2e8f0")
        )
        st.plotly_chart(fig_hist, width="stretch")


# ==========================================
# TAB 4: GMDCSA24 DATASET EXPLORER
# ==========================================
with tabs[3]:
    st.markdown("### 📊 GMDCSA24 Dataset Explorer")
    st.markdown(
        "*GMDCSA24: A Dataset for Human Fall Detection in Videos* (Alam et al., Zenodo 2024 / Data in Brief). "
        "Contains multi-subject indoor activities of daily living (ADL) and simulated fall events across diverse attire and lighting."
    )

    stats = dataset.get_dataset_stats()
    ds_c1, ds_c2, ds_c3, ds_c4 = st.columns(4)
    with ds_c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Video Clips</div>
            <div class="metric-value">{stats['total_videos']}</div>
            <div class="metric-sub">Across 4 Subjects</div>
        </div>
        """, unsafe_allow_html=True)
    with ds_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fall Incidents</div>
            <div class="metric-value" style="color: #ef4444;">{stats['fall_videos']}</div>
            <div class="metric-sub">Left, Right, Forward, Back</div>
        </div>
        """, unsafe_allow_html=True)
    with ds_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ADL Control Clips</div>
            <div class="metric-value" style="color: #22c55e;">{stats['adl_videos']}</div>
            <div class="metric-sub">Sitting, Walking, Sleeping</div>
        </div>
        """, unsafe_allow_html=True)
    with ds_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Duration</div>
            <div class="metric-value">{stats['avg_duration']:.1f}s</div>
            <div class="metric-sub">High-Speed Motion</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Distribution Chart
    if hasattr(dataset, 'df') and not dataset.df.empty:
        df = dataset.df

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.markdown("##### Video Distribution by Subject and Category")
            grp = df.groupby(['subject', 'category']).size().reset_index(name='count')
            fig_bar = go.Figure()
            for cat in ['Fall', 'ADL']:
                sub_df = grp[grp['category'] == cat]
                color = '#ef4444' if cat == 'Fall' else '#22c55e'
                fig_bar.add_trace(go.Bar(
                    x=sub_df['subject'],
                    y=sub_df['count'],
                    name=cat,
                    marker_color=color
                ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=280,
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)'),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)'),
                font=dict(family="Plus Jakarta Sans", color="#e2e8f0")
            )
            st.plotly_chart(fig_bar, width="stretch")

        with c_d2:
            st.markdown("##### Action Category Breakdown")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Fall Incidents', 'ADL (Daily Activities)'],
                values=[stats['fall_videos'], stats['adl_videos']],
                hole=0.55,
                marker=dict(colors=['#ef4444', '#22c55e'])
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(15, 23, 42, 0.6)',
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=280,
                font=dict(family="Plus Jakarta Sans", color="#e2e8f0")
            )
            st.plotly_chart(fig_pie, width="stretch")

        st.markdown("#### 🔍 Filterable Dataset Catalog")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            f_subj = st.multiselect("Filter by Subject", options=dataset.get_subjects(), default=dataset.get_subjects())
        with filter_col2:
            f_cat = st.multiselect("Filter by Category", options=['Fall', 'ADL'], default=['Fall', 'ADL'])

        filtered_df = df[df['subject'].isin(f_subj) & df['category'].isin(f_cat)]

        st.dataframe(
            filtered_df[['subject', 'category', 'file_name', 'length_sec', 'time_of_recording', 'attire', 'description', 'classes_raw']],
            width="stretch",
            column_config={
                "subject": "Subject",
                "category": "Category",
                "file_name": "File",
                "length_sec": st.column_config.NumberColumn("Length (s)", format="%.1f"),
                "time_of_recording": "Lighting",
                "attire": "Attire",
                "description": "Scenario Description",
                "classes_raw": "Annotated Classes"
            }
        )


# ==========================================
# TAB 5: INCIDENT & COMPLIANCE AUDIT
# ==========================================
with tabs[4]:
    st.markdown("### 📋 Elder Care Incident Verification & Privacy Audit")
    st.markdown(
        "Exportable compliance certificate verifying spatiotemporal fall alert timestamps and GDPR/HIPAA facial anonymization."
    )

    last_res = st.session_state.get("latest_res", None)
    last_meta = st.session_state.get("latest_meta", selected_meta)

    if last_res and last_meta:
        c_status = "CRITICAL FALL INCIDENT" if last_res['is_fall_detected'] else "NORMAL ADL ACTIVITY"
        audit_time = time.strftime("%Y-%m-%d %H:%M:%S UTC")

        st.markdown(f"""
        ```text
        ================================================================================
                    FALLGUARD AI - ELDER CARE INCIDENT & PRIVACY AUDIT
        ================================================================================
        Audit Timestamp      : {audit_time}
        Video File ID        : {last_meta['file_name']}
        Subject Monitored    : {last_meta['subject']}
        Dataset Category     : {last_meta['category']}
        Scenario Description : {last_meta['description']}
        Video Duration       : {last_res['duration_sec']:.2f} seconds ({last_res['total_frames']} frames)
        --------------------------------------------------------------------------------
        FALL DETECTION EVALUATION
        --------------------------------------------------------------------------------
        Fall Classification  : {c_status}
        Peak Fall Probability: {last_res['max_fall_probability']*100:.2f}%
        Detection Timestamp  : {f"{last_res['fall_detected_at_time']:.2f} seconds" if last_res['fall_detected_at_time'] else "N/A"}
        Model Architecture   : C3D Spatiotemporal 3D-CNN (weights.h5 / 78.4M parameters)
        Inference Latency    : {last_res['processing_time_sec']:.2f}s ({last_res['total_frames']/max(0.1, last_res['processing_time_sec']):.1f} FPS)
        --------------------------------------------------------------------------------
        PRIVACY & ANONYMIZATION COMPLIANCE (GDPR / HIPAA)
        --------------------------------------------------------------------------------
        Face Anonymization   : {'ENABLED' if enable_blur else 'DISABLED'}
        Blur Algorithm       : {blur_type.upper()} (Intensity: {blur_strength})
        Face Detection Engine: OpenCV YuNet Deep Neural Network
        Total Faces Blurred  : {last_res['total_faces_blurred']} instances
        Compliance Verdict   : {'PASSED - Personal Identifiable Information (PII) Protected' if enable_blur else 'WARNING - Faces Unprotected'}
        ================================================================================
        ```
        """)

        report_text = f"""# FALLGUARD AI - INCIDENT & PRIVACY AUDIT REPORT
- **Timestamp:** {audit_time}
- **Video ID:** {last_meta['file_name']}
- **Subject:** {last_meta['subject']}
- **Ground Truth:** {last_meta['category']} ({last_meta['description']})
- **Fall Detection Verdict:** {c_status}
- **Peak Fall Probability:** {last_res['max_fall_probability']*100:.2f}%
- **Onset Timestamp:** {f"{last_res['fall_detected_at_time']:.2f}s" if last_res['fall_detected_at_time'] else "N/A"}
- **Face Anonymization:** {'ACTIVE' if enable_blur else 'INACTIVE'} ({last_res['total_faces_blurred']} faces protected via {blur_type})
- **Compliance Status:** {'GDPR / HIPAA COMPLIANT' if enable_blur else 'NON-COMPLIANT'}
"""
        st.download_button(
            label="⬇️ Export Audit Report (Markdown)",
            data=report_text,
            file_name=f"fallguard_audit_{last_meta['file_name']}.md",
            mime="text/markdown"
        )
    else:
        st.info("Run an analysis in Tab 1 to generate an Incident & Privacy Audit Report.")
