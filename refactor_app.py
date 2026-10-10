import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
# 1-179: Keep
new_lines.extend(lines[:179])

# New Sidebar Navigation
new_lines.extend([
    '\n# Sidebar Configuration\n',
    'st.sidebar.title("🛡️ FallGuard AI")\n',
    'st.sidebar.markdown("### 🧭 Navigation")\n',
    'page = st.sidebar.radio(\n',
    '    "Go to",\n',
    '    ["🎬 Main Dashboard", "🧠 Model Inspection", "📊 Dataset Explorer", "🛡️ Face Anonymization Lab", "📋 Incident & Compliance Audit"],\n',
    '    label_visibility="collapsed"\n',
    ')\n\n',
    'st.sidebar.markdown("---")\n'
])

# Settings Expanders in Sidebar
new_lines.extend([
    'with st.sidebar.expander("⚡ Model Inference Settings", expanded=False):\n',
    '    fall_threshold = st.slider("Fall Alert Sensitivity Threshold", min_value=0.20, max_value=0.85, value=0.50, step=0.05)\n',
    '    stride_frames = st.select_slider("Inference Sliding Window Stride (Frames)", options=[4, 6, 8, 12, 16], value=8)\n\n',
    'with st.sidebar.expander("🛡️ Privacy & Face Blur Settings", expanded=False):\n',
    '    enable_blur = st.toggle("Enable Facial Anonymization", value=True)\n',
    '    blur_type = "gaussian"\n',
    '    blur_strength = 45\n',
    '    face_padding = 0.25\n',
    '    conf_threshold = 0.50\n',
    '    if enable_blur:\n',
    '        blur_type_label = st.selectbox("Blur Technique", ["Gaussian Blur (Smooth)", "Pixelate (Mosaic)", "Solid Blackout Box", "Frosted Glass (Heavy)"], index=0)\n',
    '        type_map = {"Gaussian Blur (Smooth)": "gaussian", "Pixelate (Mosaic)": "pixelate", "Solid Blackout Box": "blackout", "Frosted Glass (Heavy)": "frosted"}\n',
    '        blur_type = type_map[blur_type_label]\n',
    '        blur_strength = st.slider("Blur Intensity", min_value=15, max_value=95, value=45, step=10)\n',
    '        face_padding = st.slider("Boundary Padding (%)", min_value=10, max_value=50, value=25, step=5) / 100.0\n',
    '        conf_threshold = st.slider("Face Detector Confidence", min_value=0.30, max_value=0.85, value=0.50, step=0.05)\n\n'
])

# Global variables needed across pages
new_lines.extend([
    'selected_video_path = st.session_state.get("selected_video_path", None)\n',
    'selected_meta = st.session_state.get("selected_meta", None)\n\n'
])

# Page 1: Main Dashboard
new_lines.extend([
    'if page == "🎬 Main Dashboard":\n',
    '    # Hero Section Banner\n',
    '    st.markdown("""\n',
    '    <div class="hero-container">\n',
    '        <div class="hero-title">🛡️ FallGuard AI: Spatiotemporal Fall Detection & Privacy Shield</div>\n',
    '        <div class="hero-subtitle">\n',
    '            Intelligent Elderly Care Monitoring powered by the <b>C3D 3D-Convolutional Neural Network</b>, trained on the <b>GMDCSA24 Dataset</b> with real-time <b>Facial Anonymization</b>.\n',
    '        </div>\n',
    '        <div>\n',
    '            <span class="badge-pill badge-blue">🧠 C3D Spatiotemporal Backbone (78.4M Params)</span>\n',
    '            <span class="badge-pill badge-purple">📁 GMDCSA24 Video Dataset (160 Videos)</span>\n',
    '            <span class="badge-pill badge-green">🛡️ OpenCV YuNet DNN Face Blur</span>\n',
    '        </div>\n',
    '    </div>\n',
    '    """, unsafe_allow_html=True)\n\n'
])

# Input Source (from lines 184-232), indented
source_logic = lines[184:233]
new_lines.append('    st.markdown("### 📂 Video Source")\n')
for line in source_logic:
    if line.strip() == '' or line.startswith('st.sidebar.markdown'):
        if line.startswith('st.sidebar.markdown("### 📂'):
            continue
    new_line = line.replace('st.sidebar.', 'st.')
    new_lines.append('    ' + new_line)

# Update session state with the selected video
new_lines.extend([
    '    st.session_state["selected_video_path"] = selected_video_path\n',
    '    st.session_state["selected_meta"] = selected_meta\n\n'
])

# Tab 1 contents (lines 309-568)
tab1_logic = lines[309:569]
for line in tab1_logic:
    if 'with tabs[0]:' in line:
        continue
    if "<div>C3D spatiotemporal model detected sudden downward posture collapse" in line:
        new_lines.append(line)
        new_lines.append('                        <div style="margin-top: 8px;"><b>Detected Fall Type:</b> {res.get("mock_fall_type", "Forward/Backward Fall")}</div>\n')
        continue
    new_lines.append(line)


# Page 2: Model Inspection (Tab 3 contents lines 668-747)
new_lines.extend(['\n', 'elif page == "🧠 Model Inspection":\n'])
for line in lines[668:748]:
    if 'with tabs[2]:' in line:
        continue
    new_lines.append(line)

# Page 3: Dataset Explorer (Tab 4 contents lines 751-866)
new_lines.extend(['\n', 'elif page == "📊 Dataset Explorer":\n'])
for line in lines[751:867]:
    if 'with tabs[3]:' in line:
        continue
    new_lines.append(line)

# Page 4: Face Anonymization Lab (Tab 2 contents lines 573-663)
new_lines.extend(['\n', 'elif page == "🛡️ Face Anonymization Lab":\n'])
for line in lines[573:664]:
    if 'with tabs[1]:' in line:
        continue
    new_lines.append(line)

# Page 5: Incident & Compliance Audit (Tab 5 contents lines 870-934)
new_lines.extend(['\n', 'elif page == "📋 Incident & Compliance Audit":\n'])
for line in lines[870:]:
    if 'with tabs[4]:' in line:
        continue
    new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
