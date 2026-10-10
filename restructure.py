import sys
import os

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_dashboard = False
dashboard_code = []

# Collect the dashboard part to replace it
for i, line in enumerate(lines):
    if line.startswith('if page == "🎬 Main Dashboard":'):
        in_dashboard = True
        dashboard_code.append(line)
        continue
    if in_dashboard:
        if line.startswith('elif page =='):
            in_dashboard = False
            new_lines.append('<DASHBOARD_PLACEHOLDER>')
            new_lines.append(line)
        else:
            dashboard_code.append(line)
    else:
        new_lines.append(line)

# Now we rewrite the dashboard_code completely but keeping functionality.
new_dashboard = []
new_dashboard.append('if page == "🎬 Main Dashboard":\n')
new_dashboard.append('    st.markdown("## 🛡️ FallGuard AI Dashboard")\n')

# Video Source in Columns for compactness
new_dashboard.append('    st.markdown("### 📂 Video Source")\n')
new_dashboard.append('    col_src1, col_src2 = st.columns([1, 2.5])\n')
new_dashboard.append('    with col_src1:\n')
new_dashboard.append('        input_mode = st.radio("Select Input Source", ["GMDCSA24 Dataset", "Custom Video Upload"], index=0)\n')
new_dashboard.append('    selected_video_path = None\n')
new_dashboard.append('    selected_meta = None\n')
new_dashboard.append('    with col_src2:\n')
new_dashboard.append('        if input_mode == "GMDCSA24 Dataset":\n')
new_dashboard.append('            c_s1, c_s2, c_s3 = st.columns(3)\n')
new_dashboard.append('            with c_s1: subjects = dataset.get_subjects(); selected_subject = st.selectbox("Subject ID", subjects, index=0)\n')
new_dashboard.append('            with c_s2: categories = dataset.get_categories(); selected_category = st.selectbox("Action Class", categories, index=0)\n')
new_dashboard.append('            with c_s3:\n')
new_dashboard.append('                videos = dataset.filter_videos(selected_subject, selected_category)\n')
new_dashboard.append('                if videos:\n')
new_dashboard.append('                    video_options = [f"{v[\'file_name\']} ({v[\'length_sec\']:.0f}s)" for v in videos]\n')
new_dashboard.append('                    selected_vid_str = st.selectbox("Video Clip", video_options, index=0)\n')
new_dashboard.append('                    selected_file_name = selected_vid_str.split(" ")[0]\n')
new_dashboard.append('                    selected_meta = dataset.get_video_record(selected_subject, selected_category, selected_file_name)\n')
new_dashboard.append('                    if selected_meta: selected_video_path = selected_meta[\'video_path\']\n')
new_dashboard.append('                else:\n')
new_dashboard.append('                    st.warning("No videos found.")\n')
new_dashboard.append('        else:\n')
new_dashboard.append('            uploaded_file = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])\n')
new_dashboard.append('            if uploaded_file is not None:\n')
new_dashboard.append('                import tempfile\n')
new_dashboard.append('                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")\n')
new_dashboard.append('                tfile.write(uploaded_file.read())\n')
new_dashboard.append('                tfile.flush()\n')
new_dashboard.append('                selected_video_path = tfile.name\n')
new_dashboard.append('                selected_meta = {\'subject\': "Custom", \'category\': "User Upload", \'file_name\': uploaded_file.name, \'video_path\': selected_video_path, \'length_sec\': 0.0, \'time_of_recording\': "Unknown", \'attire\': "Unknown", \'description\': "User uploaded video", \'classes_raw\': "Custom", \'intervals\': [], \'fall_count\': 0}\n')

new_dashboard.append('    st.session_state["selected_video_path"] = selected_video_path\n')
new_dashboard.append('    st.session_state["selected_meta"] = selected_meta\n')

new_dashboard.append('    if selected_video_path and os.path.exists(selected_video_path):\n')
new_dashboard.append('        st.markdown("<br>", unsafe_allow_html=True)\n')
new_dashboard.append('        run_btn = st.button("🚀 Process & Analyze Video", type="primary", width="stretch")\n')
new_dashboard.append('        res_key = f"result_{selected_meta[\'subject\']}_{selected_meta[\'file_name\']}_{enable_blur}_{blur_type}_{fall_threshold}_{stride_frames}"\n')

new_dashboard.append('        if run_btn:\n')
new_dashboard.append('            progress_bar = st.progress(0, text="Initializing spatiotemporal pipeline...")\n')
new_dashboard.append('            import tempfile, time\n')
new_dashboard.append('            out_dir = os.path.join(tempfile.gettempdir(), "fallguard_output")\n')
new_dashboard.append('            os.makedirs(out_dir, exist_ok=True)\n')
new_dashboard.append('            output_vid_path = os.path.join(out_dir, f"proc_{int(time.time())}_{selected_meta[\'file_name\']}.mp4")\n')
new_dashboard.append('            def on_progress(curr, total, prob):\n')
new_dashboard.append('                frac = min(1.0, curr / max(1, total))\n')
new_dashboard.append('                status = "🚨 Fall Risk High" if prob >= fall_threshold else "🟢 Normal Motion"\n')
new_dashboard.append('                progress_bar.progress(frac, text=f"Processing Frame {curr}/{total} ({frac*100:.0f}%) — {status} ({prob*100:.1f}%)")\n')
new_dashboard.append('            with st.spinner("Extracting 3D clips & applying privacy shield..."):\n')
new_dashboard.append('                res = video_processor.process_video(input_video_path=selected_video_path, output_video_path=output_vid_path, enable_blur=enable_blur, blur_type=blur_type, blur_strength=blur_strength, face_padding=face_padding, conf_threshold=conf_threshold, fall_threshold=fall_threshold, stride_frames=stride_frames, progress_callback=on_progress)\n')
new_dashboard.append('                progress_bar.progress(1.0, text="Processing Complete!")\n')
new_dashboard.append('                st.session_state[res_key] = res\n')
new_dashboard.append('                st.session_state["latest_res"] = res\n')
new_dashboard.append('                st.session_state["latest_meta"] = selected_meta\n')

new_dashboard.append('        current_res = st.session_state.get(res_key, st.session_state.get("latest_res", None))\n')
new_dashboard.append('        if current_res is not None and os.path.exists(current_res.get("output_video_path", "")):\n')
new_dashboard.append('            res = current_res\n')
new_dashboard.append('            st.markdown("---")\n')

new_dashboard.append('            vid_col1, vid_col2 = st.columns(2)\n')
new_dashboard.append('            with vid_col1:\n')
new_dashboard.append('                st.markdown("#### 🛡️ Processed Video (Anonymized + HUD)")\n')
new_dashboard.append('                if os.path.exists(res[\'output_video_path\']):\n')
new_dashboard.append('                    with open(res[\'output_video_path\'], \'rb\') as f: st.video(f.read())\n')
new_dashboard.append('                    st.download_button("⬇️ Download Anonymized MP4 Video", data=open(res[\'output_video_path\'], \'rb\').read(), file_name=f"fallguard_{selected_meta[\'file_name\']}", mime="video/mp4", width="stretch")\n')
new_dashboard.append('            with vid_col2:\n')
new_dashboard.append('                st.markdown("#### 📹 Original Video")\n')
new_dashboard.append('                if os.path.exists(selected_video_path):\n')
new_dashboard.append('                    with open(selected_video_path, \'rb\') as f: st.video(f.read())\n')

new_dashboard.append('            st.markdown("<br>", unsafe_allow_html=True)\n')
new_dashboard.append('            if res[\'is_fall_detected\']:\n')
new_dashboard.append('                fall_time_msg = f"at <b>{res[\'fall_detected_at_time\']:.2f}s</b>" if res[\'fall_detected_at_time\'] else ""\n')
new_dashboard.append('                st.markdown(f"""\n')
new_dashboard.append('                <div class="alert-fall">\n')
new_dashboard.append('                    <span style="font-size: 1.8rem;">🚨</span>\n')
new_dashboard.append('                    <div>\n')
new_dashboard.append('                        <div style="font-size: 1.2rem; font-weight: 700;">EMERGENCY ALERT: HUMAN FALL DETECTED!</div>\n')
new_dashboard.append('                        <div>C3D spatiotemporal model detected sudden downward posture collapse {fall_time_msg} with <b>{res[\'max_fall_probability\']*100:.1f}%</b> peak confidence.</div>\n')
new_dashboard.append('                    </div>\n')
new_dashboard.append('                </div>\n')
new_dashboard.append('                """, unsafe_allow_html=True)\n')
new_dashboard.append('            else:\n')
new_dashboard.append('                st.markdown(f"""\n')
new_dashboard.append('                <div class="alert-safe">\n')
new_dashboard.append('                    <span style="font-size: 1.8rem;">🟢</span>\n')
new_dashboard.append('                    <div>\n')
new_dashboard.append('                        <div style="font-size: 1.2rem; font-weight: 700;">STATUS: NORMAL ACTIVITY OF DAILY LIVING (ADL)</div>\n')
new_dashboard.append('                        <div>No abnormal collapse detected. Maximum fall risk remained at <b>{res[\'max_fall_probability\']*100:.1f}%</b> (below {fall_threshold*100:.0f}% threshold).</div>\n')
new_dashboard.append('                    </div>\n')
new_dashboard.append('                </div>\n')
new_dashboard.append('                """, unsafe_allow_html=True)\n')

new_dashboard.append('            # Primary metrics\n')
new_dashboard.append('            m_col1, m_col2, m_col3, m_col4 = st.columns(4)\n')
new_dashboard.append('            with m_col1:\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Fall Verdict</div>\n')
new_dashboard.append('                    <div class="metric-value" style="color: {\'#ef4444\' if res[\'is_fall_detected\'] else \'#22c55e\'};">{\'FALL\' if res[\'is_fall_detected\'] else \'NORMAL\'}</div>\n')
new_dashboard.append('                    <div class="metric-sub">Threshold: {fall_threshold*100:.0f}%</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')
new_dashboard.append('            with m_col2:\n')
new_dashboard.append('                onset_str = f"{res[\'fall_detected_at_time\']:.2f}s" if res[\'fall_detected_at_time\'] else "N/A"\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Detection Time</div>\n')
new_dashboard.append('                    <div class="metric-value">{onset_str}</div>\n')
new_dashboard.append('                    <div class="metric-sub">Total: {res[\'duration_sec\']:.1f}s</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')
new_dashboard.append('            with m_col3:\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Peak Confidence</div>\n')
new_dashboard.append('                    <div class="metric-value">{res[\'max_fall_probability\']*100:.1f}%</div>\n')
new_dashboard.append('                    <div class="metric-sub">C3D Model</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')
new_dashboard.append('            with m_col4:\n')
new_dashboard.append('                fall_type = res.get(\'mock_fall_type\', \'None\') if res[\'is_fall_detected\'] else \'N/A\'\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Detected Fall Type</div>\n')
new_dashboard.append('                    <div class="metric-value" style="font-size: 1.2rem;">{fall_type}</div>\n')
new_dashboard.append('                    <div class="metric-sub">Skeleton AI Prediction</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')

new_dashboard.append('            # Secondary metrics\n')
new_dashboard.append('            st.markdown("<br>", unsafe_allow_html=True)\n')
new_dashboard.append('            s_col1, s_col2, s_col3 = st.columns(3)\n')
new_dashboard.append('            with s_col1:\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Faces Blurred</div>\n')
new_dashboard.append('                    <div class="metric-value" style="color: #D9795E;">{res[\'total_faces_blurred\']}</div>\n')
new_dashboard.append('                    <div class="metric-sub">Mode: {blur_type.title()}</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')
new_dashboard.append('            with s_col2:\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Processing Speed</div>\n')
new_dashboard.append('                    <div class="metric-value">{res[\'total_frames\']/max(0.1, res[\'processing_time_sec\']):.1f} FPS</div>\n')
new_dashboard.append('                    <div class="metric-sub">Elapsed: {res[\'processing_time_sec\']:.1f}s</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')
new_dashboard.append('            with s_col3:\n')
new_dashboard.append('                is_gt_fall = selected_meta[\'category\'] == \'Fall\'\n')
new_dashboard.append('                is_pred_fall = res[\'is_fall_detected\']\n')
new_dashboard.append('                match = (is_gt_fall == is_pred_fall)\n')
new_dashboard.append('                match_label = "CORRECT (TP)" if (is_gt_fall and is_pred_fall) else ("CORRECT (TN)" if (not is_gt_fall and not is_pred_fall) else "DISCREPANCY")\n')
new_dashboard.append('                st.markdown(f"""<div class="metric-card">\n')
new_dashboard.append('                    <div class="metric-label">Ground-Truth Evaluation</div>\n')
new_dashboard.append('                    <div class="metric-value" style="color: {\'#34d399\' if match else \'#f59e0b\'}; font-size: 1.2rem;">{match_label}</div>\n')
new_dashboard.append('                    <div class="metric-sub">Actual Label: {selected_meta[\'category\']}</div>\n')
new_dashboard.append('                </div>""", unsafe_allow_html=True)\n')

# Find the start of the chart logic inside dashboard_code
chart_start = -1
for i, line in enumerate(dashboard_code):
    if "st.markdown(\"### 📈 Spatiotemporal Fall Confidence Timeline\")" in line:
        chart_start = i
        break
if chart_start != -1:
    new_dashboard.append('            st.markdown("<br>", unsafe_allow_html=True)\n')
    for line in dashboard_code[chart_start:]:
        if 'else:' in line and 'st.info("👆 Click **\'🚀 Process & Analyze Video\'**' in dashboard_code[dashboard_code.index(line)+1]:
            break
        new_dashboard.append(line)

new_dashboard.append('            with st.expander("ℹ️ Video Details & Ground-Truth Metadata"):\n')
new_dashboard.append('                c_det1, c_det2 = st.columns(2)\n')
new_dashboard.append('                with c_det1:\n')
new_dashboard.append('                    st.markdown(f"**Clip:** `{selected_meta[\'file_name\']}` | **Subject:** `{selected_meta[\'subject\']}`")\n')
new_dashboard.append('                    st.markdown(f"**Lighting:** `{selected_meta[\'time_of_recording\']}` | **Attire:** `{selected_meta[\'attire\']}`")\n')
new_dashboard.append('                with c_det2:\n')
new_dashboard.append('                    st.markdown(f"**Scenario:** *{selected_meta[\'description\']}*")\n')
new_dashboard.append('                    if selected_meta.get(\'intervals\'):\n')
new_dashboard.append('                        int_str = " | ".join([f"`{i[\'action\']} ({i[\'start\']}-{i[\'end\']}s)`" for i in selected_meta[\'intervals\']])\n')
new_dashboard.append('                        st.markdown(f"**Ground Truth Segments:** {int_str}")\n')

new_dashboard.append('        else:\n')
new_dashboard.append('            st.info("👆 Click **\'🚀 Process & Analyze Video\'** above to run C3D spatiotemporal fall detection and facial anonymization.")\n')
new_dashboard.append('            st.markdown("#### 📹 Video Preview")\n')
new_dashboard.append('            if os.path.exists(selected_video_path):\n')
new_dashboard.append('                with open(selected_video_path, \'rb\') as f: st.video(f.read())\n')
new_dashboard.append('    else:\n')
new_dashboard.append('        st.warning("Please select or upload a video file to begin.")\n')


# Move the big hero banner to sidebar expander
hero_idx = -1
for i, line in enumerate(new_lines):
    if "st.sidebar.markdown(\"### 🧭 Navigation\")" in line:
        hero_idx = i
        break
if hero_idx != -1:
    hero_code = [
        'with st.sidebar.expander("ℹ️ About FallGuard AI", expanded=False):\n',
        '    st.markdown("""\n',
        '        <div style="font-size: 0.9rem; color: #77736B;">\n',
        '            Intelligent Elderly Care Monitoring powered by the <b>C3D 3D-Convolutional Neural Network</b>, trained on the <b>GMDCSA24 Dataset</b> with real-time <b>Facial Anonymization</b>.\n',
        '        </div>\n',
        '        <div style="margin-top: 10px;">\n',
        '            <div class="badge-pill badge-blue" style="margin-bottom:4px; display:block;">🧠 C3D Spatiotemporal (78.4M)</div>\n',
        '            <div class="badge-pill badge-purple" style="margin-bottom:4px; display:block;">📁 GMDCSA24 Dataset</div>\n',
        '            <div class="badge-pill badge-green" style="display:block;">🛡️ OpenCV YuNet Face Blur</div>\n',
        '        </div>\n',
        '    """, unsafe_allow_html=True)\n'
    ]
    new_lines = new_lines[:hero_idx] + hero_code + new_lines[hero_idx:]

final_lines = []
for line in new_lines:
    if line.strip() == '<DASHBOARD_PLACEHOLDER>':
        final_lines.extend(new_dashboard)
    else:
        final_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(final_lines)
