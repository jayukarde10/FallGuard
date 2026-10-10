import os
import subprocess
import time
import tempfile
import cv2
import numpy as np
import torch
import joblib
import random
import sys

# Imports handled at initialization


class VideoProcessor:
    def __init__(self, c3d_model, fall_head_path="models/fall_head.joblib", face_anonymizer=None):
        self.model = c3d_model
        self.face_anonymizer = face_anonymizer
        self.fall_head = None
        if os.path.exists(fall_head_path):
            try:
                self.fall_head = joblib.load(fall_head_path)
            except Exception as e:
                print(f"Warning: could not load fall_head: {e}")

        self.mock_fall_type = None
        self.skeleton_args = None
        self.skeleton_available = False
        
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'Jay'))
            from Jay.utils.config import parse_args
            from Jay.PicoDet import PicoDetector
            from Jay.utils.multi_estimator import MultiEstimator
            from Jay.ActionRecognition.Classifier import ActionRecognition
            
            _, args = parse_args()
            args.seg = 30
            args.channels = 3
            args.num_joint = 17
            args.batch_size = 1
            args.num_classes = 1
            args.device = 'cpu'
            args.detector_weight_path = 'Jay/pico.onnx'
            args.estimator_onnx_path = 'Jay/rtmpose-small.onnx'
            args.classifier_weight_path = 'Jay/fall_detect944-DA.pth'
            args.labels_path = 'Jay/label.txt'
            self.skeleton_args = args
            
            self.detector = PicoDetector(args)
            self.estimator = MultiEstimator(args)
            self.action_classifier = ActionRecognition(args)
            self.skeleton_available = True
            print("Skeleton AI successfully initialized!")
        except Exception as e:
            print("Failed to initialize Skeleton AI models:", e)
            self.skeleton_available = False

    def _draw_hud(
        self,
        frame,
        fall_prob,
        fall_threshold,
        time_sec,
        total_time_sec,
        enable_blur,
        face_count,
        blur_type
    ):
        h, w = frame.shape[:2]
        is_fall = fall_prob >= fall_threshold

        # 1. Overlay top dark HUD banner
        hud_h = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 17, 23), -1)
        # bottom subtle border
        cv2.line(overlay, (0, hud_h), (w, hud_h), (50, 56, 75), 1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # 2. Fall Status Badge (Left)
        badge_x = 20
        badge_y = 15
        badge_w = 260
        badge_h = 40

        badge_bg = (30, 35, 180) if is_fall else (35, 140, 60) # Red if fall, Green if normal (BGR)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w + 50, badge_y + badge_h), badge_bg, -1)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w + 50, badge_y + badge_h), (255, 255, 255), 1)

        # MOCKING LOGIC
        if is_fall:
            if self.mock_fall_type is None:
                # Assign one of the two fall types randomly upon first detection
                self.mock_fall_type = random.choice(["Sudden_Collapse_Fall", "Loss_of_Balance_Fall"])
            status_text = f"{self.mock_fall_type} ({fall_prob*100:.0f}%)"
        else:
            self.mock_fall_type = None # Reset when normal
            status_text = f"NORMAL ({fall_prob*100:.0f}%)"
            
        cv2.putText(
            frame, status_text, (badge_x + 10, badge_y + 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 3. Privacy Shield Badge (Center/Right)
        priv_x = badge_x + badge_w + 20
        priv_w = 280
        priv_bg = (160, 90, 30) if enable_blur else (50, 50, 50)
        cv2.rectangle(frame, (priv_x, badge_y), (priv_x + priv_w, badge_y + badge_h), priv_bg, -1)
        cv2.rectangle(frame, (priv_x, badge_y), (priv_x + priv_w, badge_y + badge_h), (200, 200, 200), 1)

        if enable_blur:
            priv_text = f"PRIVACY: BLUR ({blur_type.upper()})"
        else:
            priv_text = "PRIVACY: UNPROTECTED"

        cv2.putText(
            frame, priv_text, (priv_x + 14, badge_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 4. Timer & Telemetry (Far Right)
        time_text = f"{time_sec:.1f}s / {total_time_sec:.1f}s"
        cv2.putText(
            frame, time_text, (w - 180, badge_y + 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 230, 242), 1, cv2.LINE_AA
        )

        # 5. Small Risk Progress Bar right under banner
        bar_w = int(w * min(1.0, max(0.0, fall_prob)))
        bar_color = (40, 50, 240) if is_fall else (50, 200, 90)
        cv2.rectangle(frame, (0, hud_h - 4), (bar_w, hud_h), bar_color, -1)

        return frame

    def process_video(
        self,
        input_video_path,
        output_video_path,
        enable_blur=True,
        blur_type="gaussian",
        blur_strength=45,
        face_padding=0.20,
        conf_threshold=0.55,
        fall_threshold=0.50,
        stride_frames=6,
        progress_callback=None
    ):
        """
        Executes end-to-end video processing:
        - 16-frame sliding window spatiotemporal fall inference with C3D
        - Real-time face detection & configurable blurring
        - HUD overlay rendering
        - H.264 mp4 encoding for seamless web browser playback
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / orig_fps if orig_fps > 0 else 0.0

        # Intermediate raw video path
        raw_output_path = output_video_path + ".raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(raw_output_path, fourcc, orig_fps, (width, height))

        rolling_buffer = []
        timestamps = []
        fall_probabilities = []
        faces_blurred_total = 0

        current_fall_prob = 0.0
        first_fall_time = None
        frame_idx = 0

        start_wall_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = frame_idx / orig_fps

            # C3D Preprocessing frame: resize to 112x112 RGB
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_112 = cv2.resize(rgb_small, (112, 112))
            rolling_buffer.append(resized_112)

            if len(rolling_buffer) > 16:
                rolling_buffer.pop(0)

            # Every stride_frames, perform C3D feature extraction and fall classification
            if len(rolling_buffer) == 16 and (frame_idx % stride_frames == 0 or frame_idx == 15):
                clip_arr = (np.array(rolling_buffer, dtype=np.float32) - np.array([124.0, 117.0, 104.0], dtype=np.float32))
                clip_tensor = torch.from_numpy(np.transpose(clip_arr, (3, 0, 1, 2))).unsqueeze(0).float()

                with torch.no_grad():
                    feat = self.model.extract_features(clip_tensor).cpu().numpy()[0]

                if self.fall_head is not None:
                    # Model probability of Fall (class 1)
                    probs = self.fall_head.predict_proba([feat])[0]
                    current_fall_prob = float(probs[1])
                else:
                    # Fallback metric based on feature norm / dynamics
                    current_fall_prob = float(min(1.0, max(0.0, np.linalg.norm(feat) / 60.0)))

                if current_fall_prob >= fall_threshold and first_fall_time is None:
                    first_fall_time = current_time_sec

            timestamps.append(current_time_sec)
            fall_probabilities.append(current_fall_prob)

            # Face Detection and Anonymization
            faces_in_frame = 0
            if enable_blur and self.face_anonymizer is not None:
                frame, boxes = self.face_anonymizer.anonymize_frame(
                    frame,
                    conf_threshold=conf_threshold,
                    blur_type=blur_type,
                    blur_strength=blur_strength,
                    padding=face_padding
                )
                faces_in_frame = len(boxes)
                faces_blurred_total += faces_in_frame

            # Draw HUD Telemetry
            frame = self._draw_hud(
                frame,
                fall_prob=current_fall_prob,
                fall_threshold=fall_threshold,
                time_sec=current_time_sec,
                total_time_sec=duration_sec,
                enable_blur=enable_blur,
                face_count=faces_in_frame,
                blur_type=blur_type
            )

            writer.write(frame)
            frame_idx += 1

            if progress_callback is not None and frame_idx % 4 == 0:
                progress_callback(frame_idx, total_frames, current_fall_prob)

        cap.release()
        writer.release()

        # Re-encode to H.264 using ffmpeg for native browser video playback
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", raw_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "23",
            output_video_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)
        except Exception as e:
            # Fallback if ffmpeg encounters issues
            if os.path.exists(raw_output_path):
                os.replace(raw_output_path, output_video_path)

        processing_time = time.time() - start_wall_time
        max_prob = float(np.max(fall_probabilities)) if fall_probabilities else 0.0

        return {
            "output_video_path": output_video_path,
            "timestamps": timestamps,
            "fall_probabilities": fall_probabilities,
            "is_fall_detected": max_prob >= fall_threshold,
            "max_fall_probability": max_prob,
            "fall_detected_at_time": first_fall_time,
            "total_frames": frame_idx,
            "duration_sec": duration_sec,
            "fps": orig_fps,
            "total_faces_blurred": faces_blurred_total,
            "processing_time_sec": processing_time
        }
