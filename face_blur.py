import os
import cv2
import numpy as np

class FaceAnonymizer:
    """
    State-of-the-art face detector and anonymizer using OpenCV 5.0 YuNet DNN.
    Detects faces at various angles (standing, falling, lying down) and provides
    multiple blurring & privacy masking algorithms.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.yunet_path = os.path.join(models_dir, "face_detection_yunet.onnx")
        self.has_yunet = os.path.exists(self.yunet_path)
        self._detector_cache = {}

    def _get_detector(self, width, height, conf_threshold=0.55, nms_threshold=0.3):
        key = (width, height, round(conf_threshold, 2))
        if key not in self._detector_cache:
            detector = cv2.FaceDetectorYN.create(
                model=self.yunet_path,
                config="",
                input_size=(width, height),
                score_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                top_k=5000
            )
            self._detector_cache[key] = detector
        return self._detector_cache[key]

    def detect_faces(self, frame, conf_threshold=0.50):
        """
        Detects faces in frame.
        Returns: list of (x, y, w, h, confidence)
        """
        if not self.has_yunet:
            return []

        h, w = frame.shape[:2]
        faces = []

        try:
            detector = self._get_detector(w, h, conf_threshold)
            _, detected = detector.detect(frame)
            if detected is not None:
                for d in detected:
                    box = d[0:4].astype(int)
                    conf = float(d[-1])
                    if conf >= conf_threshold:
                        faces.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf))
        except Exception as e:
            # If resolution changed or error
            pass

        return faces

    def anonymize_frame(
        self,
        frame,
        conf_threshold=0.50,
        blur_type="gaussian",
        blur_strength=45,
        padding=0.25
    ):
        """
        Detects and anonymizes faces in the frame.
        Returns:
            processed_frame (np.ndarray): Frame with anonymized faces
            padded_boxes (list): List of (x, y, w, h, conf)
        """
        out_frame = frame.copy()
        faces = self.detect_faces(out_frame, conf_threshold=conf_threshold)
        h, w = frame.shape[:2]
        padded_boxes = []

        for (fx, fy, fw, fh, conf) in faces:
            # Apply boundary padding to guarantee full facial anonymization
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(w, fx + fw + pad_w)
            y2 = min(h, fy + fh + pad_h)

            roi_w = x2 - x1
            roi_h = y2 - y1
            if roi_w <= 3 or roi_h <= 3:
                continue

            padded_boxes.append((x1, y1, roi_w, roi_h, conf))
            face_roi = out_frame[y1:y2, x1:x2]

            if blur_type == "gaussian":
                ksize = max(15, blur_strength if blur_strength % 2 == 1 else blur_strength + 1)
                blurred = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
                out_frame[y1:y2, x1:x2] = blurred

            elif blur_type == "pixelate":
                # Mosaic / Pixelation effect
                step = max(3, int(roi_w / max(4, blur_strength // 3)))
                small_w = max(1, roi_w // step)
                small_h = max(1, roi_h // step)
                small = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                out_frame[y1:y2, x1:x2] = pixelated

            elif blur_type == "blackout":
                # High security solid sensor box
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (18, 18, 20), -1)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (80, 200, 120), 1)
                cv2.putText(
                    out_frame, "ANONYMIZED", (x1 + 6, y1 + max(16, roi_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA
                )

            elif blur_type == "frosted":
                # Bilateral filter + Gaussian smoothing for frosted glass effect
                ksize = max(25, blur_strength if blur_strength % 2 == 1 else blur_strength + 1)
                smooth = cv2.bilateralFilter(face_roi, 9, 75, 75)
                blurred = cv2.GaussianBlur(smooth, (ksize, ksize), 0)
                out_frame[y1:y2, x1:x2] = blurred

        return out_frame, padded_boxes
