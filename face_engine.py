"""
EduFace AI - Robust Face Detection & Recognition Engine
Powered by OpenCV Deep Neural Network (DNN):
- YuNet: Ultra-fast, lightweight face detector with 5 facial landmarks
- SFace: High-accuracy deep face recognition model (128D embeddings)
No dlib or C++ compilation required.
"""

import os
import urllib.request
import base64
import time
import numpy as np
import cv2

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# SFace official cosine similarity threshold: 0.363
# Default system threshold: 0.42 for high precision and minimal false positives
DEFAULT_SIMILARITY_THRESHOLD = 0.42


def ensure_models_exist():
    """Verify that ONNX models exist on disk, or download them automatically."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"Downloading YuNet face detection model to {YUNET_MODEL_PATH}...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL_PATH)
    if not os.path.exists(SFACE_MODEL_PATH):
        print(f"Downloading SFace face recognition model to {SFACE_MODEL_PATH}...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_MODEL_PATH)


def compute_eye_aspect_ratio(image: np.ndarray, eye_pt: np.ndarray, eye_dist: float) -> float:
    """
    Computes Eye Aspect Ratio (EAR) from eye landmark center and inter-ocular distance.
    Returns value ~0.50-0.95 for open eyes, and < 0.28 for closed eyes.
    """
    if image is None or image.size == 0 or eye_dist <= 0:
        return 0.5

    ex, ey = int(eye_pt[0]), int(eye_pt[1])
    hw = max(4, int(eye_dist * 0.16))
    hh = max(4, int(eye_dist * 0.12))
    h, w = image.shape[:2]

    x1, y1 = max(0, ex - hw), max(0, ey - hh)
    x2, y2 = min(w, ex + hw), min(h, ey + hh)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.5

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh_val = np.percentile(gray, 25)
    mask = gray < thresh_val

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(largest)
    return float(bh / max(1, bw))


class BlinkDetector:
    """
    Real-time anti-spoofing and liveness verification tracker.
    Tracks eye state over consecutive frames to confirm a genuine physiological blink.
    Static photographs never blink (constant eye opening ratio), so their status
    stays strictly at 'blink to confirm' and attendance is never marked.
    """
    def __init__(self, ear_closed_thresh=0.28, ear_open_thresh=0.38, verification_duration=15.0):
        self.ear_closed_thresh = ear_closed_thresh
        self.ear_open_thresh = ear_open_thresh
        self.verification_duration = verification_duration
        self.tracks = {}

    def update(self, track_id: str, ear: float):
        now = time.time()
        # Clean up tracks older than 60 seconds
        stale = [k for k, v in self.tracks.items() if now - v.get("last_seen", 0) > 60]
        for k in stale:
            self.tracks.pop(k, None)

        track = self.tracks.setdefault(track_id, {
            "state": "WAITING",
            "had_closed": False,
            "verified_until": 0.0,
            "last_seen": now
        })
        track["last_seen"] = now

        # If already verified within active grace period
        if now < track["verified_until"]:
            return True, "verified"

        # Check for eye closure
        if ear < self.ear_closed_thresh:
            track["had_closed"] = True
            track["state"] = "EYE_CLOSED"
            return False, "blink to confirm"

        # Check for reopening after closure -> BLINK COMPLETE!
        if ear >= self.ear_open_thresh:
            if track["had_closed"]:
                track["had_closed"] = False
                track["verified_until"] = now + self.verification_duration
                track["state"] = "VERIFIED"
                return True, "verified"
            else:
                track["state"] = "WAITING"
                return False, "blink to confirm"

        return False, "blink to confirm"

    def reset(self, track_id: str = None):
        if track_id:
            self.tracks.pop(track_id, None)
        else:
            self.tracks.clear()


class FaceEngine:
    def __init__(self, score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        ensure_models_exist()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        self.current_input_size = None
        self.blink_detector = BlinkDetector()

    def _get_detector(self, width: int, height: int):
        """Creates or reconfigures the YuNet detector with current frame dimensions."""
        if self.detector is None or self.current_input_size != (width, height):
            self.detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH,
                "",
                (width, height),
                self.score_threshold,
                self.nms_threshold,
                self.top_k
            )
            self.current_input_size = (width, height)
        return self.detector

    def detect_faces(self, image: np.ndarray, min_score=None):
        """
        Detect faces in a BGR image.
        Returns:
            list of face rows (each row contains bbox, landmarks, score)
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detector = self._get_detector(w, h)
        if min_score is not None and min_score != self.score_threshold:
            detector.setScoreThreshold(min_score)

        _, faces = detector.detect(image)
        if faces is None:
            return []
        return faces

    def extract_feature(self, image: np.ndarray, face_data: np.ndarray) -> np.ndarray:
        """
        Align crop and compute 128D feature embedding for a detected face.
        """
        aligned_face = self.recognizer.alignCrop(image, face_data)
        feature = self.recognizer.feature(aligned_face)
        # Normalize feature vector
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

    def extract_feature_from_image(self, image: np.ndarray):
        """
        Detects primary face in image and returns (face_feature, bbox, face_data)
        Returns (None, None, None) if no face detected.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None, None, None

        # Pick face with highest confidence / largest area
        best_face = faces[0]
        if len(faces) > 1:
            best_face = max(faces, key=lambda f: f[2] * f[3])  # max width * height

        feature = self.extract_feature(image, best_face)
        bbox = best_face[0:4].astype(int).tolist()
        return feature, bbox, best_face

    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Computes cosine similarity between two 128D face embeddings.
        Returns float score typically between 0.0 and 1.0.
        """
        if feature1 is None or feature2 is None:
            return 0.0
        # Cosine similarity via SFace match function
        sim = self.recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(sim)

    def find_best_match(self, query_feature: np.ndarray, known_profiles: list, threshold=DEFAULT_SIMILARITY_THRESHOLD):
        """
        Compares query_feature against a list of known student profiles.
        Each profile dict: {'id': int, 'roll_no': str, 'name': str, 'feature': np.ndarray, ...}
        Returns:
            (best_profile, similarity_score) if score >= threshold, else (None, best_score)
        """
        if query_feature is None or not known_profiles:
            return None, 0.0

        best_score = -1.0
        best_profile = None

        for profile in known_profiles:
            known_feat = profile.get("feature")
            if known_feat is None:
                continue
            sim = self.compute_similarity(query_feature, known_feat)
            if sim > best_score:
                best_score = sim
                best_profile = profile

        if best_score >= threshold:
            return best_profile, best_score
        return None, max(0.0, best_score)

    def process_frame(self, frame: np.ndarray, known_profiles: list, threshold=DEFAULT_SIMILARITY_THRESHOLD):
        """
        Processes a full webcam or video frame:
        Detects all faces, computes Eye Aspect Ratio (EAR), checks blink liveness,
        matches against known student profiles, and returns detection results.
        If a photograph is shown, eye never blinks so label stays at 'blink to confirm'.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        faces = self.detect_faces(frame)
        results = []

        for face in faces:
            bbox = face[0:4].astype(int).tolist()
            conf = float(face[-1])
            landmarks = face[4:14].reshape((5, 2)).astype(int).tolist()

            # Eye landmarks (Right eye, Left eye)
            re_pt = face[4:6]
            le_pt = face[6:8]
            eye_dist = float(np.linalg.norm(re_pt - le_pt))

            # Compute Eye Aspect Ratio (EAR)
            ear_r = compute_eye_aspect_ratio(frame, re_pt, eye_dist)
            ear_l = compute_eye_aspect_ratio(frame, le_pt, eye_dist)
            avg_ear = float((ear_r + ear_l) / 2.0)

            feature = self.extract_feature(frame, face)
            matched_profile, sim = self.find_best_match(feature, known_profiles, threshold)

            # Determine tracking ID for blink liveness state
            if matched_profile:
                track_id = f"student_{matched_profile['id']}"
            else:
                track_id = f"face_{bbox[0]//30}_{bbox[1]//30}"

            blink_confirmed, liveness_status = self.blink_detector.update(track_id, avg_ear)

            # User requirement:
            # When a photograph of a registered student is shown to the camera, the face is matched
            # but the label stays at "blink to confirm", because the photograph never blinks.
            if matched_profile:
                if blink_confirmed:
                    label = f"{matched_profile['name']} (Verified)"
                else:
                    label = f"{matched_profile['name']} - blink to confirm"
            else:
                label = f"Face Detected - blink to confirm"

            results.append({
                "bbox": {
                    "x": max(0, bbox[0]),
                    "y": max(0, bbox[1]),
                    "width": min(w - bbox[0], bbox[2]),
                    "height": min(h - bbox[1], bbox[3])
                },
                "detection_confidence": round(conf, 3),
                "landmarks": landmarks,
                "ear": round(avg_ear, 3),
                "blink_confirmed": blink_confirmed,
                "liveness_status": liveness_status,
                "label": label,
                "is_recognized": matched_profile is not None,
                "similarity": round(sim, 3),
                "student": matched_profile if matched_profile else None
            })

        return results

    @staticmethod
    def decode_base64_image(b64_string: str) -> np.ndarray:
        """
        Decodes a base64-encoded image string (with or without data:image/... prefix)
        into an OpenCV BGR numpy array.
        """
        if not b64_string:
            return None
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        try:
            image_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    @staticmethod
    def encode_base64_image(image: np.ndarray, format_ext=".jpg") -> str:
        """Encodes an OpenCV BGR image to base64 JPEG data URL."""
        if image is None:
            return ""
        _, buffer = cv2.imencode(format_ext, image)
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
