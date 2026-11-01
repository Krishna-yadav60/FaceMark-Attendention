"""
EduFace AI - School & College Face Recognition Attendance System
Entry point for Web Application and Native Desktop Kiosk mode.

Usage:
  python main.py            -> Launches the Web Application & opens http://127.0.0.1:5000
  python main.py --desktop  -> Launches Native OpenCV Desktop Attendance Window
"""

import sys
import os
import webbrowser
import threading
import time

def run_desktop_kiosk():
    """
    Native OpenCV GUI Desktop Attendance Kiosk mode.
    Embeds the live webcam feed into Resources/Background.png
    and logs attendance into eduface.db.
    """
    import cv2
    import numpy as np
    import database as db
    from face_engine import FaceEngine

    print("[*] Initializing Database & Neural Face Engine...")
    db.init_database()
    engine = FaceEngine()
    db.seed_sample_students_from_existing_images(engine)
    profiles = db.get_all_student_face_profiles()
    print(f"[*] Loaded {len(profiles)} enrolled student face embeddings.")

    bg_path = os.path.join("Resources", "Background.png")
    if not os.path.exists(bg_path):
        print(f"[!] Warning: {bg_path} not found. Running with direct camera display.")
        img_background_template = None
    else:
        img_background_template = cv2.imread(bg_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam device (index 0).")
        return

    print("=" * 60)
    print("🎥 Native Desktop Face Attendance Kiosk Running")
    print("Press 'q' in the window to quit.")
    print("=" * 60)

    last_recognized_text = "Ready to scan"
    last_status_color = (200, 200, 200)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Resize camera frame to exact 640x480
        img_cam = cv2.resize(img, (640, 480))

        # Run face recognition
        threshold = float(db.get_setting("similarity_threshold", "0.42"))
        detections = engine.process_frame(img_cam, profiles, threshold=threshold)

        for det in detections:
            bbox = det["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            is_match = det["is_recognized"]
            blink_confirmed = det.get("blink_confirmed", False)
            color = (0, 220, 0) if (is_match and blink_confirmed) else (0, 165, 255)

            cv2.rectangle(img_cam, (x, y), (x + w, y + h), color, 2)

            if is_match and det["student"]:
                student = det["student"]
                name = student["name"]
                roll = student["roll_no"]
                conf = det["similarity"]

                # User requirement: photo never blinks, so label stays at "blink to confirm" and attendance not marked
                if not blink_confirmed:
                    label = f"{name} - blink to confirm"
                    last_recognized_text = f"{name} - blink to confirm"
                    last_status_color = (0, 165, 255)
                else:
                    # Record attendance upon confirmed blink
                    res = db.record_attendance(student["id"], confidence=conf, scan_mode="Native Desktop Kiosk")
                    status = res.get("attendance_status", "Present")
                    last_recognized_text = f"{name} ({roll}) - {status}"
                    last_status_color = (0, 255, 0) if status == "Present" else (0, 165, 255)
                    label = f"{name} (Verified)"
            else:
                label = f"Scanning ({int(det['detection_confidence'] * 100)}%)"

            cv2.putText(img_cam, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if img_background_template is not None:
            canvas = img_background_template.copy()
            # Overlay webcam at y:162..642, x:55..695
            canvas[162:162 + 480, 55:55 + 640] = img_cam

            # Render status text overlay
            cv2.putText(canvas, last_recognized_text, (750, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_status_color, 2)
            cv2.putText(canvas, f"Enrolled: {len(profiles)} Students", (750, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

            cv2.imshow("EduFace AI - Native Attendance Kiosk", canvas)
        else:
            cv2.imshow("EduFace AI - Native Attendance Kiosk", img_cam)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_web_app():
    """Starts the full web application and opens browser."""
    from app import app
    port = 5000

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 65)
    print("🚀 EduFace AI - Institutional Biometric Attendance System")
    print(f"📡 Web Application Server starting at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    if "--desktop" in sys.argv or "--gui" in sys.argv:
        run_desktop_kiosk()
    else:
        run_web_app()