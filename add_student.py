"""
EduFace AI - Manual Student Enrollment Tool (CLI & Interactive)

Usage:
  Interactive Mode:
    python add_student.py

  Command-Line Mode:
    python add_student.py --name "John Doe" --roll "105" --dept "Computer Engineering" --image "path/to/photo.jpg"
"""

import sys
import os
import argparse
import shutil
import cv2

import database as db
from face_engine import FaceEngine


def enroll(name: str, roll_no: str, department: str, image_path: str, semester: str = "Sem 1", section: str = "A", email: str = "", phone: str = ""):
    db.init_database()
    engine = FaceEngine()

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return False

    # 1. Read and validate face
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image file: {image_path}")
        return False

    print(f"[*] Analyzing face in {image_path}...")
    feat, bbox, _ = engine.extract_feature_from_image(img)
    if feat is None:
        print("[ERROR] No face detected in the provided image! Please use a clear front-facing photo.")
        return False

    print(f"[+] Face detected successfully (Bounding box: {bbox})")

    # 2. Ensure department exists
    conn = db.get_db_connection()
    conn.execute("INSERT OR IGNORE INTO departments (name, code) VALUES (?, ?)", (department.strip(), department.strip()[:4].upper()))
    conn.commit()
    conn.close()

    # 3. Copy image to static/uploads/
    ext = os.path.splitext(image_path)[1].lower() or ".jpg"
    safe_roll = roll_no.replace("-", "_").replace(" ", "_")
    target_filename = f"{safe_roll}_{os.path.basename(image_path)}"
    dest_path = os.path.join(db.UPLOAD_DIR, target_filename)
    shutil.copyfile(image_path, dest_path)

    # Also copy to images/ if not already there
    dest_images = os.path.join("images", f"{safe_roll}{ext}")
    if not os.path.exists(dest_images):
        try:
            shutil.copyfile(image_path, dest_images)
        except Exception:
            pass

    # 4. Check if student already exists in database
    existing = db.get_student_by_roll_no(roll_no)
    if existing:
        print(f"[*] Student with Roll No '{roll_no}' already exists. Updating record...")
        db.update_student(
            student_id=existing["id"],
            roll_no=roll_no,
            name=name,
            department=department,
            semester=semester,
            section=section,
            email=email,
            phone=phone,
            photo_filename=target_filename,
            face_feature=feat
        )
        print(f"[SUCCESS] Updated student '{name}' (Roll No: {roll_no}) successfully!")
    else:
        student_id = db.add_student(
            roll_no=roll_no,
            name=name,
            department=department,
            semester=semester,
            section=section,
            email=email,
            phone=phone,
            photo_filename=target_filename,
            face_feature=feat
        )
        print(f"[SUCCESS] Enrolled new student '{name}' (Roll No: {roll_no}, ID: {student_id}) successfully!")

    return True


def interactive_prompt():
    print("=" * 60)
    print("🎓 EduFace AI - Manual Student Enrollment")
    print("=" * 60)

    name = input("Enter Student Full Name: ").strip()
    while not name:
        name = input("Name is required. Enter Student Full Name: ").strip()

    roll_no = input("Enter Roll Number / Student ID: ").strip()
    while not roll_no:
        roll_no = input("Roll Number is required. Enter Roll Number: ").strip()

    dept = input("Enter Department [default: Computer Engineering]: ").strip()
    if not dept:
        dept = "Computer Engineering"

    semester = input("Enter Semester [default: Sem 1]: ").strip() or "Sem 1"
    section = input("Enter Section [default: A]: ").strip() or "A"
    email = input("Enter Email (optional): ").strip()
    phone = input("Enter Phone (optional): ").strip()

    image_path = input("Enter Path to Photo Image (e.g., images/105.jpg or C:/path/to/pic.jpg): ").strip()
    # Strip quotes if copied from explorer
    image_path = image_path.strip('"').strip("'")
    while not os.path.exists(image_path):
        print(f"[ERROR] File does not exist at: {image_path}")
        image_path = input("Please enter a valid image file path: ").strip().strip('"').strip("'")

    print("-" * 60)
    enroll(name, roll_no, dept, image_path, semester, section, email, phone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll student manually into EduFace AI")
    parser.add_argument("--name", help="Student Full Name")
    parser.add_argument("--roll", help="Student Roll Number")
    parser.add_argument("--dept", help="Academic Department", default="Computer Engineering")
    parser.add_argument("--image", help="Path to student photo file")
    parser.add_argument("--sem", help="Semester / Class", default="Sem 1")
    parser.add_argument("--sec", help="Section", default="A")
    parser.add_argument("--email", help="Student Email", default="")
    parser.add_argument("--phone", help="Student Phone", default="")

    args = parser.parse_args()

    if args.name and args.roll and args.image:
        enroll(
            name=args.name,
            roll_no=args.roll,
            department=args.dept,
            image_path=args.image,
            semester=args.sem,
            section=args.sec,
            email=args.email,
            phone=args.phone
        )
    else:
        interactive_prompt()
