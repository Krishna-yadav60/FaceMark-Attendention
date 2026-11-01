"""
EduFace AI - Enterprise Institutional Face Recognition Attendance Server & REST API
Features:
- Biometric Face Detection & Recognition (YuNet + SFace)
- Role-based User Authentication & Session Security (Admin / Faculty / Kiosk)
- Real-time Server-Sent Events (SSE) Event Bus
- Course & Lecture Slot Attendance Engine
- Automated Absenteeism & Punctuality Intelligence
- Folder Ingestion, Batch CSV Synchronization & One-Click Backup
"""

import os
import io
import csv
import time
import datetime
import traceback
import cv2
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    send_file, Response, flash, session
)
from werkzeug.utils import secure_filename

import database as db
from face_engine import FaceEngine, MODEL_DIR
from services.event_stream import event_bus
from services.data_import_service import import_students_from_csv
from services.analytics_service import (
    get_hourly_arrival_distribution,
    get_at_risk_students_report,
    generate_daily_executive_summary
)

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "eduface-ai-institutional-attendance-secret-key-2026"
app.config["UPLOAD_FOLDER"] = db.UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Initialize Core Services
db.init_database()
face_engine = FaceEngine()
db.seed_sample_students_from_existing_images(face_engine)

# In-memory cache of student face profiles for fast real-time recognition
cached_profiles = db.get_all_student_face_profiles()
cached_profiles_last_updated = time.time()


def refresh_cached_profiles():
    """Reloads student profiles and face feature vectors into memory."""
    global cached_profiles, cached_profiles_last_updated
    cached_profiles = db.get_all_student_face_profiles()
    cached_profiles_last_updated = time.time()


# ==========================================
# AUTHENTICATION & ACCESS CONTROL
# ==========================================

def get_current_user():
    """Returns currently authenticated user from session."""
    uid = session.get("user_id")
    if uid:
        user = db.get_user_by_id(uid)
        if user and session.get("student_id") and not user.get("student_id"):
            user["student_id"] = session.get("student_id")
        return user
    return None


@app.context_processor
def inject_user_context():
    """Injects current_user and system settings into all Jinja templates."""
    user = get_current_user()
    inst_name = db.get_setting("institution_name", "EduFace AI Institution")
    academic_sess = db.get_setting("academic_session", "2026-2027")
    shortage_lim = float(db.get_setting("attendance_shortage_limit", "75.0"))
    return {
        "current_user": user,
        "institution_name": inst_name,
        "academic_session": academic_sess,
        "attendance_shortage_limit": shortage_lim
    }


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to access this portal.", "warning")
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user or user.get("role") != "admin":
            flash("Administrative privileges required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = db.verify_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            if user.get("student_id"):
                session["student_id"] = user["student_id"]

            db.log_audit("USER_LOGIN", user["username"], f"Signed in as {user['role']}", request.remote_addr)
            flash(f"Welcome back, {user['full_name']}!", "success")

            # Route student role straight to dedicated Student Portal
            if user.get("role") == "student":
                return redirect(url_for("student_portal"))

            next_url = request.args.get("next") or url_for("dashboard")
            return redirect(next_url)
        else:
            flash("Invalid username or password. Please try again.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    uname = session.get("username", "Unknown")
    db.log_audit("USER_LOGOUT", uname, "User logged out", request.remote_addr)
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login_page"))


# ==========================================
# PAGE VIEWS
# ==========================================

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    stats = db.get_dashboard_stats()
    hourly = get_hourly_arrival_distribution()
    at_risk = get_at_risk_students_report()
    return render_template(
        "dashboard.html",
        stats=stats,
        hourly_data=hourly,
        at_risk_count=len(at_risk),
        active_page="dashboard"
    )


@app.route("/scanner")
def scanner():
    threshold = float(db.get_setting("similarity_threshold", "0.42"))
    cooldown = int(db.get_setting("cooldown_minutes", "10"))
    courses = db.get_all_courses()
    session_id = request.args.get("session_id")
    active_session = None
    if session_id:
        sessions = db.get_lecture_sessions()
        active_session = next((s for s in sessions if str(s["id"]) == str(session_id)), None)
    return render_template(
        "scanner.html",
        similarity_threshold=threshold,
        cooldown_minutes=cooldown,
        courses=courses,
        session_id=session_id,
        active_session=active_session,
        active_page="scanner"
    )


@app.route("/timetable")
def timetable_page():
    weekly = db.get_weekly_timetable()
    courses = db.get_all_courses()
    departments = db.get_departments()
    return render_template(
        "timetable.html",
        weekly_timetable=weekly,
        courses=courses,
        departments=departments,
        active_page="timetable"
    )


@app.route("/academics")
def academics_page():
    records = db.get_academic_records(limit=200)
    summary = db.get_academics_summary()
    courses = db.get_all_courses()
    students = db.get_all_students()
    return render_template(
        "academics.html",
        records=records,
        summary=summary,
        courses=courses,
        students=students,
        active_page="academics"
    )


@app.route("/students")
def students_page():
    search = request.args.get("search", "").strip()
    dept = request.args.get("department", "").strip()
    students = db.get_all_students(search=search or None, department=dept or None)
    departments = db.get_departments()
    return render_template(
        "students.html",
        students=students,
        departments=departments,
        current_search=search,
        current_dept=dept,
        active_page="students"
    )


@app.route("/courses")
def courses_page():
    courses = db.get_all_courses()
    departments = db.get_departments()
    students = db.get_all_students()
    sessions = db.get_lecture_sessions()
    return render_template(
        "courses.html",
        courses=courses,
        departments=departments,
        students=students,
        sessions=sessions[:20],
        active_page="courses"
    )


@app.route("/attendance")
def attendance_page():
    date_filter = request.args.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    dept_filter = request.args.get("department", "")
    status_filter = request.args.get("status", "")
    search_filter = request.args.get("search", "")

    logs = db.get_attendance_logs(
        date=date_filter or None,
        department=dept_filter or None,
        status=status_filter or None,
        search=search_filter or None
    )
    departments = db.get_departments()
    students = db.get_all_students()

    return render_template(
        "attendance.html",
        logs=logs,
        departments=departments,
        students=students,
        current_date=date_filter,
        current_dept=dept_filter,
        current_status=status_filter,
        current_search=search_filter,
        active_page="attendance"
    )


@app.route("/reports")
def reports_page():
    """
    ReportsWindow view:
    Shows student-wise and subject-wise attendance as total sessions, present sessions, and percentage.
    Lists shortage alerts (< configurable limit, default 75%).
    Enables export to Excel/CSV in the exports/ folder.
    """
    user = get_current_user()
    if user and user.get("role") == "student":
        # Students are redirected to their own portal (cannot see other students' data)
        return redirect(url_for("student_portal"))

    threshold = float(db.get_setting("attendance_shortage_limit", "75.0"))
    dept = request.args.get("department", "").strip()

    student_report = db.get_student_attendance_summary_report(department=dept or None, limit_threshold=threshold)
    subject_report = db.get_subject_attendance_summary_report(department=dept or None, limit_threshold=threshold)
    shortage_alerts = [s for s in student_report["students"] if s["is_shortage"]]
    departments = db.get_departments()
    courses = db.get_all_courses(department=dept or None)

    return render_template(
        "reports.html",
        student_report=student_report,
        subject_report=subject_report,
        shortage_alerts=shortage_alerts,
        departments=departments,
        courses=courses,
        current_dept=dept,
        threshold=threshold,
        active_page="reports"
    )


@app.route("/portal")
@app.route("/student/portal")
def student_portal():
    """
    Student Portal:
    Shows the logged-in student's subject-wise attendance percentage and marks.
    STRICT PRIVACY: A student cannot see the data of any other student.
    """
    user = get_current_user()
    if not user:
        flash("Please sign in with your student credentials.", "warning")
        return redirect(url_for("login_page", next=request.url))

    # Determine student ID with strict security enforcement
    if user.get("role") == "student":
        student_id = user.get("student_id")
        if not student_id:
            # Look up student by username (roll_no)
            stu = db.get_student_by_roll_no(user["username"])
            student_id = stu["id"] if stu else None
    else:
        # Faculty / Admin preview
        req_id = request.args.get("student_id", type=int)
        if req_id:
            student_id = req_id
        else:
            all_s = db.get_all_students()
            student_id = all_s[0]["id"] if all_s else None

    if not student_id:
        flash("Student profile not found.", "danger")
        return redirect(url_for("login_page"))

    portal_data = db.get_student_portal_data(student_id)
    if not portal_data or not portal_data.get("student"):
        flash("Student records unavailable.", "danger")
        return redirect(url_for("login_page"))

    return render_template(
        "student_portal.html",
        portal_data=portal_data,
        data=portal_data,
        student=portal_data["student"],
        summary=portal_data["summary"],
        subject_attendance=portal_data["subject_attendance"],
        academic_marks=portal_data["academic_marks"],
        recent_logs=portal_data["recent_logs"],
        threshold=portal_data["threshold"],
        average_marks=portal_data["average_marks"],
        is_preview=(user.get("role") != "student"),
        active_page="student_portal"
    )


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        for key in [
            "institution_name", "institution_code", "class_start_time",
            "late_cutoff_time", "similarity_threshold", "cooldown_minutes",
            "academic_session", "attendance_shortage_limit"
        ]:
            if key in request.form:
                db.update_setting(key, request.form[key].strip())
        db.log_audit("SETTINGS_UPDATED", session.get("username", "Admin"), "Updated system settings", request.remote_addr)
        flash("Settings saved successfully!", "success")
        return redirect(url_for("settings_page"))

    settings = db.get_all_settings()
    departments = db.get_departments()
    users = db.get_all_users()
    logs = db.get_system_logs(25)
    return render_template(
        "settings.html",
        settings=settings,
        departments=departments,
        users=users,
        logs=logs,
        active_page="settings"
    )


# ==========================================
# REST API: BIOMETRICS & RECOGNITION
# ==========================================

@app.route("/api/recognize_frame", methods=["POST"])
def api_recognize_frame():
    """
    Receives a webcam frame as base64 string from client browser,
    runs face detection & recognition against cached student profiles,
    and records attendance if student is recognized.
    Broadcasts real-time event to SSE listeners.
    """
    global cached_profiles
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"status": "error", "message": "Missing image data"}), 400

        frame = FaceEngine.decode_base64_image(image_b64)
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        threshold = float(db.get_setting("similarity_threshold", "0.42"))
        scan_mode = data.get("scan_mode", "Web Scanner")
        course_session_id = data.get("session_id")
        skip_liveness = bool(data.get("skip_liveness", False))

        detections = face_engine.process_frame(frame, cached_profiles, threshold=threshold)
        results = []

        for det in detections:
            blink_confirmed = det.get("blink_confirmed", False) or skip_liveness
            liveness_status = "verified" if blink_confirmed else "blink to confirm"

            res_item = {
                "bbox": det["bbox"],
                "landmarks": det["landmarks"],
                "detection_confidence": det["detection_confidence"],
                "is_recognized": det["is_recognized"],
                "similarity": det["similarity"],
                "ear": det.get("ear", 0.5),
                "blink_confirmed": blink_confirmed,
                "liveness_status": liveness_status,
                "label": det.get("label", "blink to confirm")
            }

            if det["is_recognized"] and det["student"]:
                student = det["student"]
                res_item["student"] = {
                    "id": student["id"],
                    "roll_no": student["roll_no"],
                    "name": student["name"],
                    "department": student["department"],
                    "semester": student["semester"],
                    "section": student["section"],
                    "photo_filename": student.get("photo_filename")
                }

                # CRITICAL USER REQUIREMENT:
                # When a photograph of a registered student is shown to the camera, the face is matched
                # but the label stays at "blink to confirm", because the photograph never blinks.
                # Attendance is therefore not marked!
                if not blink_confirmed:
                    # Photo or awaiting blink: Keep label at "blink to confirm" and DO NOT record attendance!
                    res_item["attendance"] = None
                    res_item["label"] = f"{student['name']} - blink to confirm"
                else:
                    # Verified via genuine blink (or explicit override):
                    res_item["label"] = f"{student['name']} (Verified)"

                    # 1. Record daily campus attendance
                    att_result = db.record_attendance(
                        student_id=student["id"],
                        confidence=det["similarity"],
                        scan_mode=scan_mode
                    )

                    # 2. Record course lecture attendance if session specified
                    if course_session_id:
                        db.record_course_attendance(
                            session_id=int(course_session_id),
                            student_id=student["id"],
                            status=att_result.get("attendance_status", "Present"),
                            confidence=det["similarity"],
                            scan_mode=scan_mode
                        )

                    res_item["attendance"] = att_result

                    # Broadcast real-time event to connected browsers (SSE)
                    if att_result.get("status") == "SUCCESS":
                        event_bus.publish("student_verified", {
                            "student": res_item["student"],
                            "attendance": att_result,
                            "time": att_result.get("time")
                        })

            else:
                res_item["student"] = None
                res_item["attendance"] = None

            results.append(res_item)

        return jsonify({
            "status": "success",
            "faces_detected": len(results),
            "results": results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/validate_face", methods=["POST"])
def api_validate_face():
    """Validates face in image without saving."""
    try:
        image = None
        if "photo_file" in request.files:
            file = request.files["photo_file"]
            if file and file.filename:
                nparr = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.is_json:
            data = request.get_json()
            image_b64 = data.get("image")
            if image_b64:
                image = FaceEngine.decode_base64_image(image_b64)

        if image is None:
            return jsonify({"valid": False, "message": "No valid image provided"}), 400

        feat, bbox, _ = face_engine.extract_feature_from_image(image)
        if feat is None:
            return jsonify({
                "valid": False,
                "message": "No face detected. Please ensure proper lighting and face the camera directly."
            })

        return jsonify({
            "valid": True,
            "message": "Face detected successfully!",
            "bbox": bbox
        })
    except Exception as e:
        return jsonify({"valid": False, "message": str(e)}), 500


# ==========================================
# REST API: STUDENTS & BULK INTEGRATION
# ==========================================

@app.route("/api/students", methods=["GET", "POST"])
def api_students():
    if request.method == "GET":
        students = db.get_all_students()
        return jsonify(students)

    try:
        roll_no = request.form.get("roll_no", "").strip()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip()
        semester = request.form.get("semester", "Sem 1").strip()
        section = request.form.get("section", "A").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        if not roll_no or not name or not department:
            return jsonify({"status": "error", "message": "Roll No, Name, and Department are required"}), 400

        existing = db.get_student_by_roll_no(roll_no)
        if existing:
            return jsonify({"status": "error", "message": f"Student with Roll No '{roll_no}' already exists"}), 400

        image = None
        photo_filename = ""

        if "photo_file" in request.files and request.files["photo_file"].filename:
            file = request.files["photo_file"]
            safe_name = f"{secure_filename(roll_no)}_{int(time.time())}.jpg"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imwrite(save_path, image)
                photo_filename = safe_name
        elif request.form.get("photo_base64"):
            b64_str = request.form.get("photo_base64")
            image = FaceEngine.decode_base64_image(b64_str)
            if image is not None:
                safe_name = f"{secure_filename(roll_no)}_{int(time.time())}.jpg"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
                cv2.imwrite(save_path, image)
                photo_filename = safe_name

        feat = None
        if image is not None:
            feat, _, _ = face_engine.extract_feature_from_image(image)

        student_id = db.add_student(
            roll_no=roll_no,
            name=name,
            department=department,
            semester=semester,
            section=section,
            email=email,
            phone=phone,
            photo_filename=photo_filename,
            face_feature=feat
        )

        refresh_cached_profiles()

        return jsonify({
            "status": "success",
            "message": f"Student '{name}' enrolled successfully!",
            "student_id": student_id,
            "has_face_embedding": feat is not None
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/students/<int:student_id>", methods=["GET", "PUT", "DELETE"])
def api_student_detail(student_id):
    student = db.get_student_by_id(student_id)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    if request.method == "GET":
        analytics = db.get_student_analytics(student_id)
        student.update(analytics)
        return jsonify(student)

    if request.method == "DELETE":
        db.delete_student(student_id)
        refresh_cached_profiles()
        return jsonify({"status": "success", "message": "Student deleted successfully"})

    if request.method == "PUT":
        data = request.form or request.get_json(force=True)
        roll_no = data.get("roll_no", student["roll_no"]).strip()
        name = data.get("name", student["name"]).strip()
        department = data.get("department", student["department"]).strip()
        semester = data.get("semester", student["semester"]).strip()
        section = data.get("section", student["section"]).strip()
        email = data.get("email", student["email"]).strip()
        phone = data.get("phone", student["phone"]).strip()

        db.update_student(
            student_id=student_id,
            roll_no=roll_no,
            name=name,
            department=department,
            semester=semester,
            section=section,
            email=email,
            phone=phone
        )
        refresh_cached_profiles()
        return jsonify({"status": "success", "message": "Student updated successfully"})


@app.route("/api/students/import_csv", methods=["POST"])
def api_import_csv():
    """Bulk imports students from CSV."""
    if "csv_file" not in request.files:
        return jsonify({"status": "error", "message": "No CSV file uploaded"}), 400
    file = request.files["csv_file"]
    if not file.filename:
        return jsonify({"status": "error", "message": "Empty file name"}), 400

    csv_text = file.read().decode("utf-8", errors="replace")
    result = import_students_from_csv(csv_text, face_engine)
    refresh_cached_profiles()
    db.log_audit("CSV_IMPORT", session.get("username", "Admin"), f"Imported {result['imported']} students, updated {result['updated']}")
    return jsonify(result)


@app.route("/api/sync_folder", methods=["POST"])
def api_sync_folder():
    """Auto-detects and enrolls all photos in images/ directory."""
    result = db.sync_images_folder(face_engine)
    refresh_cached_profiles()
    db.log_audit("FOLDER_SYNC", session.get("username", "Admin"), f"Synced images/ folder: {result.get('newly_enrolled', 0)} new enrollments")
    return jsonify(result)


# ==========================================
# REST API: COURSES & SESSIONS
# ==========================================

@app.route("/api/courses", methods=["GET", "POST"])
def api_courses():
    if request.method == "GET":
        dept = request.args.get("department")
        return jsonify(db.get_all_courses(dept))

    data = request.get_json(force=True)
    code = data.get("code")
    name = data.get("name")
    dept = data.get("department")
    sem = data.get("semester", "Sem 1")
    credits = int(data.get("credits", 3))
    faculty = data.get("faculty_name", "")

    if not code or not name or not dept:
        return jsonify({"status": "error", "message": "Code, Name, and Department are required"}), 400

    cid = db.create_course(code, name, dept, sem, credits, faculty)
    return jsonify({"status": "success", "course_id": cid})


@app.route("/api/courses/<int:course_id>/enroll", methods=["POST"])
def api_course_enroll(course_id):
    data = request.get_json(force=True)
    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"status": "error", "message": "Student ID required"}), 400
    db.enroll_student_in_course(course_id, int(student_id))
    return jsonify({"status": "success", "message": "Student enrolled in course"})


@app.route("/api/sessions", methods=["GET", "POST"])
def api_sessions():
    if request.method == "GET":
        course_id = request.args.get("course_id")
        return jsonify(db.get_lecture_sessions(course_id=int(course_id) if course_id else None))

    data = request.get_json(force=True)
    cid = data.get("course_id")
    date_val = data.get("session_date")
    time_val = data.get("start_time")
    room = data.get("room", "Hall 101")

    sid = db.create_lecture_session(int(cid), date_val, time_val, room=room)
    return jsonify({"status": "success", "session_id": sid})


@app.route("/api/sessions/<int:session_id>/end", methods=["POST"])
def api_end_session(session_id):
    """
    Teacher ends active lecture attendance session:
    - Summarizes students marked Present with exact time & confidence.
    - All remaining students of the class are stored as ABSENT in course_attendance.
    - Strict deduplication ensures each student appears only once.
    """
    try:
        summary = db.end_lecture_session(session_id)
        if not summary:
            return jsonify({"status": "error", "message": f"Session #{session_id} not found"}), 404

        username = session.get("username", "Instructor")
        db.log_audit(
            "SESSION_COMPLETED", username,
            f"Ended session #{session_id} for {summary['course_code']}. Present: {summary['present_count']}, Absent: {summary['absent_count']}",
            request.remote_addr
        )

        # Broadcast SSE event to real-time clients
        event_bus.publish("session_ended", {
            "session_id": session_id,
            "course_code": summary["course_code"],
            "course_name": summary["course_name"],
            "summary": summary
        })

        return jsonify({
            "status": "success",
            "message": f"Lecture session ended successfully. {summary['present_count']} marked Present, {summary['absent_count']} recorded as ABSENT.",
            "summary": summary
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================
# REST API: TIMETABLE SCHEDULE
# ==========================================

@app.route("/api/timetable", methods=["GET", "POST"])
def api_timetable():
    if request.method == "GET":
        day = request.args.get("day")
        course_id = request.args.get("course_id")
        return jsonify(db.get_timetable_entries(day_of_week=day or None, course_id=int(course_id) if course_id else None))

    data = request.get_json(force=True) if request.is_json else request.form
    day = data.get("day_of_week", "").strip()
    st = data.get("start_time", "").strip()
    et = data.get("end_time", "").strip()
    course_id = data.get("course_id")
    room = data.get("room", "Room 101").strip()
    faculty = data.get("faculty_name", "").strip()

    if not day or not st or not et or not course_id:
        return jsonify({"status": "error", "message": "Day, Start Time, End Time, and Course are required"}), 400

    tid = db.add_timetable_entry(day, st, et, int(course_id), room, faculty)
    return jsonify({"status": "success", "timetable_id": tid, "message": "Timetable entry created successfully"})


@app.route("/api/timetable/<int:entry_id>", methods=["PUT", "DELETE"])
def api_timetable_item(entry_id):
    if request.method == "DELETE":
        db.delete_timetable_entry(entry_id)
        return jsonify({"status": "success", "message": "Timetable entry deleted successfully"})

    data = request.get_json(force=True) if request.is_json else request.form
    day = data.get("day_of_week", "").strip()
    st = data.get("start_time", "").strip()
    et = data.get("end_time", "").strip()
    course_id = data.get("course_id")
    room = data.get("room", "Room 101").strip()
    faculty = data.get("faculty_name", "").strip()

    db.update_timetable_entry(entry_id, day, st, et, int(course_id), room, faculty)
    return jsonify({"status": "success", "message": "Timetable entry updated successfully"})


@app.route("/api/timetable/<int:entry_id>/start_session", methods=["POST"])
def api_start_session_from_timetable(entry_id):
    """Launches an instant lecture attendance session directly from a timetable slot."""
    entry = db.get_timetable_entry_by_id(entry_id)
    if not entry:
        return jsonify({"status": "error", "message": "Timetable slot not found"}), 404

    now = datetime.datetime.now()
    sid = db.create_lecture_session(
        course_id=entry["course_id"],
        session_date=now.strftime("%Y-%m-%d"),
        start_time=now.strftime("%H:%M"),
        end_time=entry["end_time"],
        room=entry["room"],
        status="Active"
    )
    return jsonify({
        "status": "success",
        "session_id": sid,
        "redirect": url_for("scanner", session_id=sid),
        "message": f"Started attendance session for {entry['course_code']} - {entry['course_name']}"
    })


# ==========================================
# REST API: ACADEMICS & EXAM RECORDS (AcademicsWindow)
# ==========================================

@app.route("/api/academics", methods=["GET", "POST"])
def api_academics():
    """
    AcademicsWindow endpoint:
    GET: Retrieve academic exam marks linked to students and subjects.
    POST: Enter exam type, marks obtained, and max marks for a student and subject.
    """
    if request.method == "GET":
        sid = request.args.get("student_id")
        cid = request.args.get("course_id")
        etype = request.args.get("exam_type")
        search = request.args.get("search")
        return jsonify(db.get_academic_records(
            student_id=int(sid) if sid else None,
            course_id=int(cid) if cid else None,
            exam_type=etype or None,
            search=search or None
        ))

    data = request.get_json(force=True) if request.is_json else request.form
    student_id = data.get("student_id")
    course_id = data.get("course_id")
    exam_type = data.get("exam_type", "Midterm Exam").strip()
    marks_obtained = data.get("marks_obtained")
    max_marks = data.get("max_marks", 100.0)
    remarks = data.get("remarks", "").strip()
    exam_date = data.get("exam_date")

    if not student_id or not course_id or marks_obtained is None or not exam_type:
        return jsonify({"status": "error", "message": "Student, Subject/Course, Exam Type, and Marks Obtained are required"}), 400

    try:
        rid = db.add_academic_record(
            student_id=int(student_id),
            course_id=int(course_id),
            exam_type=exam_type,
            marks_obtained=float(marks_obtained),
            max_marks=float(max_marks),
            remarks=remarks,
            exam_date=exam_date or None
        )
        rec = db.get_academic_record_by_id(rid)
        db.log_audit("ACADEMIC_RECORD_ADDED", session.get("username", "Faculty"),
                     f"Entered {exam_type} marks for {rec['student_name']}: {rec['marks_obtained']}/{rec['max_marks']} ({rec['grade']})")

        return jsonify({
            "status": "success",
            "message": f"Academic record saved for {rec['student_name']}: {rec['marks_obtained']}/{rec['max_marks']} ({rec['grade']})",
            "record": rec
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/academics/<int:record_id>", methods=["PUT", "DELETE"])
def api_academics_item(record_id):
    if request.method == "DELETE":
        db.delete_academic_record(record_id)
        return jsonify({"status": "success", "message": "Academic record deleted successfully"})

    data = request.get_json(force=True) if request.is_json else request.form
    exam_type = data.get("exam_type", "").strip()
    marks_obtained = data.get("marks_obtained")
    max_marks = data.get("max_marks", 100.0)
    remarks = data.get("remarks", "").strip()
    exam_date = data.get("exam_date")

    db.update_academic_record(
        record_id=record_id,
        exam_type=exam_type,
        marks_obtained=float(marks_obtained),
        max_marks=float(max_marks),
        remarks=remarks,
        exam_date=exam_date or None
    )
    rec = db.get_academic_record_by_id(record_id)
    return jsonify({"status": "success", "record": rec, "message": "Academic record updated successfully"})


@app.route("/api/academics/batch", methods=["POST"])
def api_academics_batch():
    """Batch grades entry for an entire class/subject."""
    try:
        data = request.get_json(force=True)
        course_id = data.get("course_id")
        exam_type = data.get("exam_type")
        max_marks = float(data.get("max_marks", 100.0))
        records = data.get("records", [])
        exam_date = data.get("exam_date")

        if not course_id or not exam_type or not records:
            return jsonify({"status": "error", "message": "Course, Exam Type, and Student marks records are required"}), 400

        result = db.batch_add_academic_records(int(course_id), exam_type, max_marks, records, exam_date=exam_date)
        return jsonify({
            "status": "success",
            "result": result,
            "message": f"Successfully recorded grades for {result['total']} students ({result['added']} new, {result['updated']} updated)!"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/academics/summary")
def api_academics_summary():
    return jsonify(db.get_academics_summary())


@app.route("/api/academics/export")
def api_export_academics():
    """Exports academic gradebook as CSV."""
    cid = request.args.get("course_id")
    etype = request.args.get("exam_type")
    records = db.get_academic_records(course_id=int(cid) if cid else None, exam_type=etype or None, limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID", "Roll Number", "Student Name", "Department", "Semester", "Section",
        "Subject Code", "Subject Name", "Exam Type", "Marks Obtained", "Maximum Marks",
        "Percentage", "Grade", "Remarks", "Exam Date"
    ])

    for r in records:
        writer.writerow([
            r["id"], r["roll_no"], r["student_name"], r["student_dept"], r["student_sem"], r["student_sec"],
            r["course_code"], r["course_name"], r["exam_type"], r["marks_obtained"], r["max_marks"],
            f"{r['percentage']:.2f}%", r["grade"], r["remarks"] or "", r["exam_date"]
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"academic_gradebook_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    )


# ==========================================
# REST API: ANALYTICS & EXECUTIVE INTELLIGENCE
# ==========================================

@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_dashboard_stats())


@app.route("/api/analytics/executive")
def api_analytics_executive():
    return jsonify(generate_daily_executive_summary())


@app.route("/api/analytics/hourly")
def api_analytics_hourly():
    date_val = request.args.get("date")
    return jsonify(get_hourly_arrival_distribution(date_val))


@app.route("/api/absentees")
def api_absentees():
    return jsonify(db.get_today_absentees())


@app.route("/api/analytics/at-risk")
def api_at_risk():
    return jsonify(get_at_risk_students_report())


# ==========================================
# REAL-TIME SERVER-SENT EVENTS (SSE)
# ==========================================

@app.route("/api/events")
def api_events():
    """SSE endpoint for live dashboard push notifications."""
    def event_stream():
        q = event_bus.subscribe()
        try:
            # Send initial ping
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                msg = q.get()
                yield msg
        except GeneratorExit:
            event_bus.unsubscribe(q)

    return Response(event_stream(), mimetype="text/event-stream")


# ==========================================
# ATTENDANCE MANUAL & EXPORT
# ==========================================

@app.route("/api/attendance/manual", methods=["POST"])
def api_manual_attendance():
    try:
        data = request.get_json(force=True)
        student_id = data.get("student_id")
        status = data.get("status", "Present")
        notes = data.get("notes", "Manual instructor override")

        if not student_id:
            return jsonify({"status": "error", "message": "Student ID is required"}), 400

        res = db.record_attendance(
            student_id=int(student_id),
            confidence=1.0,
            scan_mode="Manual Admin",
            notes=notes,
            force=True
        )
        db.log_audit("MANUAL_ATTENDANCE", session.get("username", "Admin"), f"Marked student ID {student_id} as {status}")
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/attendance/export")
def api_export_attendance():
    date_filter = request.args.get("date")
    dept_filter = request.args.get("department")
    status_filter = request.args.get("status")

    logs = db.get_attendance_logs(
        date=date_filter or None,
        department=dept_filter or None,
        status=status_filter or None,
        limit=10000
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID", "Roll Number", "Student Name", "Department",
        "Semester", "Section", "Date", "Time", "Status", "Confidence", "Scan Mode", "Notes"
    ])

    for row in logs:
        writer.writerow([
            row["id"],
            row["roll_no"],
            row["name"],
            row["department"],
            row["semester"],
            row["section"],
            row["date"],
            row["time"],
            row["status"],
            f"{row['confidence'] * 100:.1f}%" if row.get("confidence") else "100%",
            row["scan_mode"],
            row["notes"] or ""
        ])

    output.seek(0)
    filename = f"attendance_report_{date_filter or 'all'}.csv"
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


# ==========================================
# REST API: REPORTS WINDOW & SHORTAGE ALERTS
# ==========================================

@app.route("/api/reports/student_wise")
def api_reports_student_wise():
    user = get_current_user()
    if user and user.get("role") == "student":
        # PRIVACY ENFORCEMENT: A student cannot see other students' data!
        return jsonify(db.get_student_portal_data(user["student_id"]))

    dept = request.args.get("department")
    shortage = request.args.get("shortage_only") == "true"
    threshold = request.args.get("threshold", type=float)
    return jsonify(db.get_student_attendance_summary_report(department=dept or None, shortage_only=shortage, limit_threshold=threshold))


@app.route("/api/reports/subject_wise")
def api_reports_subject_wise():
    user = get_current_user()
    if user and user.get("role") == "student":
        return jsonify({"status": "error", "message": "Access restricted"}), 403

    cid = request.args.get("course_id", type=int)
    dept = request.args.get("department")
    threshold = request.args.get("threshold", type=float)
    return jsonify(db.get_subject_attendance_summary_report(course_id=cid, department=dept or None, limit_threshold=threshold))


@app.route("/api/reports/shortage_alerts")
def api_reports_shortage_alerts():
    user = get_current_user()
    if user and user.get("role") == "student":
        return jsonify({"status": "error", "message": "Access restricted"}), 403

    dept = request.args.get("department")
    threshold = request.args.get("threshold", type=float)
    return jsonify(db.get_shortage_alerts_report(department=dept or None, threshold=threshold))


@app.route("/api/reports/export")
def api_export_reports():
    """
    Exports attendance report to an Excel or CSV file in the exports/ folder.
    Returns the file directly for browser download.
    """
    user = get_current_user()
    if user and user.get("role") == "student":
        return jsonify({"status": "error", "message": "Export restricted to faculty and administrators"}), 403

    rep_type = request.args.get("type", "student_wise")
    dept = request.args.get("department")
    cid = request.args.get("course_id", type=int)
    shortage_only = request.args.get("shortage_only") == "true"
    threshold = request.args.get("threshold", type=float)

    filepath = db.export_attendance_report(
        report_type=rep_type,
        department=dept or None,
        course_id=cid,
        shortage_only=shortage_only,
        threshold=threshold
    )

    filename = os.path.basename(filepath)
    return send_file(
        filepath,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/reports/send_shortage_alert", methods=["POST"])
def api_send_shortage_alert():
    """Action for teacher to notify a student regarding attendance shortage before end of term."""
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        student_id = data.get("student_id")
        student = db.get_student_by_id(int(student_id))
        if not student:
            return jsonify({"status": "error", "message": "Student not found"}), 404

        pct = data.get("percentage", "unknown")
        needed = data.get("recovery_needed", "classes")
        db.create_notification(
            title=f"Attendance Shortage Warning ({student['roll_no']})",
            message=f"Attendance warning issued for {student['name']}. Current attendance is {pct}%. Needs to attend {needed} consecutive classes before end of term.",
            n_type="warning",
            student_id=int(student_id)
        )
        db.log_audit("SHORTAGE_WARNING_ISSUED", session.get("username", "Faculty"),
                     f"Issued official attendance shortage alert for {student['name']} ({student['roll_no']})")

        return jsonify({
            "status": "success",
            "message": f"Official shortage alert logged and notice issued for {student['name']} ({student['roll_no']})!"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================
# REST API: STUDENT PORTAL (Logged-In Student Only)
# ==========================================

@app.route("/api/portal/summary")
def api_portal_summary():
    """
    CRITICAL PRIVACY REQUIREMENT:
    Returns subject-wise attendance and marks strictly for the logged-in student.
    A student cannot see data of any other student.
    """
    user = get_current_user()
    if not user:
        return jsonify({"status": "error", "message": "Authentication required"}), 401

    if user.get("role") == "student":
        student_id = user.get("student_id")
        if not student_id:
            stu = db.get_student_by_roll_no(user["username"])
            student_id = stu["id"] if stu else None
    else:
        student_id = request.args.get("student_id", type=int)
        if not student_id:
            all_s = db.get_all_students()
            student_id = all_s[0]["id"] if all_s else None

    if not student_id:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    data = db.get_student_portal_data(student_id)
    return jsonify(data)


# ==========================================
# BACKUP & DIAGNOSTICS
# ==========================================

@app.route("/api/backup")
def api_backup():
    """Generates and downloads full system backup."""
    backup_file = db.create_database_backup()
    return send_file(backup_file, as_attachment=True, download_name=os.path.basename(backup_file))


@app.route("/api/health")
def api_health():
    """System health check & diagnostics endpoint."""
    conn = db.get_db_connection()
    stu_cnt = conn.execute("SELECT COUNT(*) as c FROM students").fetchone()["c"]
    att_cnt = conn.execute("SELECT COUNT(*) as c FROM attendance_logs").fetchone()["c"]
    conn.close()

    return jsonify({
        "status": "healthy",
        "server_time": datetime.datetime.now().isoformat(),
        "database": "eduface.db",
        "enrolled_students": stu_cnt,
        "attendance_records": att_cnt,
        "models": {
            "yunet": os.path.exists(os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")),
            "sface": os.path.exists(os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"))
        }
    })


# ==========================================
# HARDWARE VIDEO FEED (MJPEG)
# ==========================================

class VideoCamera:
    def __init__(self):
        self.cap = None

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.start()
        if not self.cap or not self.cap.isOpened():
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        threshold = float(db.get_setting("similarity_threshold", "0.42"))
        detections = face_engine.process_frame(frame, cached_profiles, threshold=threshold)

        for det in detections:
            bbox = det["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            color = (0, 200, 0) if det["is_recognized"] else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = "Unknown"
            if det["is_recognized"] and det["student"]:
                label = f"{det['student']['name']} ({det['similarity']*100:.0f}%)"
                db.record_attendance(det["student"]["id"], confidence=det["similarity"], scan_mode="Server Webcam")

            cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


server_camera = VideoCamera()


def generate_frames():
    while True:
        frame_bytes = server_camera.get_frame()
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print("🚀 EduFace AI - Institutional Biometric Attendance Server")
    print(f"📡 Web Application running at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=True)
