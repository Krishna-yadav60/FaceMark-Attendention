# 🎓 EduFace AI &bull; School & College Face Recognition Attendance System

An enterprise-ready, institutional biometric face recognition attendance management web application and software designed for schools, colleges, and universities.

Built with **Python**, **OpenCV DNN Deep Learning (YuNet + SFace)**, **Flask**, **SQLite**, and modern **Tailwind CSS**.

---

## 🌟 Full-Stack Architecture

### 1. 🛡️ User Authentication & Role-Based Access Control (RBAC)
- **Roles**: System Administrator (`admin`), Academic Faculty/Instructor (`faculty`), and Kiosk Terminal (`kiosk`).
- **Security**: Cryptographically salted password hashing via Werkzeug (`generate_password_hash` & `check_password_hash`).
- **Default Accounts**:
  - **Admin**: `admin` / `admin123`
  - **Faculty**: `faculty` / `faculty123`
- Session-based cookie authentication with protected API endpoints and audit trail logging.

### 2. ⚡ Real-Time Push Events (Server-Sent Events - SSE)
- Low-latency backend event bus (`services/event_stream.py`).
- Whenever a student is verified at any kiosk or web camera, a real-time event (`student_verified`) is pushed instantly to all connected browser dashboards and faculty monitors.
- Dynamic counter updates, live table row insertion, and synthesized acoustic chimes trigger without refreshing the page!

### 3. 📚 Course & Lecture Timetable Attendance Engine
- Academic course catalog with departments, semester levels, credits, and assigned professors.
- Lecture slot scheduling: create time slots (e.g. `10:00 AM - 11:00 AM`) in specific classrooms.
- Biometric Lecture Mode: Take face recognition attendance for specific classes with dedicated per-course attendance ledgers.

### 4. 🧠 Automated Absenteeism & Student Risk Intelligence
- **Absenteeism Engine**: Automatically evaluates all enrolled students who have not checked in for the day (`/api/absentees`).
- **At-Risk Indicator**: Flags students whose attendance percentage falls below standard university minimums (<75%) with warning badges.
- **Hourly Rush-Hour Analytics**: Calculates arrival distributions across 24-hour time slots.

### 5. 🔄 Bulk Ingestion, Folder Sync & One-Click Backup
- **Folder Auto-Sync**: Ingests student photos directly placed in `images/` (e.g. `105_John_Doe.jpg`) and computes 128D face embeddings automatically.
- **Bulk CSV Importer**: Ingests whole student rosters from CSV spreadsheets (`/api/students/import_csv`).
- **One-Click Backup**: Creates a complete, timestamped ZIP archive containing the SQLite database and all enrolled photo assets (`/api/backup`).
- **Health Check & Diagnostics**: Live system monitoring endpoint (`/api/health`).

---

## 🧠 AI Neural Network Architecture

| Component | Model | Architecture | Role |
| :--- | :--- | :--- | :--- |
| **Face Detection** | YuNet (2023Mar ONNX) | Ultra-lightweight DNN | Detects face bounding boxes + 5 facial landmarks (eyes, nose, mouth corners) |
| **Face Recognition**| SFace (2021Dec ONNX) | Deep Metric Learning | Extracts 128-dimensional L2-normalized feature embeddings |
| **Metric Matching** | Cosine Similarity | Vector Space Dot Product | Matches query faces against stored embeddings (Baseline >= 0.363, optimal threshold = 0.42) |

*Zero dlib, CMake, or C++ MSVC compilation required. Runs natively via OpenCV DNN C++ backend at 60+ FPS on standard CPUs.*

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/recognize_frame` | `POST` | Receives base64 camera frame, detects face, matches student, and records attendance. |
| `/api/students` | `GET`, `POST` | Lists all students or registers a new student with photo and 128D embedding. |
| `/api/students/<id>` | `GET`, `PUT`, `DELETE`| Inspects student profile + personal attendance ledger, updates, or deletes. |
| `/api/students/import_csv` | `POST` | Bulk imports student roster from a CSV file. |
| `/api/sync_folder` | `POST` | Scans `images/` directory and auto-enrolls new photos. |
| `/api/courses` | `GET`, `POST` | Fetches course directory or creates a new course/subject. |
| `/api/courses/<id>/enroll` | `POST` | Enrolls a student into a specific course. |
| `/api/sessions` | `GET`, `POST` | Schedules and lists lecture session slots. |
| `/api/attendance` | `GET` | Fetches filtered attendance logs by date, department, or status. |
| `/api/attendance/manual` | `POST` | Instructor manual override or excused absence mark. |
| `/api/attendance/export` | `GET` | Generates downloadable CSV attendance report. |
| `/api/events` | `GET` | Server-Sent Events (SSE) live push stream. |
| `/api/absentees` | `GET` | Lists all students currently absent today. |
| `/api/analytics/executive` | `GET` | Complete executive turnout report for principals and deans. |
| `/api/backup` | `GET` | Downloads full system database + photo backup ZIP. |
| `/api/health` | `GET` | Server diagnostics, database record counts, and model status. |

---

## 🚀 Quick Start Guide

### Option 1: Double-Click Launcher (Windows)
Double-click `run.bat` in the project directory. It launches the application and opens your browser automatically to `http://127.0.0.1:5000`.

### Option 2: Terminal / Command Prompt
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch EduFace AI Web Application
python main.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 🧪 Automated System Verification

Run the full end-to-end unit and integration test suite:
```bash
python -m unittest tests/test_system.py
```
All 11 tests pass:
```text
Ran 11 tests in 0.906s
OK
```
