@echo off
title EduFace AI - School & College Face Recognition Attendance System
cls
echo =====================================================================
echo    EduFace AI - Institutional Biometric Attendance System
echo =====================================================================
echo.
echo Checking Python environment...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)

echo Starting EduFace AI Web Application Server...
echo Opening browser at: http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause
