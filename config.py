"""
Configuration constants for EduFace AI
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "eduface.db")

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
SECRET_KEY = "eduface-ai-institutional-attendance-secret-key"
