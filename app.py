import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import datetime
from PIL import Image
import face_recognition
import numpy as np
import json
from ultralytics import YOLO
import sqlite3
import pickle
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Response, jsonify
from utils.detection import detect_faces_and_ppe, load_known_faces_from_db

# Global variables
DATABASE_NAME = "personnel_data.db"
REGISTERED_IMAGES_DIR = "registered_images"
CONFIG = None
yolo_ppe_model = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = REGISTERED_IMAGES_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    global CONFIG
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found")
            
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
        print("Configuration loaded successfully")
        return CONFIG
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        raise

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS persons
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      display_name TEXT NOT NULL,
                      age INTEGER,
                      function_text TEXT,
                      person_id_code TEXT UNIQUE,
                      hashcode TEXT,
                      image_filename TEXT,
                      face_encoding BLOB)''')
    conn.commit()
    conn.close()
    print("Database initialized")

def initialize_ppe_model():
    """Initialize YOLO model for PPE detection"""
    global yolo_ppe_model, CONFIG
    
    if CONFIG is None:
        CONFIG = load_config()
    
    model_path = CONFIG.get("yolov8_ppe_model_path")
    if not model_path or not os.path.exists(model_path):
        print(f"Warning: YOLO model not found at {model_path}")
        return
        
    try:
        yolo_ppe_model = YOLO(model_path)
        print("PPE model initialized")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        yolo_ppe_model = None

def init_app():
    """Initialize all application components"""
    global CONFIG
    
    # Create required directories
    os.makedirs(REGISTERED_IMAGES_DIR, exist_ok=True)
    os.makedirs('static/registered_images', exist_ok=True)
    
    # Load configuration
    CONFIG = load_config()
    print("Config loaded:", CONFIG is not None)
    
    # Initialize database
    init_db()
    
    # Initialize PPE model
    initialize_ppe_model()
    print("PPE model loaded:", yolo_ppe_model is not None)
    
    # Load known faces from database
    load_known_faces_from_db()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Process registration form
            data = request.form
            image_file = request.files['image']
            
            if not image_file:
                return jsonify({'error': 'No image uploaded'}), 400
                
            filename = secure_filename(f"{data['display_name']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            
            # Load image and generate face encoding
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                os.remove(filepath)
                return jsonify({'error': 'No face detected in image'}), 400
                
            face_encoding = encodings[0]
            
            # Add to database
            conn = sqlite3.connect(DATABASE_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO persons 
                (display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['display_name'],
                int(data['age']) if data['age'] else None,
                data['function'],
                data['person_id'],
                data['hashcode'],
                filename,
                pickle.dumps(face_encoding)
            ))
            conn.commit()
            conn.close()
            
            # Reload known faces
            load_known_faces_from_db()
            
            return jsonify({'success': True, 'filename': filename})
            
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Person ID already exists'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return render_template('register.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/persons')
def persons():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT display_name, age, function_text, person_id_code, image_filename 
        FROM persons
    ''')
    persons = cursor.fetchall()
    conn.close()
    return render_template('persons.html', persons=persons)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
        
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame")
            break
            
        try:
            # Process frame with detection
            processed_frame = detect_faces_and_ppe(frame)
            
            # Convert to jpeg
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            break
            
    camera.release()

if __name__ == '__main__':
    try:
        init_app()
        print("Application initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting application: {e}")