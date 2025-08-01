from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import numpy as np
from io import BytesIO
import sqlite3
import face_recognition
import pickle
import os
from utils.database import init_db, add_person_to_db, get_all_persons, get_person_image
from utils.detection import detect_faces_and_ppe, load_known_faces
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database and models on startup
init_db()
load_known_faces()

# Load YOLO model
try:
    yolo_model = YOLO("models/ppe_custom_yolov8.pt")
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            display_name = request.form['display_name']
            person_id = request.form['person_id']
            image_file = request.files['image']
            
            if not image_file:
                return jsonify({'error': 'No image uploaded'}), 400
                
            # Read image data
            image_data = image_file.read()
            
            # Load image and generate face encoding
            image = face_recognition.load_image_file(BytesIO(image_data))
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                return jsonify({'error': 'No face detected in image'}), 400
                
            face_encoding = encodings[0]
            
            # Add to database
            add_person_to_db(
                display_name,
                int(request.form['age']) if request.form['age'] else None,
                request.form['function'],
                person_id,
                request.form.get('hashcode', ''),
                image_file.filename,
                image_data,
                face_encoding
            )
            
            # Reload known faces after new registration
            load_known_faces()
            
            return jsonify({'success': True})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return render_template('register.html')

@app.route('/persons')
def persons():
    persons = get_all_persons()
    return render_template('persons.html', persons=persons)

@app.route('/person_image/<int:person_id>')
def person_image(person_id):
    image_data = get_person_image(person_id)
    if image_data:
        return send_file(BytesIO(image_data), mimetype='image/jpeg')
    return Response(status=404)

@app.route('/detection')
def detection():
    return render_template('detection.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame with detection
            processed_frame = detect_faces_and_ppe(frame, yolo_model)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
