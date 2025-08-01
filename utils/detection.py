import cv2
import numpy as np
import face_recognition
import sqlite3
import pickle
import json

known_face_encodings = []
known_face_names = []
known_face_info = []

def load_known_faces():
    global known_face_encodings, known_face_names, known_face_info
    
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT display_name, age, function_text, person_id_code, face_encoding 
        FROM persons
    ''')
    
    known_face_encodings = []
    known_face_names = []
    known_face_info = []
    
    for person in cursor.fetchall():
        try:
            face_encoding = pickle.loads(person[4])
            known_face_encodings.append(face_encoding)
            known_face_names.append(person[0])
            known_face_info.append({
                'name': person[0],
                'age': person[1],
                'function': person[2],
                'id': person[3]
            })
        except Exception as e:
            print(f"Error loading face encoding for {person[0]}: {e}")
    
    conn.close()
    print(f"Loaded {len(known_face_names)} known faces from database")

def detect_faces_and_ppe(frame, yolo_model=None):
    if frame is None:
        return frame
    
    # Convert BGR to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # PPE detection with YOLO if model is available
    if yolo_model is not None:
        try:
            results = yolo_model(frame, conf=0.35)[0]
            
            # Draw PPE detections
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                label = yolo_model.names[int(cls)]
                
                # Get color from config
                with open("config.json") as f:
                    config = json.load(f)
                color = tuple(config["ppe_class_colors"].get(label, config["ppe_class_colors"]["default"]))
                
                # Draw bounding box
                cv2.rectangle(frame, 
                             (int(x1), int(y1)), 
                             (int(x2), int(y2)), 
                             color, 2)
                
                # Add label
                cv2.putText(frame, 
                            f"{label} {conf:.2f}", 
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, color, 2)
        except Exception as e:
            print(f"Error in PPE detection: {e}")
    
    # Draw face detections
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        info = ""
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            info = f"{known_face_info[first_match_index]['function']} (ID: {known_face_info[first_match_index]['id']})"
        
        # Draw face box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw additional info if available
        if info:
            cv2.putText(frame, info, (left, bottom + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

# Initialize known faces on import
load_known_faces()