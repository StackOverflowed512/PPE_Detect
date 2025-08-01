import sqlite3
import pickle

def init_db():
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL,
            age INTEGER,
            function_text TEXT,
            person_id_code TEXT UNIQUE,
            hashcode TEXT,
            image_filename TEXT,
            face_encoding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def add_person_to_db(display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding):
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO persons 
        (display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (display_name, age, function_text, person_id_code, hashcode, 
          image_filename, pickle.dumps(face_encoding)))
    conn.commit()
    conn.close()

def get_all_persons():
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT display_name, age, function_text, person_id_code, image_filename 
        FROM persons
    ''')
    persons = cursor.fetchall()
    conn.close()
    return persons