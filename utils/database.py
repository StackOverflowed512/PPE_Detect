import sqlite3
import pickle

def init_db():
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    
    # Drop existing table to ensure clean schema
    cursor.execute('DROP TABLE IF EXISTS persons')
    
    # Create table with correct schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL,
            age INTEGER,
            function_text TEXT,
            person_id_code TEXT UNIQUE,
            hashcode TEXT,
            image_filename TEXT,
            image_data BLOB NOT NULL,
            face_encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_person_to_db(display_name, age, function_text, person_id_code, hashcode, image_filename, image_data, face_encoding):
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO persons 
            (display_name, age, function_text, person_id_code, hashcode, image_filename, image_data, face_encoding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (display_name, age, function_text, person_id_code, hashcode, 
              image_filename, image_data, pickle.dumps(face_encoding)))
        conn.commit()
    except sqlite3.IntegrityError:
        raise Exception(f"Person ID Code '{person_id_code}' already exists in the database.")
    finally:
        conn.close()

def get_all_persons():
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, display_name, age, function_text, person_id_code 
        FROM persons
    ''')
    persons = cursor.fetchall()
    conn.close()
    return persons

def get_person_image(person_id):
    conn = sqlite3.connect('personnel_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT image_data FROM persons WHERE id = ?', (person_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None