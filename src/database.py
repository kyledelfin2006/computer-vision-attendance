import sqlite3
from datetime import datetime

# Handles database connection, persistence, SQL Queries

DB_NAME = "database.db"

def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)

def init_db():
    """Create the required tables if they don't exist."""
    with get_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS persons (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT UNIQUE NOT NULL,
                            reg_date DATETIME DEFAULT CURRENT_TIMESTAMP
                        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                            end_time DATETIME
                        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            person_id INTEGER NOT NULL,
                            session_id INTEGER NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (person_id) REFERENCES persons(id),
                            FOREIGN KEY (session_id) REFERENCES sessions(id)
                        )''')

def add_person(name):
    """
    Insert a new person into the database.
    Returns the new person's ID, or None if the name already exists.
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None  # name already exists

def get_person_id_by_name(name):
    """Return the person ID for a given name, or None if not found."""
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

def get_person_name_by_id(pid):
    """Return the person's name for a given ID, or None."""
    with get_connection() as conn:
        row = conn.execute("SELECT name FROM persons WHERE id=?", (pid,)).fetchone()
        return row[0] if row else None

def get_all_persons():
    """Return a list of (id, name) for all registered persons."""
    with get_connection() as conn:
        return conn.execute("SELECT id, name FROM persons ORDER BY id").fetchall()

def create_session(name):
    """Create a new attendance session and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        return cursor.lastrowid

def end_session(session_id):
    """Update the session's end_time to the current time."""
    with get_connection() as conn:
        conn.execute("UPDATE sessions SET end_time=? WHERE id=?",
                     (datetime.now(), session_id))

def log_attendance(person_id, session_id):
    """
    Log attendance for a person in a session.
    Returns True if logged, False if already logged in this session.
    """
    with get_connection() as conn:
        # Check for existing attendance in this session
        existing = conn.execute(
            "SELECT id FROM attendance WHERE person_id=? AND session_id=?",
            (person_id, session_id)
        ).fetchone()
        if existing:
            return False
        conn.execute(
            "INSERT INTO attendance (person_id, session_id, timestamp) VALUES (?, ?, ?)",
            (person_id, session_id, datetime.now())
        )
        return True

def get_attendance_for_session(session_id):
    """Return a list of (name, timestamp) for all attendance records of a session."""
    with get_connection() as conn:
        return conn.execute('''
            SELECT p.name, a.timestamp
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
            WHERE a.session_id = ?
            ORDER BY a.timestamp
        ''', (session_id,)).fetchall()

def get_all_sessions():
    """Return a list of (id, name, start_time, end_time) for all sessions,
       ordered most recent first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, name, start_time, end_time FROM sessions ORDER BY start_time DESC"
        ).fetchall()