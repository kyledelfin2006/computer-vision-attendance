import cv2
import os
import csv
import re
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Constants
MODEL_PATH = "model.h5"
ATT_VIDEO_WIDTH = 640
ATT_VIDEO_HEIGHT = 480

def create_session(name):
    # Simulate creating a session and return a unique ID
    return len(get_all_sessions()) + 1

def end_session(session_id):
    # Simulate ending a session
    pass

def get_person_id_by_name(name):
    # Simulate getting person ID by name
    return None

def log_attendance(pid, session_id):
    # Simulate logging attendance
    return True

def get_all_sessions():
    # Simulate retrieving all sessions
    return [(1, "Session 1"), (2, "Session 2")]

def get_session_by_id(session_id):
    # Simulate retrieving a session by ID
    for sid, name in get_all_sessions():
        if sid == session_id:
            return (sid, name)
    return None

def get_attendance_for_session(session_id):
    # Simulate retrieving attendance records for a session
    return [("John Doe", "2023-10-01 14:30:00"), ("Jane Smith", "2023-10-01 15:00:00")]

def get_connection():
    # Simulate getting database connection
    pass

class FaceManager:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        # Load the face recognition model
        self.model = cv2.dnn.readNetFromONNX(MODEL_PATH)

    def detect_faces(self, frame):
        # Detect faces in the frame
        pass

    def get_face_roi(self, frame, box):
        # Get region of interest for a detected face
        pass

    def recognize_face(self, roi):
        # Recognize a face from the ROI
        return "Unknown", 0.5

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        # Initialize components
        self.init_components()

        # Start camera thread
        self.camera_running = True
        self.cap = cv2.VideoCapture(0)
        self.face_manager = FaceManager()
        self.start_camera_thread()

    def init_components(self):
        # Create and configure UI elements
        self.session_name_entry = tk.Entry(self.root, width=30)
        self.session_name_entry.pack(pady=10)

        self.attendance_active = False
        self.current_session_id = None
        self.current_session_name = None
        self.current_session_start_time = None
        self.current_session_csv = None

        self.start_btn = tk.Button(self.root, text="Start Session", command=self.start_session)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(self.root, text="Stop Session", state=tk.DISABLED, command=lambda: self.stop_session(save_csv=True))
        self.stop_btn.pack(pady=5)

        self.att_status = tk.Label(self.root, text="", fg="green")
        self.att_status.pack(pady=10)

        self.start_time_label = tk.Label(self.root, text="Start Time: --", fg="blue")
        self.start_time_label.pack(pady=5)

        self.end_time_label = tk.Label(self.root, text="End Time: --", fg="blue")
        self.end_time_label.pack(pady=5)

        self.att_video_label = tk.Label(self.root)
        self.att_video_label.pack(pady=10)

        self.session_combo = ttk.Combobox(self.root, values=[], state=tk.DISABLED)
        self.session_combo.pack(pady=5)

        self.export_btn = tk.Button(self.root, text="Export CSV", command=self.export_csv)
        self.export_btn.pack(pady=5)

        self.reset_btn = tk.Button(self.root, text="Reset Database", command=self.reset_database)
        self.reset_btn.pack(pady=5)

    def start_camera_thread(self):
        # Start a thread to process video frames
        import threading

        def process_frames():
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.process_attendance_frame(frame)
                cv2.imshow("Attendance", frame)

        camera_thread = threading.Thread(target=process_frames)
        camera_thread.start()

    def process_attendance_frame(self, frame):
        # Process each frame to detect and recognize faces
        boxes = self.face_manager.detect_faces(frame)
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = self.face_manager.get_face_roi(frame, box)
            if roi is None:
                continue
            name, confidence = self.face_manager.recognize_face(roi)

            status_text = "Unknown"
            color = (0, 0, 255)  # blue

            pid = get_person_id_by_name(name) if name else None

            if pid and self.current_session_id:
                if log_attendance(pid, self.current_session_id):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.append_attendance_csv(name, timestamp)
                    status_text, color = "Registered in session", (0, 255, 0)  # green

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{name if name else 'Unknown'}: {status_text}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.display_frame(self.att_video_label, frame, ATT_VIDEO_WIDTH, ATT_VIDEO_HEIGHT)

    def display_frame(self, label, frame, width, height):
        # Display the processed frame in a label
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image.resize((width, height)))
        label.config(image=photo)
        label.image = photo

    def start_session(self):
        name = self.session_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Enter a session name", "Please enter a session name")
            return
        if self.attendance_active:
            return
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_session_id = create_session(name)
        self.current_session_name = name
        self.current_session_start_time = start_time
        self.attendance_feedback_until.clear()
        self.current_session_csv = self.write_session_csv(
            self.current_session_id,
            name,
            [],
            start_time=start_time
        )
        self.attendance_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.start_time_label.config(text=f"Start Time: {start_time}")
        self.end_time_label.config(text="End Time: --")
        self.att_status.config(text=f"Session started: {name}\nWaiting for registered users.", fg="green")
        self.refresh_sessions()

    def stop_session(self, save_csv=True):
        if self.attendance_active and self.current_session_id:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            end_session(self.current_session_id)
            if save_csv:
                records = get_attendance_for_session(self.current_session_id)
                self.current_session_csv = self.write_session_csv(
                    self.current_session_id,
                    self.current_session_name or "session",
                    records,
                    start_time=self.current_session_start_time,
                    end_time=end_time
                )
            self.attendance_active = False
            self.current_session_id = None
            self.current_session_name = None
            self.current_session_start_time = None
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.end_time_label.config(text=f"End Time: {end_time}")
            if save_csv:
                self.att_status.config(text=f"Session stopped.\nCSV saved.", fg="blue")
            self.refresh_sessions()

    def write_session_csv(self, session_id, session_name, records, start_time=None, end_time=None):
        # Write attendance records to a CSV file
        os.makedirs("exports", exist_ok=True)
        filename = os.path.join("exports", f"session_{session_id}_{self.safe_filename(session_name)}.csv")
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Session ID", session_id])
            writer.writerow(["Session Name", session_name])
            writer.writerow(["Start Time", start_time or ""])
            writer.writerow(["End Time", end_time or ""])
            writer.writerow([])
            writer.writerow(["Name", "Timestamp"])
            writer.writerows(records)
        return filename

    def append_attendance_csv(self, name, timestamp):
        # Append attendance record to the CSV file
        if not self.current_session_csv:
            return
        with open(self.current_session_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])

    def refresh_sessions(self):
        # Refresh session list in the combo box
        sessions = get_all_sessions()
        self.session_combo.config(values=[f"{sid} - {name}" for sid, name in sessions])
        self.session_combo.set("")

    def export_csv(self):
        # Export attendance records to a CSV file
        if not self.current_session_csv:
            messagebox.showwarning("No session active", "Please start a session first")
            return
        filename = os.path.join("exports", f"session_{self.current_session_id}_{self.safe_filename(self.current_session_name)}.csv")
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            records = list(csv.reader(f))
        messagebox.showinfo("Export CSV", f"Attendance records exported to {filename}")

    def reset_database(self):
        # Reset the database
        pass

    def safe_filename(self, filename):
        # Sanitize the filename for use in a file path
        return re.sub(r'[^a-zA-Z0-9\s]', '', filename).strip()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()