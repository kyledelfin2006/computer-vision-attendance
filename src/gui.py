import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import os
import csv
import shutil
from datetime import datetime

from database import (
    init_db, create_session, end_session, log_attendance,
    get_all_sessions, get_attendance_for_session,
    get_person_id_by_name, get_person_name_by_id,
    get_connection
)
from face_utils import FaceManager


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("900x700")

        # Initialize database
        init_db()

        # Create necessary folders
        os.makedirs("images", exist_ok=True)
        os.makedirs("models/dnn", exist_ok=True)
        os.makedirs("exports", exist_ok=True)

        # Face manager (DNN + LBPH)
        self.face_manager = FaceManager()

        # Camera and state
        self.cap = None
        self.camera_running = False
        self.attendance_active = False
        self.current_session_id = None
        self.register_frames = []

        # Build GUI
        self._build_ui()

        # Start camera
        self.start_camera()

    # ---------------------- UI Construction ----------------------
    def _build_ui(self):
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ---- Tab 1: Registration ----
        self.reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reg_frame, text="Registration")

        # Video display
        self.reg_video_label = tk.Label(self.reg_frame)
        self.reg_video_label.pack(pady=10)

        # Controls
        ctrl_reg = tk.Frame(self.reg_frame)
        ctrl_reg.pack(pady=10)
        tk.Label(ctrl_reg, text="Name:").grid(row=0, column=0, padx=5)
        self.reg_name_entry = tk.Entry(ctrl_reg, width=20)
        self.reg_name_entry.grid(row=0, column=1, padx=5)
        self.reg_btn = tk.Button(ctrl_reg, text="Register", command=self.register_person)
        self.reg_btn.grid(row=0, column=2, padx=5)
        self.reset_btn = tk.Button(ctrl_reg, text="Wipe Entire Database", command=self.reset_database, bg="orange")
        self.reset_btn.grid(row=0, column=3, padx=5)
        self.reg_status = tk.Label(self.reg_frame, text="", font=("Arial", 10))
        self.reg_status.pack(pady=5)

        # ---- Tab 2: Attendance ----
        self.att_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.att_frame, text="Attendance")

        # Video display
        self.att_video_label = tk.Label(self.att_frame)
        self.att_video_label.pack(pady=10)

        # Controls
        ctrl_att = tk.Frame(self.att_frame)
        ctrl_att.pack(pady=10)
        tk.Label(ctrl_att, text="Session Name:").grid(row=0, column=0, padx=5)
        self.session_name_entry = tk.Entry(ctrl_att, width=20)
        self.session_name_entry.grid(row=0, column=1, padx=5)
        self.start_btn = tk.Button(ctrl_att, text="Start Session", command=self.start_session)
        self.start_btn.grid(row=0, column=2, padx=5)
        self.stop_btn = tk.Button(ctrl_att, text="Stop Session", command=self.stop_session, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=3, padx=5)
        self.att_status = tk.Label(self.att_frame, text="", font=("Arial", 10))
        self.att_status.pack(pady=5)

        # Export section
        export_frame = tk.Frame(self.att_frame)
        export_frame.pack(pady=10)
        tk.Label(export_frame, text="Export Session:").pack(side=tk.LEFT, padx=5)
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(export_frame, textvariable=self.session_var, state="readonly")
        self.session_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(export_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        self.refresh_sessions()

    # ---------------------- Camera & Video ----------------------
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
        self.camera_running = True
        self.update_video()

    def update_video(self):
        if not self.camera_running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Mirror horizontally (self‑view effect)
            frame = cv2.flip(frame, 1)

            # ---------- Registration Tab: always show detection boxes ----------
            reg_frame = self.process_registration_preview(frame.copy())
            self.display_frame(self.reg_video_label, reg_frame)

            # ---------- Attendance Tab ----------
            if self.attendance_active:
                # Process for attendance (draws boxes, logs, etc.)
                self.process_attendance_frame(frame)
            else:
                # Just show raw (or we could show detection here too, but keep it clean)
                self.display_frame(self.att_video_label, frame)

        self.root.after(30, self.update_video)

    def process_registration_preview(self, frame):
        """
        Detect faces and draw boxes with name if known, else 'Face Detected'.
        This is shown in the Registration tab to give immediate feedback.
        """
        boxes = self.face_manager.detect_faces(frame)
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = self.face_manager.get_face_roi(frame, box)
            name, _ = self.face_manager.recognize_face(roi) if roi is not None else (None, None)
            label = name if name else "Face Detected"
            color = (0, 255, 0)  # green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def display_frame(self, label, frame):
        """Display an OpenCV frame on a Tkinter Label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    # ---------------------- Registration ----------------------
    def register_person(self):
        name = self.reg_name_entry.get().strip()
        if not name:
            self.reg_status.config(text="Please enter a name", fg="red")
            return
        if get_person_id_by_name(name):
            self.reg_status.config(text=f"Name '{name}' already registered", fg="red")
            return

        self.reg_btn.config(state=tk.DISABLED)
        self.reg_status.config(text="Capturing faces... Please move your head slowly.", fg="blue")
        self.root.update()

        # Show instruction on the video feed for a moment
        instruction_frame = None
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Overlay instruction
            cv2.putText(frame, "Please slowly move your head", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(frame, "left and right for the next 5 seconds", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            self.display_frame(self.reg_video_label, frame)
            self.root.update()
            time.sleep(1.5)  # Show instruction for 1.5 seconds

        # Capture up to 20 frames over 5 seconds
        collected = []
        start_time = time.time()
        capture_interval = 0.25  # seconds between captures
        last_capture_time = start_time
        count = 0

        while time.time() - start_time < 5.0 and self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            boxes = self.face_manager.detect_faces(frame)

            # Draw all detected faces (just for feedback)
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Capture a face crop at intervals
            if time.time() - last_capture_time >= capture_interval and boxes:
                # Take the first detected face for simplicity (or could take largest)
                box = boxes[0]
                roi = self.face_manager.get_face_roi(frame, box)
                if roi is not None:
                    collected.append(roi)
                    count += 1
                    last_capture_time = time.time()
                    # Update progress on the frame
                    cv2.putText(frame, f"Captured {count}/20", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show remaining time
            remaining = max(0, 5.0 - (time.time() - start_time))
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Display the frame
            self.display_frame(self.reg_video_label, frame)
            self.root.update()

            if count >= 20:
                break

        # If not enough crops, ask to try again
        if len(collected) < 10:
            self.reg_status.config(text="Not enough face images captured. Please try again.", fg="red")
            self.reg_btn.config(state=tk.NORMAL)
            return

        # Register the person (trains model)
        pid = self.face_manager.register_person(name, collected)
        if pid:
            self.reg_status.config(text=f"Registered '{name}' with ID {pid}!", fg="green")
            self.reg_name_entry.delete(0, tk.END)
            self.refresh_sessions()
        else:
            self.reg_status.config(text="Registration failed (duplicate name or error)", fg="red")
        self.reg_btn.config(state=tk.NORMAL)

    # ---------------------- Attendance ----------------------
    def start_session(self):
        name = self.session_name_entry.get().strip()
        if not name:
            self.att_status.config(text="Enter a session name", fg="red")
            return
        if self.attendance_active:
            return
        # Create session in DB
        self.current_session_id = create_session(name)
        self.attendance_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.att_status.config(text=f"Session '{name}' started", fg="green")

    def stop_session(self):
        if self.attendance_active and self.current_session_id:
            end_session(self.current_session_id)
            self.attendance_active = False
            self.current_session_id = None
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.att_status.config(text="Session stopped", fg="blue")
            self.refresh_sessions()

    def process_attendance_frame(self, frame):
        """Detect faces, recognise, and log attendance."""
        boxes = self.face_manager.detect_faces(frame)
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = self.face_manager.get_face_roi(frame, box)
            if roi is None:
                continue
            name, confidence = self.face_manager.recognize_face(roi)

            status_text = ""
            color = (0, 0, 255)  # red
            if name:
                pid = get_person_id_by_name(name)
                if pid and self.current_session_id:
                    logged = log_attendance(pid, self.current_session_id)
                    if logged:
                        status_text = "Successfully registered"
                        color = (0, 255, 0)  # green
                    else:
                        status_text = "Already logged"
                        color = (0, 255, 255)  # yellow
                else:
                    status_text = "Unknown"
                    color = (0, 0, 255)
            else:
                status_text = "Unknown"
                color = (0, 0, 255)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name if name else 'Unknown'}: {status_text}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the processed frame in the attendance tab
        self.display_frame(self.att_video_label, frame)

    # ---------------------- Export CSV ----------------------
    def refresh_sessions(self):
        sessions = get_all_sessions()
        self.session_combo['values'] = [f"{s[1]} (ID:{s[0]})" for s in sessions]
        if sessions:
            self.session_combo.current(0)

    def export_csv(self):
        selection = self.session_combo.get()
        if not selection:
            messagebox.showwarning("No selection", "Please select a session")
            return
        try:
            sid = int(selection.split("ID:")[1].split(")")[0])
        except:
            messagebox.showerror("Error", "Invalid session selection")
            return

        records = get_attendance_for_session(sid)
        if not records:
            messagebox.showinfo("Empty", "No attendance records for this session")
            return

        session_name = selection.split(" (ID:")[0]
        filename = f"exports/{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])
            writer.writerows(records)
        messagebox.showinfo("Export", f"Exported to {filename}")

    # ---------------------- Reset / Wipe All Data ----------------------
    def reset_database(self):
        """Clear all data, delete images/model, and reload everything."""
        if not messagebox.askyesno("Reset Database", "Delete all persons, sessions, attendance, images, and model?"):
            return

        # Stop attendance if active
        if self.attendance_active:
            self.stop_session()

        # Empty all tables
        with get_connection() as conn:
            conn.execute("DELETE FROM attendance")
            conn.execute("DELETE FROM persons")
            conn.execute("DELETE FROM sessions")
            # Reset auto-increment counters
            conn.execute("DELETE FROM sqlite_sequence")

        # Delete images folder and model
        if os.path.exists("images"):
            shutil.rmtree("images")
        os.makedirs("images", exist_ok=True)

        if os.path.exists("models/face_model.yml"):
            os.remove("models/face_model.yml")

        # Reload the face manager (empty model)
        self.face_manager.load_model()

        # Refresh UI
        self.refresh_sessions()
        self.reg_status.config(text="All data wiped", fg="blue")
        self.att_status.config(text="Database reset", fg="blue")
        self.reg_name_entry.delete(0, tk.END)

    # ---------------------- Cleanup ----------------------
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()