import csv
import os
import shutil
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk
from database import (
    init_db, create_session, end_session, log_attendance,
    get_all_sessions, get_attendance_for_session,
    get_person_id_by_name, get_connection
)
from face_utils import FaceManager

# Fixed video display size
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("1000x750")

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

        self.reg_video_label = tk.Label(self.reg_frame)
        self.reg_video_label.pack(pady=5, fill=tk.BOTH, expand=True)  # reduced pady

        # Controls – placed closer to video
        ctrl_reg = tk.Frame(self.reg_frame)
        ctrl_reg.pack(pady=5)  # less padding
        # Larger fonts and wider elements
        tk.Label(ctrl_reg, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5)
        self.reg_name_entry = tk.Entry(ctrl_reg, width=30, font=("Arial", 12))  # wider
        self.reg_name_entry.grid(row=0, column=1, padx=10, pady=5)

        # Bigger buttons
        self.reg_btn = tk.Button(ctrl_reg, text="Register", command=self.register_person,
                                 font=("Arial", 12, "bold"), width=12, height=1)
        self.reg_btn.grid(row=0, column=2, padx=10, pady=5)

        self.reset_btn = tk.Button(ctrl_reg, text="Wipe Entire Database", command=self.reset_database,
                                   bg="orange", font=("Arial", 12, "bold"), width=18, height=1)
        self.reset_btn.grid(row=0, column=3, padx=10, pady=5)

        self.reg_status = tk.Label(self.reg_frame, text="", font=("Arial", 12))
        self.reg_status.pack(pady=5)

        # ---- Tab 2: Attendance ----
        self.att_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.att_frame, text="Attendance")

        self.att_video_label = tk.Label(self.att_frame)
        self.att_video_label.pack(pady=5, fill=tk.BOTH, expand=True)  # reduced pady

        ctrl_att = tk.Frame(self.att_frame)
        ctrl_att.pack(pady=5)
        tk.Label(ctrl_att, text="Session Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5)
        self.session_name_entry = tk.Entry(ctrl_att, width=30, font=("Arial", 12))
        self.session_name_entry.grid(row=0, column=1, padx=10, pady=5)

        self.start_btn = tk.Button(ctrl_att, text="Start Session", command=self.start_session,
                                   font=("Arial", 12, "bold"), width=14, height=1)
        self.start_btn.grid(row=0, column=2, padx=10, pady=5)

        self.stop_btn = tk.Button(ctrl_att, text="Stop Session", command=self.stop_session,
                                  state=tk.DISABLED, font=("Arial", 12, "bold"), width=14, height=1)
        self.stop_btn.grid(row=0, column=3, padx=10, pady=5)

        self.att_status = tk.Label(self.att_frame, text="", font=("Arial", 12))
        self.att_status.pack(pady=5)

        # Export section
        export_frame = tk.Frame(self.att_frame)
        export_frame.pack(pady=5)
        tk.Label(export_frame, text="Export Session:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(export_frame, textvariable=self.session_var,
                                          state="readonly", font=("Arial", 12), width=30)
        self.session_combo.pack(side=tk.LEFT, padx=10)
        tk.Button(export_frame, text="Export CSV", command=self.export_csv,
                  font=("Arial", 12, "bold"), width=12, height=1).pack(side=tk.LEFT, padx=10)
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
            frame = cv2.flip(frame, 1)

            # Registration Tab: always show detection boxes
            reg_frame = self.process_registration_preview(frame.copy())
            self.display_frame(self.reg_video_label, reg_frame)

            # Attendance Tab
            if self.attendance_active:
                self.process_attendance_frame(frame)
            else:
                self.display_frame(self.att_video_label, frame)

        self.root.after(30, self.update_video)

    def process_registration_preview(self, frame):
        boxes = self.face_manager.detect_faces(frame)
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = self.face_manager.get_face_roi(frame, box)
            name, _ = self.face_manager.recognize_face(roi) if roi is not None else (None, None)
            label = name if name else "Face Detected"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def display_frame(self, label, frame):
        resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    # ---------------------- Registration (enhanced overlay) ----------------------
    def register_person(self):
        name = self.reg_name_entry.get().strip()
        if not name:
            self.reg_status.config(text="Please enter a name", fg="red")
            return
        if get_person_id_by_name(name):
            self.reg_status.config(text=f"Name '{name}' already registered", fg="red")
            return

        # Disable button and show status
        self.reg_btn.config(state=tk.DISABLED)
        self.reg_status.config(text="Capturing faces... Please move your head slowly.", fg="blue")
        self.root.update()

        collected = []
        start_time = time.time()
        capture_interval = 0.25   # seconds between captures
        last_capture_time = start_time
        count = 0
        capture_duration = 8.0    # 8 seconds

        while time.time() - start_time < capture_duration and self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            boxes = self.face_manager.detect_faces(frame)

            # Draw all detected faces (feedback)
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ---- Enhanced overlay (bigger, clearer, with background) ----
            h, w = frame.shape[:2]
            elapsed = time.time() - start_time
            remaining = max(0, capture_duration - elapsed)

            # Semi‑transparent background for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 160), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Big, bold text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Please slowly move your head left/right", (30, 55),
                        font, 1.0, (0, 255, 255), 3)
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (30, 100),
                        font, 1.0, (255, 255, 0), 3)
            cv2.putText(frame, f"Captured: {count}/20", (30, 145),
                        font, 1.0, (0, 255, 0), 3)

            # ---- Progress bar (visual timer) ----
            progress = 1.0 - (remaining / capture_duration)
            bar_x, bar_y, bar_w, bar_h = 30, 170, 300, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)

            # Capture a face crop at intervals
            if elapsed >= 0.5 and time.time() - last_capture_time >= capture_interval and boxes:
                box = boxes[0]  # take the first (largest) face
                roi = self.face_manager.get_face_roi(frame, box)
                if roi is not None:
                    collected.append(roi)
                    count += 1
                    last_capture_time = time.time()

            # Display the frame
            self.display_frame(self.reg_video_label, frame)
            self.root.update()

            if count >= 20:
                break

        # If not enough crops, abort
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name if name else 'Unknown'}: {status_text}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
        if not messagebox.askyesno("Reset Database", "Delete all persons, sessions, attendance, images, and model?"):
            return

        if self.attendance_active:
            self.stop_session()

        with get_connection() as conn:
            conn.execute("DELETE FROM attendance")
            conn.execute("DELETE FROM persons")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM sqlite_sequence")

        if os.path.exists("images"):
            shutil.rmtree("images")
        os.makedirs("images", exist_ok=True)

        if os.path.exists("models/face_model.yml"):
            os.remove("models/face_model.yml")

        self.face_manager.load_model()
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