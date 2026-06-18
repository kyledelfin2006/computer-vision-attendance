import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import os
import csv
from datetime import datetime

from database import (
    init_db, create_session, end_session, log_attendance,
    get_all_sessions, get_attendance_for_session,
    get_person_id_by_name, get_person_name_by_id
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
            # Show in registration tab (always)
            self.display_frame(self.reg_video_label, frame.copy())

            # If attendance is active, process for attendance
            if self.attendance_active:
                self.process_attendance_frame(frame)
            else:
                # Otherwise just show the raw frame in the attendance tab
                self.display_frame(self.att_video_label, frame)
        self.root.after(30, self.update_video)

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
        self.reg_status.config(text="Capturing faces...", fg="blue")
        self.root.update()

        # Capture up to 20 frames with faces
        collected = []
        count = 0
        while count < 20 and self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            boxes = self.face_manager.detect_faces(frame)
            for box in boxes:
                roi = self.face_manager.get_face_roi(frame, box)
                if roi is not None:
                    collected.append(roi)
                    count += 1
                    # Draw feedback on the frame
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Captured {count}/20", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.display_frame(self.reg_video_label, frame)
                    self.root.update()
                    if count >= 20:
                        break
                cv2.waitKey(50)

        if len(collected) < 10:
            self.reg_status.config(text="Not enough face images captured. Try again.", fg="red")
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
                        status_text = "Marked"
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

    # ---------------------- Cleanup ----------------------
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()