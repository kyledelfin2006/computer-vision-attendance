import csv
import os
import re
import shutil
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from app.about_app import create_about_frame
from data.database import (
    init_db, create_session, end_session, log_attendance,
    get_all_sessions, get_attendance_for_session,
    get_person_id_by_name, get_connection, get_session_by_id
)
from face.face_manager import DNN_PROTOTXT, FaceManager, MODEL_PATH

# Fixed video display size
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
ATT_VIDEO_WIDTH = 975
ATT_VIDEO_HEIGHT = 731
APP_NAME = "Facial Recognition Attendance System"


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1300x900")

        # Initialize database
        init_db()

        # Create necessary folders
        os.makedirs("images", exist_ok=True)
        os.makedirs(DNN_PROTOTXT.parent, exist_ok=True)
        os.makedirs("exports", exist_ok=True)

        # Face manager (DNN + LBPH)
        self.face_manager = FaceManager()

        # Camera and state
        self.cap = None
        self.camera_running = False
        self.attendance_active = False
        self.current_session_id = None
        self.current_session_name = None
        self.current_session_start_time = None
        self.current_session_csv = None
        self.attendance_feedback_until = {}

        # Timer for registration status messages
        self._reg_status_timer = None

        # Build GUI
        self._build_ui()

        # Start camera
        self.start_camera()

    def set_reg_status(self, text, fg="black", clear_after=0):
        """Set the registration status label and append to the side panel."""
        # Cancel any pending clear timer
        if hasattr(self, '_reg_status_timer') and self._reg_status_timer:
            self.root.after_cancel(self._reg_status_timer)
            self._reg_status_timer = None

        # Update the label below the video
        self.reg_status.config(text=text, fg=fg)

        # Append to the side panel with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.feedback_text.see(tk.END)  # scroll to the latest message
        self.feedback_text.config(state=tk.DISABLED)

        if clear_after > 0:
            # Clear the label below video after the delay (but not the side panel)
            self._reg_status_timer = self.root.after(
                clear_after,
                lambda: self.reg_status.config(text="", fg="black")
            )

    # ---------------------- UI Construction ----------------------

    def _build_ui(self):
        # ---- Style for larger tabs ----
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', 14, 'bold'), padding=[20, 10])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ---- Tab 1: Registration ----
        self.reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reg_frame, text="Registration")

        # Main container with 2 columns
        reg_main = tk.Frame(self.reg_frame)
        reg_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        reg_main.grid_columnconfigure(0, weight=1)  # video area
        reg_main.grid_columnconfigure(1, weight=0)  # side panel (fixed width)
        reg_main.grid_rowconfigure(0, weight=1)

        # ---- Left: Video + title ----
        left_frame = tk.Frame(reg_main)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        tk.Label(left_frame, text=f"{APP_NAME} - Registration",
                 font=("Arial", 18, "bold")).grid(row=0, column=0, pady=(0, 5))

        self.reg_video_label = tk.Label(left_frame)
        self.reg_video_label.grid(row=1, column=0, sticky="nsew", pady=5)

        # Controls (Name entry and buttons) – placed below video
        ctrl_reg = tk.Frame(left_frame)
        ctrl_reg.grid(row=2, column=0, pady=5)
        ctrl_reg.grid_columnconfigure(1, weight=1)
        tk.Label(ctrl_reg, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=6, sticky="e")
        self.reg_name_entry = tk.Entry(ctrl_reg, width=30, font=("Arial", 12))
        self.reg_name_entry.grid(row=0, column=1, padx=10, pady=6, sticky="ew")
        self.reg_btn = tk.Button(ctrl_reg, text="Register Person", command=self.register_person,
                                 font=("Arial", 12, "bold"), width=18, height=2)
        self.reg_btn.grid(row=0, column=2, padx=10, pady=6, sticky="ew")
        self.reset_btn = tk.Button(ctrl_reg, text="Reset All Data", command=self.reset_database,
                                   bg="red", fg="white", activebackground="#b00000", activeforeground="white",
                                   font=("Arial", 12, "bold"), width=16, height=2)
        self.reset_btn.grid(row=0, column=3, padx=10, pady=6, sticky="ew")

        # Status label below controls (still kept for compatibility)
        self.reg_status = tk.Label(left_frame, text="", font=("Arial", 12))
        self.reg_status.grid(row=3, column=0, pady=5)

        # ---- Right: Side panel for feedback messages ----
        side_frame = tk.Frame(reg_main, width=300)
        side_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        side_frame.grid_propagate(False)  # keep fixed width
        side_frame.grid_rowconfigure(0, weight=0)
        side_frame.grid_rowconfigure(1, weight=1)
        side_frame.grid_columnconfigure(0, weight=1)

        tk.Label(side_frame, text="Feedback", font=("Arial", 16, "bold")).grid(row=0, column=0, pady=(0, 10))

        # A larger, scrollable text area for feedback messages (read‑only)
        self.feedback_text = tk.Text(side_frame, wrap=tk.WORD, font=("Arial", 12), height=20,
                                     relief=tk.GROOVE, bd=2)
        self.feedback_text.grid(row=1, column=0, sticky="nsew")
        self.feedback_text.config(state=tk.DISABLED)  # start as read‑only

        # ---- Tab 2: Attendance ----
        self.att_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.att_frame, text="Attendance")

        tk.Label(self.att_frame, text=f"{APP_NAME} - Attendance",
                 font=("Arial", 13, "bold")).pack(pady=(8, 2))

        attendance_body = tk.Frame(self.att_frame)
        attendance_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 8))
        attendance_body.grid_columnconfigure(0, weight=1)
        attendance_body.grid_columnconfigure(1, weight=0)
        attendance_body.grid_rowconfigure(0, weight=1)

        video_area = tk.Frame(attendance_body)
        video_area.grid(row=0, column=0, sticky="n", padx=(0, 10))

        feedback_frame = tk.Frame(video_area, bd=1, relief=tk.SOLID)
        feedback_frame.pack(fill=tk.X, anchor="w", pady=(0, 6))
        feedback_frame.grid_columnconfigure(1, weight=1)
        self.att_feedback_title = tk.Label(feedback_frame, text="Registered:", font=("Arial", 12, "bold"))
        self.att_feedback_title.grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        self.att_status = tk.Label(feedback_frame, text="No one registered yet.", font=("Arial", 12),
                                   wraplength=760, justify=tk.LEFT)
        self.att_status.grid(row=0, column=1, padx=(4, 8), pady=6, sticky="ew")

        self.att_video_label = tk.Label(video_area)
        self.att_video_label.pack(anchor="n")

        side_panel = tk.Frame(attendance_body, width=270)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        side_panel.grid_propagate(False)
        side_panel.grid_columnconfigure(0, weight=1)

        ctrl_att = tk.Frame(side_panel)
        ctrl_att.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctrl_att.grid_columnconfigure(0, weight=1)

        tk.Label(ctrl_att, text="Session Name:", font=("Arial", 12)).grid(row=0, column=0, padx=4, pady=(0, 4),
                                                                          sticky="w")
        self.session_name_entry = tk.Entry(ctrl_att, width=30, font=("Arial", 12))
        self.session_name_entry.grid(row=1, column=0, padx=4, pady=(0, 8), sticky="ew")

        self.start_btn = tk.Button(ctrl_att, text="Start Attendance", command=self.start_session,
                                   font=("Arial", 12, "bold"), width=18, height=2)
        self.start_btn.grid(row=2, column=0, padx=4, pady=4, sticky="ew")

        self.stop_btn = tk.Button(ctrl_att, text="Stop Attendance", command=self.stop_session,
                                  state=tk.DISABLED, font=("Arial", 12, "bold"), width=18, height=2)
        self.stop_btn.grid(row=3, column=0, padx=4, pady=4, sticky="ew")

        self.att_reset_btn = tk.Button(ctrl_att, text="Reset All Data", command=self.reset_database,
                                       bg="red", fg="white", activebackground="#b00000", activeforeground="white",
                                       font=("Arial", 12, "bold"), width=18, height=2)
        self.att_reset_btn.grid(row=4, column=0, padx=4, pady=(10, 4), sticky="ew")

        session_time_frame = tk.Frame(side_panel)
        session_time_frame.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        session_time_frame.grid_columnconfigure(0, weight=1)
        self.start_time_label = tk.Label(session_time_frame, text="Start Time: --", font=("Arial", 11))
        self.start_time_label.grid(row=0, column=0, padx=4, pady=3, sticky="w")
        self.end_time_label = tk.Label(session_time_frame, text="End Time: --", font=("Arial", 11))
        self.end_time_label.grid(row=1, column=0, padx=4, pady=3, sticky="w")

        # Export section
        export_frame = tk.Frame(side_panel)
        export_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        export_frame.grid_columnconfigure(0, weight=1)
        tk.Label(export_frame, text="Export Session:", font=("Arial", 12)).grid(row=0, column=0, padx=4, pady=(0, 4),
                                                                                sticky="w")
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(export_frame, textvariable=self.session_var,
                                          state="readonly", font=("Arial", 12), width=30)
        self.session_combo.grid(row=1, column=0, padx=4, pady=(0, 8), sticky="ew")
        self.export_btn = tk.Button(export_frame, text="Export CSV", command=self.export_csv,
                                    bg="orange", activebackground="#ffb347",
                                    font=("Arial", 12, "bold"), width=14, height=2)
        self.export_btn.grid(row=2, column=0, padx=4, pady=4, sticky="ew")
        self.refresh_sessions()

        # ---- Tab 3: About ----
        self.about_frame = create_about_frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")

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
                self.display_frame(self.att_video_label, frame, ATT_VIDEO_WIDTH, ATT_VIDEO_HEIGHT)

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


    def display_frame(self, label, frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    # ---------------------- Registration (enhanced overlay) ----------------------
    def draw_registration_feedback(self, frame, message, countdown=None, captured=None, progress=None, color=(0, 255, 255)):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 190), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, APP_NAME, (30, 45), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, message, (30, 85), font, 0.9, color, 2)

        if countdown is not None:
            cv2.putText(frame, f"Countdown: {countdown:0.1f}s", (30, 125),
                        font, 0.9, (255, 255, 0), 2)
        if captured is not None:
            cv2.putText(frame, f"Captured images: {captured}", (30, 165),
                        font, 0.9, (0, 255, 0), 2)
        if progress is not None:
            bar_x, bar_y, bar_w, bar_h = 30, 205, 360, 22
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)
        return frame

    def register_person(self):
        name = self.reg_name_entry.get().strip()
        if not name:
            self.set_reg_status("Please enter a name", fg="red")
            return
        if get_person_id_by_name(name):
            self.set_reg_status(f"Name '{name}' already registered", fg="red")
            return

        # Disable button
        self.reg_btn.config(state=tk.DISABLED)

        # ---- 4‑second capture phase (unchanged) ----
        self.set_reg_status("Starting 4 second capture. Slowly move your head left and right.", fg="blue")
        self.root.update()

        collected = []
        start_time = time.time()
        capture_interval = 0.25
        last_capture_time = start_time
        count = 0
        capture_duration = 4.0

        while time.time() - start_time < capture_duration and self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            boxes = self.face_manager.detect_faces(frame)

            elapsed = time.time() - start_time
            remaining = max(0, capture_duration - elapsed)

            if elapsed >= 0.5 and time.time() - last_capture_time >= capture_interval and boxes:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                roi = self.face_manager.get_face_roi(frame, box)
                if roi is not None:
                    collected.append(roi)
                    count += 1
                    last_capture_time = time.time()

            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            progress = 1.0 - (remaining / capture_duration)
            frame = self.draw_registration_feedback(
                frame,
                "Slowly move your head left and right",
                countdown=remaining,
                captured=count,
                progress=progress
            )

            self.set_reg_status(
                f"Capturing... {remaining:0.1f}s left. Keep moving your head slowly.",
                fg="blue"
            )
            self.display_frame(self.reg_video_label, frame)
            self.root.update()

        if len(collected) < 10:
            self.set_reg_status("Not enough face images captured. Please try again.", fg="red")
            self.reg_btn.config(state=tk.NORMAL)
            return

        # ---- Show "Processing" with timer in background ----
        self.set_reg_status("Processing registration... 0.0s", fg="blue")
        self.root.update()

        # Store references for the background thread
        self._reg_name = name
        self._reg_collected = collected
        self._reg_start_time = time.time()
        self._reg_finished = False
        self._reg_result = None  # will store pid or None

        # Start the background thread for training
        self._reg_timer_running = True
        self._update_processing_timer()  # start UI timer updates

        thread = threading.Thread(target=self._do_registration, daemon=True)
        thread.start()

    def _update_processing_timer(self):
        """Update the processing status label with elapsed time while training."""
        if not hasattr(self, '_reg_timer_running') or not self._reg_timer_running:
            return

        elapsed = time.time() - self._reg_start_time
        self.set_reg_status(f"Processing registration... {elapsed:.1f}s", fg="blue")

        # Schedule next update after 500 ms if still running
        if self._reg_timer_running:
            self.root.after(500, self._update_processing_timer)

    def _do_registration(self):
        """Run the actual registration (training) in a background thread."""
        pid = self.face_manager.register_person(self._reg_name, self._reg_collected)
        self._reg_result = pid
        self._reg_finished = True
        self._reg_timer_running = False

        # Schedule the UI update on the main thread
        self.root.after(0, self._finish_registration)

    def _finish_registration(self):
        """Called after training completes to show the final status."""
        if self._reg_result:
            self.set_reg_status(
                f"Registration complete: '{self._reg_name}' saved with ID {self._reg_result}.",
                fg="green",
                clear_after=20000
            )
            self.reg_name_entry.delete(0, tk.END)
            self.refresh_sessions()
        else:
            self.set_reg_status("Registration failed (duplicate name or error)", fg="red")
        self.reg_btn.config(state=tk.NORMAL)

        # Clean up thread-related variables
        self._reg_timer_running = False
        if hasattr(self, '_reg_name'):
            del self._reg_name
        if hasattr(self, '_reg_collected'):
            del self._reg_collected

    # ---------------------- Attendance ----------------------
    def safe_filename(self, value):
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        return clean.strip("_") or "session"

    def get_session_csv_path(self, session_id, session_name):
        safe_name = self.safe_filename(session_name)
        return os.path.join("exports", f"session_{session_id}_{safe_name}.csv")

    def write_session_csv(self, session_id, session_name, records, start_time=None, end_time=None):
        os.makedirs("exports", exist_ok=True)
        filename = self.get_session_csv_path(session_id, session_name)
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
        if not self.current_session_csv:
            return
        with open(self.current_session_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])

    def start_session(self):
        name = self.session_name_entry.get().strip()
        if not name:
            self.att_status.config(text="Enter a session name", fg="red")
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

    def process_attendance_frame(self, frame):
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
                    self.attendance_feedback_until[pid] = time.time() + 4.0
                    self.att_status.config(text=f"{name}\n{timestamp}", fg="green")
                    status_text, color = "Registered in session", (0, 255, 0)  # green

                elif time.time() < self.attendance_feedback_until.get(pid, 0):
                    status_text, color = "Registered in session", (0, 255, 0)  # green

                else:
                    status_text, color = "Already registered in session", (0, 255, 255)  # yellow

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name if name else 'Unknown'}: {status_text}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.display_frame(self.att_video_label, frame, ATT_VIDEO_WIDTH, ATT_VIDEO_HEIGHT)

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

        session = get_session_by_id(sid)
        if not session:
            messagebox.showerror("Error", "Session not found")
            return

        records = get_attendance_for_session(sid)
        if not records:
            messagebox.showinfo("Empty", "No attendance records for this session")
            return

        _, session_name, start_time, end_time = session
        filename = self.write_session_csv(sid, session_name, records, start_time, end_time)
        messagebox.showinfo("Export", f"Exported to {filename}")

    # ---------------------- Reset / Wipe All Data ----------------------
    def reset_database(self):
        if not messagebox.askyesno("Reset Database", "Delete all persons, sessions, attendance, images, and model?"):
            return

        if self.attendance_active:
            self.stop_session(save_csv=False)

        with get_connection() as conn:
            conn.execute("DELETE FROM attendance")
            conn.execute("DELETE FROM persons")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM sqlite_sequence")

        if os.path.exists("images"):
            shutil.rmtree("images")
        os.makedirs("images", exist_ok=True)

        if os.path.exists("exports"):
            shutil.rmtree("exports")
        os.makedirs("exports", exist_ok=True)

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        self.current_session_csv = None
        self.session_var.set("")
        self.session_combo['values'] = []
        self.face_manager.load_model()
        self.refresh_sessions()
        self.reg_status.config(text="All data wiped", fg="blue")
        self.att_status.config(text="No one registered yet.", fg="blue")
        self.start_time_label.config(text="Start Time: --")
        self.end_time_label.config(text="End Time: --")
        self.reg_name_entry.delete(0, tk.END)

        # Clear the feedback panel
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.config(state=tk.DISABLED)

    # ---------------------- Cleanup ----------------------
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
