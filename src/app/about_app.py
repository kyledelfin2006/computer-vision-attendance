# app/about.py
import tkinter as tk

def create_about_frame(parent):
    """Return a frame with the About information, centered in the tab."""
    # Outer frame fills the entire tab
    outer = tk.Frame(parent)
    outer.pack(fill=tk.BOTH, expand=True)

    # Configure grid to center the inner frame
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    # Inner frame that holds all content – this will be centered
    center = tk.Frame(outer)
    center.grid(row=0, column=0)

    # Title
    tk.Label(center, text="About", font=("Arial", 20, "bold")).pack(pady=(10, 5))

    # Project name
    tk.Label(center, text="Facial Recognition Attendance System", font=("Arial", 16)).pack(pady=5)

    # ----- Tech Stack -----
    tk.Label(center, text="Tech Stack & Libraries", font=("Arial", 14, "bold")).pack(pady=(15, 5))

    tech_text = (
        "• OpenCV (opencv-python & opencv-contrib-python) – used for face detection with a "
        "pre‑trained DNN (SSD) and face recognition via LBPH (Local Binary Patterns Histograms).\n\n"
        "• SQLite3 – lightweight embedded database that stores registered persons, attendance "
        "sessions, and logs without needing a separate server.\n\n"
        "• Tkinter – the built‑in Python GUI toolkit that builds the entire user interface, "
        "including tabs, buttons, video display, and status messages.\n\n"
        "• Pillow – converts OpenCV frames (NumPy arrays) into images that Tkinter can display "
        "inside the video labels.\n\n"
        "• CSV – built‑in module used to export attendance records into comma‑separated files "
        "for easy viewing in spreadsheet programs.\n\n"
        "• Threading – enables background processing during registration so the interface "
        "remains responsive while the face model is being trained."
    )
    tk.Label(center, text=tech_text, font=("Arial", 11), justify=tk.LEFT, wraplength=700).pack(
        pady=5, padx=20, anchor="center"
    )

    # ----- Developed for / Users -----
    tk.Label(center, text="Developed for", font=("Arial", 14, "bold")).pack(pady=(15, 5))
    dev_text = (
        "This application is designed for classrooms, small offices, event check‑ins, and "
        "any environment where a fast, offline, privacy‑preserving attendance system is needed. "
        "It runs entirely on the local machine without sending any data over the network."
    )
    tk.Label(center, text=dev_text, font=("Arial", 11), wraplength=700, justify=tk.LEFT).pack(
        pady=5, padx=20, anchor="center"
    )

    tk.Label(center, text="Highlighted Users", font=("Arial", 14, "bold")).pack(pady=(10, 5))
    users_text = (
        "• Teachers and instructors for taking class attendance.\n"
        "• HR managers for tracking employee attendance in small offices.\n"
        "• Event organizers for quick participant check‑in.\n"
        "• Any group that needs a reliable, no‑internet attendance solution."
    )
    tk.Label(center, text=users_text, font=("Arial", 11), justify=tk.LEFT, wraplength=700).pack(
        pady=5, padx=20, anchor="center"
    )

    return outer