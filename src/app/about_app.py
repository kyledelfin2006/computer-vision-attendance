# app/about.py
import tkinter as tk
from tkinter import scrolledtext

def create_about_frame(parent):
    """Return a frame with the About information."""
    frame = tk.Frame(parent)

    # Title
    tk.Label(frame, text="About", font=("Arial", 20, "bold")).pack(pady=(10, 5))

    # Project name
    tk.Label(frame, text="Facial Recognition Attendance System", font=("Arial", 16)).pack(pady=5)

    # ----- Tech Stack -----
    tk.Label(frame, text="Tech Stack & Libraries", font=("Arial", 14, "bold")).pack(pady=(15, 5))

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
    tech_label = tk.Label(frame, text=tech_text, font=("Arial", 11), justify=tk.LEFT, wraplength=700)
    tech_label.pack(pady=5, padx=20, anchor="w")

    # ----- Developed for / Users -----
    tk.Label(frame, text="Developed for", font=("Arial", 14, "bold")).pack(pady=(15, 5))
    dev_text = (
        "This application is designed for classrooms, small offices, event check‑ins, and "
        "any environment where a fast, offline, privacy‑preserving attendance system is needed. "
        "It runs entirely on the local machine without sending any data over the network."
    )
    tk.Label(frame, text=dev_text, font=("Arial", 11), wraplength=700, justify=tk.LEFT).pack(pady=5, padx=20, anchor="w")

    tk.Label(frame, text="Highlighted Users", font=("Arial", 14, "bold")).pack(pady=(10, 5))
    users_text = (
        "• Teachers and instructors for taking class attendance.\n"
        "• HR managers for tracking employee attendance in small offices.\n"
        "• Event organizers for quick participant check‑in.\n"
        "• Any group that needs a reliable, no‑internet attendance solution."
    )
    tk.Label(frame, text=users_text, font=("Arial", 11), justify=tk.LEFT, wraplength=700).pack(pady=5, padx=20, anchor="w")

    # ----- FAQ -----
    tk.Label(frame, text="Frequently Asked Questions", font=("Arial", 14, "bold")).pack(pady=(15, 5))

    faq_text = (
        "Q: How do I register a new person?\n"
        "A: Go to the Registration tab, enter the name, and click 'Register Person'. "
        "The system will capture face images for 4 seconds while you slowly move your head. "
        "After capturing at least 10 images, the model is trained automatically.\n\n"

        "Q: Can I delete a single person?\n"
        "A: Currently only the 'Reset All Data' button removes all persons and sessions. "
        "For individual deletions, you would need to manually delete the person's image folder "
        "and retrain the model.\n\n"

        "Q: How do I take attendance?\n"
        "A: Go to the Attendance tab, enter a session name, and click 'Start Attendance'. "
        "The camera will recognise registered faces and log each person once per session.\n\n"

        "Q: Where are the attendance records saved?\n"
        "A: Each session is exported as a CSV file inside the 'exports' folder. "
        "You can also click 'Export CSV' on an existing session to download it again.\n\n"

        "Q: Is my data secure?\n"
        "A: Yes – all data (images, database, model) stays on your local computer. "
        "No information is sent to any external server.\n\n"

        "Q: What happens if I lose the face model?\n"
        "A: The system will rebuild the model the next time you register a person. "
        "Existing images are stored in the 'images' folder, but you would need to re‑register "
        "each person to retrain the model if the model file is deleted."
    )

    # Use a scrolled text widget for FAQ so it's scrollable
    faq_text_widget = scrolledtext.ScrolledText(
        frame, wrap=tk.WORD, width=80, height=15, font=("Arial", 11), relief=tk.FLAT
    )
    faq_text_widget.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
    faq_text_widget.insert(tk.END, faq_text)
    faq_text_widget.config(state=tk.DISABLED)  # make read‑only

    return frame