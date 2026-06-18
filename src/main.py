import tkinter as tk
from app.attendance_app import AttendanceApp

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


# Main Launch of the program