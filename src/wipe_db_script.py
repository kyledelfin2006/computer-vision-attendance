import sqlite3
import os
import shutil

# This file executes a script to delete the entire DB (Python Scripting)

def wipe_database():
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DELETE FROM {table_name};")
            print(f"Cleared table: {table_name}")
        conn.commit()
        conn.close()
        print("Database tables cleared.")
    except Exception as e:
        print(f"Error clearing database: {e}")

def delete_folders_and_model():
    folders = ["images", "exports"]
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}/")
            except Exception as e:
                print(f"Could not delete {folder}/: {e}")
        else:
            print(f"Folder {folder}/ does not exist – skipping.")

    model_path = "models/face_model.yml"
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"Deleted model: {model_path}")
        except Exception as e:
            print(f"Could not delete {model_path}: {e}")
    else:
        print("Model file not found – skipping.")

if __name__ == "__main__":

    # Run everything unconditionally
    wipe_database()
    delete_folders_and_model()
    print("System fully reset. Run 'python main.py' to start fresh.")