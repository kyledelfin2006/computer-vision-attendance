import cv2
import os
from pathlib import Path
import numpy as np
from data.database import get_all_persons, add_person, get_person_name_by_id, get_person_id_by_name

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
MODEL_PATH = SRC_DIR / "models" / "face_model.yml"
IMAGE_DIR = PROJECT_DIR / "images"
DNN_PROTOTXT = SRC_DIR / "models" / "dnn" / "deploy.prototxt"
DNN_CAFFEMODEL = SRC_DIR / "models" / "dnn" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 70   # LBPH confidence (lower = more certain)
DNN_CONFIDENCE = 0.7        # DNN detection confidence

class FaceManager:
    def __init__(self):
        self.face_net = cv2.dnn.readNetFromCaffe(str(DNN_PROTOTXT), str(DNN_CAFFEMODEL))
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.recognizer.read(str(MODEL_PATH))
        else:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DNN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def get_face_roi(self, frame, box):
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, FACE_SIZE)

    def recognize_face(self, face_roi):
        if face_roi is None:
            return None, None
        try:
            label, confidence = self.recognizer.predict(face_roi)
        except cv2.error:
            return None, None
        if confidence < CONFIDENCE_THRESHOLD:
            name = get_person_name_by_id(label)
            return name, confidence
        else:
            return None, confidence

    def register_person(self, name, face_crops):
        """Register a new person with their face crops."""
        if not face_crops:
            return None
        existing_id = get_person_id_by_name(name)
        if existing_id:
            return None

        pid = add_person(name)
        if pid is None:
            return None

        person_dir = os.path.join(IMAGE_DIR, str(pid))
        os.makedirs(person_dir, exist_ok=True)
        for i, crop in enumerate(face_crops):
            cv2.imwrite(os.path.join(person_dir, f"frame_{i}.jpg"), crop)

        self.train_model()
        return pid

    def train_model(self):
        images = []
        labels = []
        persons = get_all_persons()
        for pid, _ in persons:
            person_dir = os.path.join(IMAGE_DIR, str(pid))
            if not os.path.exists(person_dir):
                continue
            for fname in os.listdir(person_dir):
                img_path = os.path.join(person_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, FACE_SIZE)
                images.append(img)
                labels.append(pid)
        if images:
            self.recognizer.train(images, np.array(labels))
            self.recognizer.save(str(MODEL_PATH))
