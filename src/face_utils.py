import cv2
import os
import numpy as np
from database import get_all_persons, add_person, get_person_name_by_id, get_person_id_by_name

MODEL_PATH = "models/face_model.yml"
IMAGE_DIR = "images"
DNN_PROTOTXT = "models/dnn/deploy.prototxt"
DNN_CAFFEMODEL = "models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 70   # LBPH confidence (lower = more certain)
DNN_CONFIDENCE = 0.7        # DNN detection confidence

class FaceManager:
    def __init__(self):
        # Load DNN face detector
        self.face_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.recognizer.read(MODEL_PATH)
        else:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def detect_faces(self, frame):
        """Return list of (x1, y1, x2, y2) bounding boxes."""
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
                # Clamp to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def get_face_roi(self, frame, box):
        """Extract the face region from a bounding box."""
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, FACE_SIZE)

    def recognize_face(self, face_roi):
        """Recognize a face ROI and return (name, confidence)."""
        if face_roi is None:
            return None, None
        label, confidence = self.recognizer.predict(face_roi)
        if confidence < CONFIDENCE_THRESHOLD:
            name = get_person_name_by_id(label)
            return name, confidence
        else:
            return None, confidence

    def register_person(self, name, face_crops):
        """Register a new person with their face crops."""
        if not face_crops:
            return None
        # Check if name already exists
        existing_id = get_person_id_by_name(name)
        if existing_id:
            return None  # duplicate name

        # Insert into DB
        pid = add_person(name)
        if pid is None:
            return None

        # Save face crops to disk
        person_dir = os.path.join(IMAGE_DIR, str(pid))
        os.makedirs(person_dir, exist_ok=True)
        for i, crop in enumerate(face_crops):
            cv2.imwrite(os.path.join(person_dir, f"frame_{i}.jpg"), crop)

        # Retrain model on all stored faces
        self.train_model()
        return pid

    def train_model(self):
        """Train the LBPH recognizer on all stored face images."""
        images = []
        labels = []
        persons = get_all_persons()  # list of (id, name)
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
            self.recognizer.save(MODEL_PATH)