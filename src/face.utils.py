import reference

import cv2
import numpy as np


# ... (other imports)

class FaceManager:
    def __init__(self):
        # Load the DNN model
        self.face_net = cv2.dnn.readNetFromCaffe(
            "models/dnn/deploy.prototxt",
            "models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )[reference:5]

        # ... (rest of your initialization for LBPH recognizer)

    def detect_faces(self, frame):
        """Detect faces in a frame using DNN."""
        (h, w) = frame.shape[:2]

        # Preprocess the image for the DNN model
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),  # Resize to model's expected input[reference:6]
            1.0,  # Scale factor
            (300, 300),  # Input image size
            (104.0, 177.0, 123.0)  # Mean subtraction for BGR channels[reference:7]
        )

        # Pass the blob through the network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()[reference:8]

        face_boxes = []
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of the detection[reference:9]

            # Filter out weak detections
            if confidence > 0.7:  # You can adjust this threshold[reference:10]
                # Scale the bounding box coordinates back to the original image size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")[reference:11]

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_boxes.append((x1, y1, x2, y2))

        return face_boxes

    def get_face_roi(self, frame, box):
        """Extract the face region from a bounding box."""
        (x1, y1, x2, y2) = box
        face_roi = frame[y1:y2, x1:x2]
        return face_roi

    # Modify your existing methods to use these
    def process_frame(self, frame):
        """Main method to detect and recognize faces in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect faces using DNN
        face_boxes = self.detect_faces(frame)

        for (x1, y1, x2, y2) in face_boxes:
            # 2. Extract face ROI
            face_roi = gray[y1:y2, x1:x2]

            # 3. Recognize the face using your existing LBPH logic
            name, confidence = self.recognize_face(face_roi)

            # 4. Draw results on the frame
            # ... (your drawing code) ...