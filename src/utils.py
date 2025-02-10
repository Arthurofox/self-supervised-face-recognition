import os
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from src.config import FRAMES_DIR, FACES_DIR, IMAGE_SIZE

def capture_frames():
    """
    Captures frames from the webcam and saves them in the FRAMES_DIR.
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    frame_count = 0
    print("Capturing webcam frames. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam Feed', frame)
        frame_filename = os.path.join(FRAMES_DIR, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Frame capture complete.")

def extract_faces():
    """
    Uses MTCNN to detect and extract faces from frames in the FRAMES_DIR.
    Cropped faces are saved in the FACES_DIR.
    """
    os.makedirs(FACES_DIR, exist_ok=True)
    detector = MTCNN()
    print("Extracting faces from captured frames...")
    for filename in os.listdir(FRAMES_DIR):
        frame_path = os.path.join(FRAMES_DIR, filename)
        try:
            image = Image.open(frame_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")
            continue

        image_np = np.array(image)
        detections = detector.detect_faces(image_np)
        if not detections:
            continue

        for i, detection in enumerate(detections):
            x, y, width, height = detection['box']
            x, y = max(0, x), max(0, y)
            face = image.crop((x, y, x + width, y + height))
            face = face.resize(IMAGE_SIZE)
            face_filename = os.path.join(FACES_DIR, f"{os.path.splitext(filename)[0]}_face_{i}.jpg")
            face.save(face_filename)
    print("Face extraction complete. Faces are saved in the FACES_DIR.")
