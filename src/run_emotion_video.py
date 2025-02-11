import cv2
import torch
from torchvision import transforms
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import os

# Import configuration and model details.
from config import DEVICE, IMAGE_SIZE  # e.g., DEVICE = "cuda" or "cpu", IMAGE_SIZE = (112, 112)
from models.emotions import EmotionRecognitionModel
from utils.training_utils import EMOTION_MAPPING

def record_video(output_file="input_video.avi", fps=20):
    """
    Records video from the webcam until the user presses 'q'.
    Saves the video to output_file.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Get the frame width and height from the camera.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    print("Recording video... Press 'q' to stop recording.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Write the frame to the output file.
        out.write(frame)
        # Display the frame.
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video recorded and saved as {output_file}")

def process_video(input_file="input_video.avi", output_file="output_video.avi"):
    """
    Processes the recorded video:
      - Reads frames from input_file.
      - Detects faces and predicts emotions.
      - Overlays bounding boxes and emotion labels.
      - Saves the processed frames into output_file.
    """
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Could not open the input video file.")
        return

    # Get video properties.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter for output.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    # Initialize the face detector (MTCNN) and load the emotion model.
    detector = MTCNN()
    model = EmotionRecognitionModel(embedding_size=512, num_emotions=8, freeze_backbone=False)
    # Load the pretrained emotion model weights.
    checkpoint = torch.load("best_emotion_model.pth", map_location=DEVICE)
    # Check if the checkpoint is a dictionary containing a model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Define the transformation for face images (same as used during inference/training, but without augmentations).
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Processing video for emotion recognition...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MTCNN expects RGB images).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the frame.
        detections = detector.detect_faces(rgb_frame)
        
        if detections:
            for detection in detections:
                x, y, w, h = detection["box"]
                # Ensure coordinates are within image bounds.
                x = max(0, x)
                y = max(0, y)
                # Crop the face region.
                face_crop = rgb_frame[y:y+h, x:x+w]
                # Convert the face crop to a PIL image.
                face_img = Image.fromarray(face_crop)
                # Apply the transformation.
                face_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
                
                # Predict the emotion.
                with torch.no_grad():
                    outputs = model(face_tensor)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    emotion = EMOTION_MAPPING.get(predicted_class, "unknown")
                
                # Draw a bounding box and overlay the predicted emotion.
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video.
        out.write(frame)
        # Optionally display the processed frame.
        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_file}")

def main():
    # First, record a video from the webcam.
    record_video(output_file="input_video.avi", fps=20)
    # Then, process the recorded video for emotion recognition.
    process_video(input_file="input_video.avi", output_file="output_video.avi")

if __name__ == "__main__":
    main()
