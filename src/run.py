import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from mtcnn import MTCNN

from config import MODEL_SAVE_PATH, DEVICE, IMAGE_SIZE
from models.arcface import ArcFaceBackbone
from models.projection import ProjectionHead
from models.simclr import SimCLRModel

def load_finetuned_model():
    """
    Load the fine-tuned SimCLR model (ArcFace backbone with projection head)
    from the saved checkpoint.
    """
    from models.arcface import ArcFaceBackbone
    from models.projection import ProjectionHead
    from models.simclr import SimCLRModel
    from config import MODEL_SAVE_PATH, DEVICE

    backbone = ArcFaceBackbone(embedding_size=512)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
    model = SimCLRModel(backbone, projection_head)
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    
    # If the checkpoint is a dictionary that includes a model state dict,
    # extract that; otherwise, assume the checkpoint is the state dict itself.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    # Load the model and set it to evaluation mode.
    model = load_finetuned_model()

    # Initialize the MTCNN face detector.
    detector = MTCNN()

    # Define preprocessing transforms for the detected face images.
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Open the webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time testing. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV) to RGB (PIL/MTCNN).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame.
        detections = detector.detect_faces(frame_rgb)
        if detections:
            for detection in detections:
                x, y, w, h = detection['box']
                # Ensure coordinates are non-negative.
                x, y = max(0, x), max(0, y)
                # Crop and resize the face.
                face_img = frame_rgb[y:y+h, x:x+w]
                face_pil = Image.fromarray(face_img).resize(IMAGE_SIZE)
                input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                # Run inference (no gradient needed).
                with torch.no_grad():
                    # The model returns the backbone embedding and the projection.
                    backbone_output, embedding = model(input_tensor)

                # As a simple example, we compute the norm of the embedding.
                emb_norm = torch.norm(embedding).item()

                # Draw a rectangle around the face.
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Overlay the embedding norm.
                cv2.putText(frame, f"Norm: {emb_norm:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame.
        cv2.imshow("Real-Time Face Test", frame)

        # Press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
