import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from config import FACES_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TEMPERATURE, DEVICE, IMAGE_SIZE, MODEL_SAVE_PATH
from utils import capture_frames, extract_faces
from models.arcface import ArcFaceBackbone
from models.projection import ProjectionHead
from models.simclr import SimCLRModel


def nt_xent_loss(z, temperature=TEMPERATURE):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Expects z of shape (2N, dim) with positive pairs in order.
    """
    z = nn.functional.normalize(z, dim=1)
    batch_size = z.shape[0]
    similarity_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    N = batch_size // 2
    pos_indices = (torch.arange(batch_size, device=z.device) + N) % batch_size
    positives = similarity_matrix[torch.arange(batch_size, device=z.device), pos_indices]
    loss = -positives + torch.log(torch.sum(torch.exp(similarity_matrix), dim=1))
    return loss.mean()

class FaceDataset(Dataset):
    """
    Dataset that loads face images from FACES_DIR and returns two augmented views.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
        else:
            view1 = image
            view2 = image
        return view1, view2

def train_model(args):
    # Data augmentation for contrastive learning
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomResizedCrop(IMAGE_SIZE[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FaceDataset(folder=FACES_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model components
    backbone = ArcFaceBackbone(embedding_size=512)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
    model = SimCLRModel(backbone, projection_head).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Starting contrastive training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for view1, view2 in dataloader:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)
            optimizer.zero_grad()
            _, z1 = model(view1)
            _, z2 = model(view2)
            z = torch.cat([z1, z2], dim=0)
            loss = nt_xent_loss(z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

def main():
    parser = argparse.ArgumentParser(description="ArcFace Fine-Tuning with Contrastive Learning")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['capture', 'train'],
        required=True,
        help="Mode: 'capture' to capture faces from webcam, 'train' to train the model on extracted faces."
    )
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs (train mode).")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training (train mode).")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate for training (train mode).")
    args = parser.parse_args()

    if args.mode == 'capture':
        capture_frames()
        extract_faces()
    elif args.mode == 'train':
        train_model(args)

if __name__ == "__main__":
    main()
