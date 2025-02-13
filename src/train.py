import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import psutil
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR

from config import (
    FACES_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TEMPERATURE, SCHEDULER_CONFIG, 
    DEVICE, IMAGE_SIZE, MODEL_SAVE_PATH
)
from models.arcface import ArcFaceBackbone
from models.projection import ProjectionHead
from models.simclr import SimCLRModel
from utils.dataset_utils import load_local_dataset, combine_datasets
from utils.training_utils import print_classification_report, EMOTION_MAPPING

def get_optimal_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            torch.zeros(1).to('mps')
            print("Using MPS (Metal) device")
            return torch.device("mps")
        except Exception:
            print("MPS (Metal) is available but not working properly")
    
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    
    print("Using CPU device")
    return torch.device("cpu")

class PerformanceMonitor:
    def __init__(self):
        self.batch_times = []
        self.memory_usage = []
        self.start_time = time.time()
        
    def update(self):
        if torch.cuda.is_available():
            memory = torch.cuda.memory_reserved() / 1024**2
        else:
            memory = psutil.Process().memory_info().rss / 1024**2
        self.memory_usage.append(memory)
        
    def get_statistics(self):
        avg_time = np.mean(self.batch_times) if self.batch_times else 0
        avg_memory = np.mean(self.memory_usage)
        return avg_time, avg_memory

def nt_xent_loss(z, temperature=TEMPERATURE):
    """
    Computes the NT-Xent loss for contrastive learning.
    Ensures all tensors are float32.
    """
    z = z.to(torch.float32)  # Ensure float32
    z = nn.functional.normalize(z, dim=1, p=2)
    batch_size = z.shape[0]
    
    similarity_matrix = torch.matmul(z, z.t()) / temperature
    
    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    pos_indices = (torch.arange(batch_size, device=z.device) + batch_size // 2) % batch_size
    positives = similarity_matrix[torch.arange(batch_size, device=z.device), pos_indices]
    
    denominator = torch.exp(similarity_matrix).sum(dim=1)
    loss = -positives + torch.log(denominator)
    return loss.mean()


class FaceDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.image_files = sorted(list(self.folder.glob("*.[jJ][pP][gG]")) + 
                                list(self.folder.glob("*.[jJ][pP][eE][gG]")) + 
                                list(self.folder.glob("*.[pP][nN][gG]")))
        self.transform = transform
        self.cache = {}
        self.max_cache_size = 1000  # Adjust based on your RAM

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx in self.cache:
            image = self.cache[idx]
        else:
            image = Image.open(self.image_files[idx]).convert('RGB')
            if len(self.cache) < self.max_cache_size:
                self.cache[idx] = image

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        return image, image

def create_transforms():
    try:
        import torchvision.transforms.v2 as transforms_v2
        print("Using transforms v2")
        return transforms_v2.Compose([
            transforms_v2.Resize(IMAGE_SIZE),
            transforms_v2.RandomResizedCrop(IMAGE_SIZE[0]),
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    except ImportError:
        print("Using transforms v1")
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomResizedCrop(IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
def train_model(args):
   print("Initializing transforms...")
   train_transform = create_transforms()
   
   print("Loading dataset...")
   dataset = FaceDataset(folder=FACES_DIR, transform=train_transform)
   print(f"Found {len(dataset)} images")
   
   print("Creating DataLoader...")
   dataloader = DataLoader(
       dataset,
       batch_size=args.batch_size,
       shuffle=True,
       num_workers=min(4, os.cpu_count()),
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
   )

   print("Initializing model...")
   backbone = ArcFaceBackbone(embedding_size=512)
   projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
   model = SimCLRModel(backbone, projection_head).to(DEVICE)
   
   # Set model to float32
   model = model.to(torch.float32)

   # Don't use torch.compile for MPS
   if DEVICE.type == "cuda" and hasattr(torch, 'compile'):
       try:
           model = torch.compile(model)
           print("Using torch.compile")
       except Exception as e:
           print(f"torch.compile not available: {e}")

   optimizer = optim.Adam(model.parameters(), lr=args.lr)
   
   
   scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(dataloader),
        **SCHEDULER_CONFIG['params']
    )


   # Disable mixed precision for MPS
   use_amp = DEVICE.type == 'cuda'  # Only use AMP with CUDA
   scaler = torch.cuda.amp.GradScaler() if use_amp else None

   print(f"\nStarting training with {args.epochs} epochs...")
   print(f"Dataset size: {len(dataset)} images")
   print(f"Batch size: {args.batch_size}")
   print(f"Device: {DEVICE}")
   print(f"Using mixed precision: {use_amp}")

   for epoch in range(args.epochs):
       model.train()
       total_loss = 0.0
       progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
       
       for batch_idx, (view1, view2) in enumerate(progress_bar):
           # Ensure inputs are float32
           view1 = view1.to(DEVICE, dtype=torch.float32, non_blocking=True)
           view2 = view2.to(DEVICE, dtype=torch.float32, non_blocking=True)
           
           optimizer.zero_grad(set_to_none=True)
           
           if use_amp:
               with torch.cuda.amp.autocast():
                   _, z1 = model(view1)
                   _, z2 = model(view2)
                   z = torch.cat([z1, z2], dim=0)
                   loss = nt_xent_loss(z)
               
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
           else:
               # Standard forward pass with explicit type handling
               _, z1 = model(view1)
               _, z2 = model(view2)
               z = torch.cat([z1, z2], dim=0).to(torch.float32)
               loss = nt_xent_loss(z)
               loss.backward()
               optimizer.step()
           
           scheduler.step()
           total_loss += loss.item()

           # Update progress bar
           progress_bar.set_postfix({
               'loss': f"{loss.item():.4f}",
               'lr': f"{scheduler.get_last_lr()[0]:.6f}"
           })

       avg_loss = total_loss / len(dataloader)
       print(f"\nEpoch {epoch+1} Summary:")
       print(f"Average Loss: {avg_loss:.4f}")

       # Save checkpoint every 5 epochs
       if (epoch + 1) % 5 == 0:
           checkpoint = {
               'epoch': epoch + 1,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'loss': avg_loss,
           }
           checkpoint_path = f"{MODEL_SAVE_PATH}.checkpoint_{epoch+1}"
           torch.save(checkpoint, checkpoint_path)
           print(f"Checkpoint saved to {checkpoint_path}")

   # Save final model
   torch.save({
       'epoch': args.epochs,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'final_loss': avg_loss,
   }, MODEL_SAVE_PATH)
   print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

   torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Training Script for ArcFace Fine-Tuning with Contrastive Learning"
    )
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate for training")
    args = parser.parse_args()
    
    # Ensure log directory exists
    os.makedirs('./log/profiler', exist_ok=True)
    
    train_model(args)

if __name__ == "__main__":
    main()