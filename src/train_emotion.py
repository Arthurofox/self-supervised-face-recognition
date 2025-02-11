import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report

from config import DEVICE, IMAGE_SIZE
from models.emotions import EmotionRecognitionModel
import numpy as np
from utils.dataset_utils import load_local_dataset, combine_datasets, calculate_class_weights
from utils.training_utils import print_classification_report, EMOTION_MAPPING, EarlyStopping
from config import IMAGE_SIZE


class EmotionDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # Pre-process all images at initialization
        self.processed_data = []
        for item in dataset:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            if self.transform:
                image = self.transform(image)
            self.processed_data.append({
                "pixel_values": image,
                "label": torch.tensor(item["label"], dtype=torch.long)
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.processed_data[idx]
    

def create_transforms(train=True):
    """Create transform pipelines for training and validation"""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def evaluate_model(model, val_loader, criterion, device):
    """Optimized evaluation function"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        zero_division=1,  # This will suppress the UndefinedMetricWarning
        output_dict=True
    )
    return avg_loss, accuracy, report

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, early_stopping=None):
    best_accuracy = 0
    best_model_path = os.path.join(os.getcwd(), "best_emotion_model.pth")
    
    # Move model to device once
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm with less frequent updates
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=100,
            mininterval=1.0  # Update every second
        )
        
        for batch in progress_bar:
            # Move data to device
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            # Update progress bar less frequently
            if progress_bar.n % 2 == 0:  # Update every 2 batches
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
        
        # Evaluate on validation set
        val_loss, accuracy, class_report = evaluate_model(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print_classification_report(class_report)
        
        # Update early stopping based on validation loss
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # (Optional) Save the best model here if needed
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
            }, best_model_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    # New argument to limit dataset size
    parser.add_argument('--subset_size', type=int, default=None, 
                        help="Maximum number of examples to use from the combined dataset")
    args = parser.parse_args()

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load datasets
    print("Loading Hugging Face dataset...")
    hf_ds = load_dataset("FastJobs/Visual_Emotional_Analysis")
    
    print("Loading local dataset...")
    local_data = load_local_dataset(
        csv_path='datasets/data.csv',
        dataset_folder='datasets/dataset'
    )
    
    print("Combining datasets...")
    combined_data = combine_datasets(hf_ds['train'], local_data)
    
    # If a subset size is provided and smaller than the combined dataset, sample that many examples.
    if args.subset_size is not None and args.subset_size < len(combined_data):
        print(f"Reducing dataset size to {args.subset_size} examples...")
        combined_data = random.sample(combined_data, args.subset_size)
    
    # Create train/val split
    train_size = int(0.8 * len(combined_data))
    indices = list(range(len(combined_data)))
    random.shuffle(indices)
    
    train_data = [combined_data[i] for i in indices[:train_size]]
    val_data = [combined_data[i] for i in indices[train_size:]]
    
    print(f"Total dataset size: {len(combined_data)}")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = EmotionDataset(train_data, transform)
    val_dataset = EmotionDataset(val_data, transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Calculate class weights for balanced loss
    class_weights = calculate_class_weights(train_data).to(device)
    
    # Initialize model and training components
    model = EmotionRecognitionModel(
        embedding_size=512,
        num_emotions=8,
        freeze_backbone=args.freeze_backbone
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5)
    
    # Train the model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        early_stopping=early_stopping
    )

if __name__ == "__main__":
   main()
