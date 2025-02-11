import os
import random
import pandas as pd
from PIL import Image
import torch
from .training_utils import EMOTION_MAPPING
import numpy as np
def load_local_dataset(csv_path, dataset_folder):
    """Load local dataset from CSV file and folder"""
    df = pd.read_csv(csv_path)
    
    local_to_global = {
        'angry': 0,    # anger
        'happy': 4,    # happy
        'neutral': 5,  # neutral
        'sad': 6,      # sad
        'surprise': 7  # surprise
    }
    
    local_data = []
    skipped_files = 0
    
    for _, row in df.iterrows():
        # Try different path combinations
        potential_paths = [
            os.path.join(dataset_folder, row['path']),
            os.path.join(dataset_folder, row['label'], row['path']),
            os.path.join(dataset_folder, row['label'].capitalize(), row['path'])
        ]
        
        image_loaded = False
        for img_path in potential_paths:
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    local_data.append({
                        'image': image,
                        'label': local_to_global[row['label'].lower()]
                    })
                    image_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        if not image_loaded:
            skipped_files += 1
    
    print(f"Successfully loaded {len(local_data)} images")
    print(f"Skipped {skipped_files} files")
    
    # Print class distribution
    labels = [item['label'] for item in local_data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution in local dataset:")
    for label, count in zip(unique_labels, counts):
        emotion = EMOTION_MAPPING[label]
        print(f"{emotion:8s}: {count}")
    
    return local_data


def combine_datasets(huggingface_ds, local_data):
    """Combine Hugging Face dataset with local dataset"""
    hf_data = [
        {
            'image': item['image'],
            'label': item['label']
        } 
        for item in huggingface_ds
    ]
    
    return hf_data + local_data

def calculate_class_weights(dataset):
    """Calculate class weights for balanced loss"""
    labels = [item['label'] for item in dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()
    return weights * len(weights)  # Scaling by number of classes