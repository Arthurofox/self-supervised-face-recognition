import os
import torch

# Data paths
FRAMES_DIR = os.path.join(os.getcwd(), "frames")
FACES_DIR = os.path.join(os.getcwd(), "faces")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "simclr_arcface_finetuned.pth")

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1

# Scheduler config
SCHEDULER_CONFIG = {
    'name': 'OneCycleLR',
    'params': {
        'pct_start': 0.1,        # 10% warmup period
        'div_factor': 10,        # Initial learning rate = max_lr/10
        'final_div_factor': 100, # Final learning rate = max_lr/1000
    }
}

# Image parameters
IMAGE_SIZE = (112, 112)

def get_optimal_device():
    if torch.backends.mps.is_available():
        # For newer PyTorch versions, check if MPS is properly built
        if torch.backends.mps.is_built():
            try:
                # Test MPS by creating a small tensor
                torch.zeros(1).to('mps')
                return torch.device("mps")
            except Exception:
                print("MPS (Metal) is available but not working properly")
        
    if torch.cuda.is_available():
        return torch.device("cuda")
        
    return torch.device("cpu")

# Use this instead of your current DEVICE definition
DEVICE = get_optimal_device()