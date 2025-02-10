import os
import torch

# Data paths
FRAMES_DIR = os.path.join(os.getcwd(), "frames")
FACES_DIR = os.path.join(os.getcwd(), "faces")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "simclr_arcface_finetuned.pth")

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEMPERATURE = 0.5

# Image parameters
IMAGE_SIZE = (112, 112)

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
