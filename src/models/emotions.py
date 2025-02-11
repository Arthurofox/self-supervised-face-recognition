import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arcface import ArcFaceBackbone

class EmotionAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class EmotionRecognitionModel(nn.Module):
    def __init__(self, embedding_size=512, num_emotions=8, freeze_backbone=True):
        super().__init__()
        
        self.backbone = ArcFaceBackbone(embedding_size=embedding_size)
        
        # Load pretrained weights
        try:
            checkpoint = torch.load('simclr_arcface_finetuned.pth')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.backbone.'):
                    new_key = k.replace('backbone.backbone.', 'backbone.')
                    new_state_dict[new_key] = v
                elif k.startswith('backbone.'):
                    new_state_dict[k] = v
            
            missing_keys, unexpected_keys = self.backbone.load_state_dict(new_state_dict, strict=False)
            print("Loaded pretrained ArcFace backbone weights")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")

        # More complex emotion classification head
        self.pre_attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.attention = EmotionAttention(embedding_size)
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, num_emotions)
        )
        
        # Residual connections
        self.residual1 = nn.Linear(embedding_size, 512)
        self.residual2 = nn.Linear(512, 256)
        self.residual3 = nn.Linear(256, 128)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get embeddings from backbone
        embeddings = self.backbone(x)
        
        # Pre-attention processing
        x = self.pre_attention(embeddings)
        
        # Apply attention
        x = self.attention(x)
        
        # First block with residual
        res1 = self.residual1(embeddings)
        x = self.emotion_classifier[0:4](x)
        x = x + res1
        
        # Second block with residual
        res2 = self.residual2(x)
        x = self.emotion_classifier[4:8](x)
        x = x + res2
        
        # Third block with residual
        res3 = self.residual3(x)
        x = self.emotion_classifier[8:12](x)
        x = x + res3
        
        # Final classification layers
        x = self.emotion_classifier[12:](x)
        
        return x

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")