import torch.nn as nn
import torchvision.models as models

class ArcFaceBackbone(nn.Module):
    """
    ArcFace Backbone using a pre-trained ResNet50.
    """
    def __init__(self, embedding_size=512):
        super(ArcFaceBackbone, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        return self.backbone(x)
