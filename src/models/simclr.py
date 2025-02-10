import torch.nn as nn

class SimCLRModel(nn.Module):
    """
    Combined model for contrastive learning: ArcFace backbone with a projection head.
    """
    def __init__(self, backbone, projection_head):
        super(SimCLRModel, self).__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z
