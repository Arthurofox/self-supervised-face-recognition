import torch
from src.models.arcface import ArcFaceBackbone
from src.models.projection import ProjectionHead
from src.models.simclr import SimCLRModel

def test_forward():
    backbone = ArcFaceBackbone(embedding_size=512)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
    model = SimCLRModel(backbone, projection_head)
    dummy_input = torch.randn(1, 3, 112, 112)
    h, z = model(dummy_input)
    assert h.shape[0] == 1
    assert z.shape[0] == 1
    print("Forward pass test successful.")

if __name__ == "__main__":
    test_forward()
