import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
