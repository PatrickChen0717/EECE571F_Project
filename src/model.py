import torch
import torch.nn as nn
from src.resnet import ResNetEncoder

class FullModelWithResNet(nn.Module):
    def __init__(self, traj_encoder, vision_dim=256, fuse_dim=128):
        super().__init__()
        self.encoder = traj_encoder
        self.vision_encoder = ResNetEncoder(out_dim=vision_dim, freeze=True)

        Dgat = traj_encoder.gat_out_dim

        self.fuse = nn.Sequential(
            nn.Linear(Dgat + vision_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, 2)   # predict delta x,y per node
        )

    def forward(self, delta, frame):
        """
        delta: (B,T,M,5,2)
        frame: (B,3,224,224)
        returns: (B,T,N,2)
        """
        _, rhat = self.encoder(delta)              # (B,T,N,Dgat)
        B, T, N, D = rhat.shape

        img_feat = self.vision_encoder(frame)      # (B,vision_dim)
        img_feat = img_feat[:, None, None, :].expand(B, T, N, -1)

        fused = torch.cat([rhat, img_feat], dim=-1)   # (B,T,N,D+vision_dim)
        out = self.fuse(fused)                        # (B,T,N,2)
        return out