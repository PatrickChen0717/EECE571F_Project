import torch
import torch.nn as nn
from src.resnet import ResNetEncoder
from src.dinov2_encoder import DINOv2Encoder

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
    
    

class FullModelWithDINOv2(nn.Module):
    def __init__(self, traj_encoder, vision_dim=256, fuse_dim=128, use_visual_diff=True):
        super().__init__()
        self.encoder = traj_encoder
        self.vision_dim = vision_dim
        self.use_visual_diff = use_visual_diff

        self.vision_encoder = DINOv2Encoder(
            model_name="facebook/dinov2-small",
            out_dim=vision_dim,
            freeze=True,
            use_cls=False
        )

        Dgat = traj_encoder.gat_out_dim
        in_dim = Dgat + vision_dim + (vision_dim if use_visual_diff else 0)

        self.fuse = nn.Sequential(
            nn.Linear(in_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, 3)
        )

    def forward(self, delta, frames=None, disable_visual=False):
        """
        delta:  (B,T,M,5,2)
        frames: (B,T,3,224,224)
        """
        _, rhat = self.encoder(delta)   # (B,T,N,Dgat)
        B, T, N, D = rhat.shape

        if disable_visual:
            img_feat = torch.zeros(B, T, self.vision_dim, device=rhat.device, dtype=rhat.dtype)
        else:
            if frames is None:
                raise ValueError("frames must be provided when disable_visual=False")
            img_feat = self.vision_encoder(frames)   # (B,T,vision_dim)
            img_feat = img_feat.to(device=rhat.device, dtype=rhat.dtype)

        feats = [img_feat]

        if self.use_visual_diff:
            diff = torch.zeros_like(img_feat)
            diff[:, 1:] = img_feat[:, 1:] - img_feat[:, :-1]
            feats.append(diff)

        vis = torch.cat(feats, dim=-1)               # (B,T,Dv)
        vis = vis[:, :, None, :].expand(B, T, N, vis.shape[-1])

        fused = torch.cat([rhat, vis], dim=-1)       # (B,T,N,*)
        out = self.fuse(fused)                       # (B,T,N,2)
        return out