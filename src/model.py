import torch
import torch.nn as nn
from src.resnet import ResNetEncoder
from src.dinov2_encoder import DINOv2Encoder

class FullModelWithResNet(nn.Module):
    def __init__(self, traj_encoder, vision_dim=256, fuse_dim=128):
        super().__init__()
        self.encoder = traj_encoder
        self.vision_encoder = ResNetEncoder(out_dim=vision_dim, freeze=True)

        Dgat = traj_encoder.out_dim

        self.fuse = nn.Sequential(
            nn.Linear(Dgat + vision_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, 2)
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

        Dgat = traj_encoder.out_dim
        vis_dim = vision_dim * 2 if self.use_visual_diff else vision_dim

        self.fuse_in = nn.Sequential(
            nn.Linear(Dgat + vis_dim, fuse_dim),
            nn.ReLU(),
        )

        self.post_gru = nn.GRU(
            input_size=fuse_dim,
            hidden_size=fuse_dim,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, 2)
        )

    def forward(self, delta, vis_feats):
        """
        delta:     (B,T,M,5,3)
        vis_feats: (B,T,V)   precomputed DINO features

        returns:
        out: (B,T,N,2)
        """
        B, T, M, K, C = delta.shape

        _, rhat = self.encoder(delta)                 # (B,T,N,Dg)
        B, T, N, Dg = rhat.shape
        
        vis = vis_feats                               # (B,T,V)

        if self.use_visual_diff:
            vis_prev = torch.cat([vis[:, :1], vis[:, :-1]], dim=1)
            vis_diff = vis - vis_prev
            vis = torch.cat([vis, vis_diff], dim=-1)  # (B,T,2V)

        vis = vis.unsqueeze(2).expand(-1, -1, N, -1)  # (B,T,N,V or 2V)

        fused = torch.cat([rhat, vis], dim=-1)        # (B,T,N,Dg+vis_dim)
        fused = self.fuse_in(fused)                   # (B,T,N,fuse_dim)

        fused = fused.permute(0, 2, 1, 3).contiguous()   # (B,N,T,F)
        fused = fused.view(B * N, T, -1)                 # (B*N,T,F)

        fused, _ = self.post_gru(fused)                  # (B*N,T,F)

        out = self.head(fused)                           # (B*N,T,2)
        out = out.view(B, N, T, 2).permute(0, 2, 1, 3).contiguous()  # (B,T,N,2)

        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
