import math
import torch
import torch.nn as nn

class TransformerTrajectoryModel(nn.Module):
    """
    Input:
      delta_in:    (B,T,M,5,3)   [dx, dy, valid]
      frame_feats: (B,T,Dv)

    Output:
      pred_delta:  (B,T,N,2), where N = M * 6
    """
    def __init__(
        self,
        M=2,
        vision_dim=128,
        d_model=256,
        nhead=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.M = M
        self.N = M * 6
        self.d_model = d_model


        # per-node delta embedding: [dx, dy, valid] -> token dim
        self.node_in = nn.Linear(3, d_model)

        # frame feature embedding
        self.vision_in = nn.Linear(vision_dim, d_model)

        # fuse node token + frame token
        self.fuse = nn.Linear(d_model * 2, d_model)

        # learnable positional embedding over time
        self.time_pos = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, delta_in, frame_feats):
        """
        delta_in:    (B,T,M,5,3)
        frame_feats: (B,T,Dv)
        returns:     (B,T,M*6,2)
        """
        B, T, M, K, C = delta_in.shape
        assert M == self.M, f"Expected M={self.M}, got {M}"
        assert K == 5 and C == 3, f"Expected (B,T,M,5,3), got {delta_in.shape}"

        # build 6 nodes by appending a virtual root from absolute positions proxy
        # here delta_in only has delta+valid, so for the transformer input we use:
        # 5 nodes from the observed deltas, plus 1 root token = mean of valid child deltas
        delta_xy = delta_in[..., :2]           # (B,T,M,5,2)
        valid5 = delta_in[..., 2] > 0.5        # (B,T,M,5)

        valid5_f = valid5.unsqueeze(-1).float()
        denom = valid5_f.sum(dim=3, keepdim=True).clamp_min(1.0)
        root_xy = (delta_xy * valid5_f).sum(dim=3, keepdim=True) / denom
        root_valid = (valid5.any(dim=3, keepdim=True)).float()

        root_token = torch.cat([root_xy, root_valid.unsqueeze(-1)], dim=-1)  # (B,T,M,1,3)
        node_tokens = torch.cat([delta_in, root_token], dim=3)               # (B,T,M,6,3)

        # flatten nodes
        x = self.node_in(node_tokens)                # (B,T,M,6,d_model)
        x = x.reshape(B, T, self.N, self.d_model)   # (B,T,N,d_model)

        # add frame feature to every node at the same time step
        vf = self.vision_in(frame_feats)             # (B,T,d_model)
        vf = vf.unsqueeze(2).expand(-1, -1, self.N, -1)

        x = self.fuse(torch.cat([x, vf], dim=-1))   # (B,T,N,d_model)

        # run temporal transformer independently per node
        x = x.permute(0, 2, 1, 3).reshape(B * self.N, T, self.d_model)  # (B*N,T,d)
        x = x + self.time_pos[:, :T, :]
        x = self.transformer(x)                                          # (B*N,T,d)

        out = self.head(x)                                               # (B*N,T,2)
        out = out.reshape(B, self.N, T, 2).permute(0, 2, 1, 3)          # (B,T,N,2)
        return out