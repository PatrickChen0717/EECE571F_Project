import torch
import torch.nn as nn
import torch.nn.functional as F

class FullModel(nn.Module):
    def __init__(self, encoder, gat_out_dim, pred_hid=128):
        super().__init__()
        self.encoder = encoder
        self.pred_head = nn.Sequential(
            nn.Linear(gat_out_dim, pred_hid),
            nn.ReLU(),
            nn.Linear(pred_hid, 2)   # predict (dx, dy)
        )

    def forward(self, pos):
        # pos: (B,T,M,5,2)
        r, rhat = self.encoder(pos)     # rhat: (B,T,N,Dgat)
        dpos_hat = self.pred_head(rhat)             # (B,T,N,2)
        return dpos_hat