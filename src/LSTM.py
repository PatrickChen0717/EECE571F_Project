import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.GAT import GAT


class LSTM_gat(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int = 64):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # Φ embedding
        self.phi = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Shared spatial LSTM
        self.lstm_spa = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
        )

        self.gat_out_dim = hidden_size
        self.gat = GAT(in_dim=hidden_size, out_dim=self.gat_out_dim, dropout=0.0, sigma="elu")

    @staticmethod
    def add_virtual_root(pos, valid_xy=None):  # pos: (B,T,M,K,2), K=5
        if valid_xy is None:
            # valid if finite
            valid_xy = torch.isfinite(pos).all(dim=-1)  # (B,T,M,K)

        pos_clean = torch.nan_to_num(pos, nan=0.0)
        w = valid_xy.unsqueeze(-1).float()              # (B,T,M,K,1)
        denom = w.sum(dim=3, keepdim=True).clamp_min(1.0)  # (B,T,M,1,1)
        root = (pos_clean * w).sum(dim=3, keepdim=True) / denom  # (B,T,M,1,2)

        return torch.cat([pos_clean, root], dim=3)      # (B,T,M,6,2) root LAST

    @staticmethod
    def build_edge_index(num_instruments, K_with_root, device):
        K = K_with_root
        edges = []

        # intra instrument: kp <-> root
        for m in range(num_instruments):
            base = m * K
            root = base + (K - 1)
            for kp in range(K - 1):
                u = base + kp
                edges.append((u, root))
                edges.append((root, u))

        # inter instrument: connect roots
        roots = [m * K + (K - 1) for m in range(num_instruments)]
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                ri, rj = roots[i], roots[j]
                edges.append((ri, rj))
                edges.append((rj, ri))

        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # (2,E)

    def forward(self, pos):
        """
        pos: (B, T, M, K, 2)
          B batch
          T timesteps
          M instruments
          K keypoints per instrument (without root)
        returns:
          r:    (B, T, N, Dlstm)   node temporal features
          rhat: (B, T, N, Dgat)    after GAT interaction
        """
        B, T, M, K, _ = pos.shape

        # add root => K+1 nodes per instrument
        valid_xy = torch.isfinite(pos).all(dim=-1)      # (B,T,M,5)
        posR = self.add_virtual_root(pos, valid_xy)  
        KR = posR.shape[3]
        N = M * KR

        # Δp_t
        delta = posR[:, 1:] - posR[:, :-1]         # (B,T-1,M,KR,2)
        delta0 = torch.zeros_like(posR[:, :1])     # (B,1,M,KR,2)
        delta = torch.cat([delta0, delta], dim=1)  # (B,T,M,KR,2)

        # Φ embedding
        v = self.phi(delta)                         # (B,T,M,KR,embed)
        v = v.view(B, T, N, self.embed_dim)         # (B,T,N,embed)

        # shared LSTM across nodes: flatten nodes into batch
        v_lstm = v.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.embed_dim)  # (T,B*N,embed)
        r_lstm, _ = self.lstm_spa(v_lstm)                                                # (T,B*N,Dlstm)
        Dl = r_lstm.shape[-1]
        r = r_lstm.view(T, B, N, Dl).permute(1, 0, 2, 3).contiguous()                    # (B,T,N,Dl)

        # build edges for this M (safe if M varies)
        edge_index = self.build_edge_index(M, KR, device=pos.device)                     # (2,E)

        # GAT per time step. If your GAT isn't batched, loop batch too.
        rhat = torch.empty((B, T, N, self.gat_out_dim), device=pos.device, dtype=r.dtype)
        edge_index = edge_index.to(pos.device)

        rhat_list = []
        for t in range(T):
            rhat_t = self.gat(r[:, t], edge_index)   # (B, N, Dgat)
            rhat_list.append(rhat_t)

        rhat = torch.stack(rhat_list, dim=1)         # (B, T, N, Dgat)

        return r, rhat
