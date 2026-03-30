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
            nn.Linear(3, embed_dim),
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
    def add_virtual_root(feat):
        """
        feat: (B,T,M,K,3)
            last dim = [a, b, vis]
            where [a,b] can be [dx,dy] or [x,y]

        returns:
        featR: (B,T,M,K+1,3)
                root appended LAST
        """
        xy = feat[..., :2]                       # (B,T,M,K,2)
        vis = feat[..., 2] > 0.5                # (B,T,M,K)

        w = vis.unsqueeze(-1).float()           # (B,T,M,K,1)
        denom = w.sum(dim=3, keepdim=True).clamp_min(1.0)
        root_xy = (xy * w).sum(dim=3, keepdim=True) / denom   # (B,T,M,1,2)

        # root visible if any child visible
        root_vis = vis.any(dim=3, keepdim=True).float().unsqueeze(-1)  # (B,T,M,1,1)

        root_feat = torch.cat([root_xy, root_vis], dim=-1)   # (B,T,M,1,3)

        return torch.cat([feat, root_feat], dim=3)           # (B,T,M,K+1,3)

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

    def forward(self, feat):
        """
        feat: (B, T, M, K, 3)
            e.g. [dx, dy, vis] or [x, y, vis]

        returns:
        r:    (B, T, N, Dlstm)
        rhat: (B, T, N, Dgat)
        """
        B, T, M, K, C = feat.shape
        assert C == 3, f"Expected last dim = 3, got {C}"

        # add virtual root per instrument
        featR = self.add_virtual_root(feat)     # (B,T,M,K+1,3)
        KR = featR.shape[3]
        N = M * KR

        # node embedding
        v = self.phi(featR)                     # (B,T,M,KR,embed_dim)
        v = v.reshape(B, T, N, self.embed_dim)

        # shared temporal LSTM over each node
        v_lstm = v.permute(1, 0, 2, 3).contiguous().reshape(T, B * N, self.embed_dim)
        r_lstm, _ = self.lstm_spa(v_lstm)       # (T,B*N,H)

        Dl = r_lstm.shape[-1]
        r = r_lstm.reshape(T, B, N, Dl).permute(1, 0, 2, 3).contiguous()  # (B,T,N,H)

        # graph edges
        edge_index = self.build_edge_index(M, KR, device=feat.device)

        # GAT at each time step
        rhat_list = []
        for t in range(T):
            rhat_t = self.gat(r[:, t], edge_index)   # (B,N,Dgat)
            rhat_list.append(rhat_t)

        rhat = torch.stack(rhat_list, dim=1)         # (B,T,N,Dgat)

        return r, rhat