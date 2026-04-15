import torch
import torch.nn as nn
from src.GAT import GAT


class LSTM_gat(nn.Module):
    def __init__(self, hidden_size: int = 32, embed_dim: int = 16):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # Φ embedding: [dx, dy, vis] -> v_t^i
        self.phi = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # LSTMspa
        self.lstm_spa = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
        )

        # BatchNorm before attention, per paper
        self.bn_before_gat = nn.BatchNorm1d(hidden_size)

        # Two-layer GAT, per paper
        self.gat1 = GAT(in_dim=hidden_size, out_dim=hidden_size, dropout=0.0, sigma="elu")
        self.gat2 = GAT(in_dim=hidden_size, out_dim=hidden_size, dropout=0.0, sigma="elu")
        
        # LSTMtemp after GAT, per paper
        self.lstm_temp = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
        )

        # output dims
        self.gat_out_dim = hidden_size 
        self.out_dim = hidden_size * 2          # [r || l]

    @staticmethod
    def add_virtual_root(feat):
        """
        feat: (B,T,M,K,3), last dim = [dx,dy,vis] or [x,y,vis]
        returns: (B,T,M,K+1,3), root appended last
        """
        xy = feat[..., :2]
        vis = feat[..., 2] > 0.5

        w = vis.unsqueeze(-1).float()
        denom = w.sum(dim=3, keepdim=True).clamp_min(1.0)
        root_xy = (xy * w).sum(dim=3, keepdim=True) / denom

        root_vis = vis.any(dim=3, keepdim=True).float().unsqueeze(-1)
        root_feat = torch.cat([root_xy, root_vis], dim=-1)

        return torch.cat([feat, root_feat], dim=3)

    @staticmethod
    def build_edge_index(num_instruments, K_with_root, device):
        K = K_with_root
        edges = []

        # intra-instrument: kp <-> root
        for m in range(num_instruments):
            base = m * K
            root = base + (K - 1)
            for kp in range(K - 1):
                u = base + kp
                edges.append((u, root))
                edges.append((root, u))

        # inter-instrument: root <-> root
        roots = [m * K + (K - 1) for m in range(num_instruments)]
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                ri, rj = roots[i], roots[j]
                edges.append((ri, rj))
                edges.append((rj, ri))

        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _apply_bn(self, x):
        """
        x: (B,N,H) -> BN over H
        """
        B, N, H = x.shape
        x = x.reshape(B * N, H)
        x = self.bn_before_gat(x)
        x = x.reshape(B, N, H)
        return x

    def forward(self, feat):
        """
        feat: (B,T,M,5,3)

        returns:
          r:    (B,T,N,H)     from LSTMspa
          h:    (B,T,N,2H)    = [r || l], closer to paper
        """
        B, T, M, K, C = feat.shape
        assert C == 3, f"Expected last dim=3, got {C}"

        # add virtual root -> 6 nodes per instrument
        featR = self.add_virtual_root(feat)     # (B,T,M,6,3)
        KR = featR.shape[3]
        N = M * KR

        # node embedding
        v = self.phi(featR)                     # (B,T,M,KR,E)
        v = v.reshape(B, T, N, self.embed_dim)

        # LSTMspa over time, shared across nodes
        v_lstm = v.permute(1, 0, 2, 3).contiguous().reshape(T, B * N, self.embed_dim)
        r_lstm, _ = self.lstm_spa(v_lstm)       # (T,B*N,H)

        H = r_lstm.shape[-1]
        r = r_lstm.reshape(T, B, N, H).permute(1, 0, 2, 3).contiguous()  # (B,T,N,H)

        edge_index = self.build_edge_index(M, KR, device=feat.device)

        # Two-layer GAT at each time step
        rhat_list = []
        for t in range(T):
            x_t = r[:, t]                       # (B,N,H)
            x_t = self._apply_bn(x_t)           # BN before attention
            x_t = self.gat1(x_t, edge_index)    # layer 1
            x_t = self.gat2(x_t, edge_index)    # layer 2
            rhat_list.append(x_t)

        rhat = torch.stack(rhat_list, dim=1)    # (B,T,N,H)

        # LSTMtemp over time on GAT output
        temp_in = rhat.permute(1, 0, 2, 3).contiguous().reshape(T, B * N, H)
        l_lstm, _ = self.lstm_temp(temp_in)     # (T,B*N,H)
        l = l_lstm.reshape(T, B, N, H).permute(1, 0, 2, 3).contiguous()  # (B,T,N,H)

        # paper Eq. (6): h = [r || l]
        h = torch.cat([r, l], dim=-1)           # (B,T,N,2H)

        return r, h