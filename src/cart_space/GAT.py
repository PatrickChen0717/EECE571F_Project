import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    """
      e_ij = LeakyReLU(a^T [W r_i || W r_j])
      alpha_ij = softmax_{j in N_i}(e_ij)
      rhat_i = sigma( sum_{j in N_i} alpha_ij * W r_j )
    Edge direction: j -> i (src -> dst), softmax over incoming edges per dst.
    """
    def __init__(self, in_dim, out_dim, negative_slope=0.2, dropout=0.0, sigma="elu"):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.view(1, -1))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)
        self.sigma = sigma

    @staticmethod
    def _scatter_softmax(e, dst, N):
        # dst = j
        # e: (B,E), dst: (E,)
        B, E = e.shape
        device = e.device

        max_per_dst = torch.full((B, N), -float("inf"), device=device)
        max_per_dst.scatter_reduce_(1, dst.unsqueeze(0).expand(B, E), e, reduce="amax", include_self=True)

        e = e - max_per_dst.gather(1, dst.unsqueeze(0).expand(B, E))
        exp_e = torch.exp(e)

        denom = torch.zeros((B, N), device=device)
        denom.scatter_add_(1, dst.unsqueeze(0).expand(B, E), exp_e)

        return exp_e / (denom.gather(1, dst.unsqueeze(0).expand(B, E)) + 1e-9)

    def forward(self, r, edge_index, return_alpha=False):
        """
        r: (B,N,Din)
        edge_index: (2,E) [src; dst] (j -> i)
        """
        B, N, Din = r.shape
        src, dst = edge_index
        E = src.numel()

        Wr = self.W(r)                 # (B,N,Dout)
        Wr_i = Wr[:, dst, :]           # (B,E,Dout)  i = dst
        Wr_j = Wr[:, src, :]           # (B,E,Dout)  j = src

        cat = torch.cat([Wr_i, Wr_j], dim=-1)                 # (B,E,2Dout)
        e = self.leaky_relu(torch.matmul(cat, self.a))        # (B,E)

        alpha = self._scatter_softmax(e, dst, N)              # (B,E)
        alpha = self.drop(alpha)

        msg = Wr_j * alpha.unsqueeze(-1)                      # (B,E,Dout)
        out = torch.zeros((B, N, Wr.shape[-1]), device=r.device, dtype=r.dtype)
        out.scatter_add_(1, dst.view(1, E, 1).expand(B, E, Wr.shape[-1]), msg)

        if self.sigma == "elu":
            out = F.elu(out)
        elif self.sigma == "relu":
            out = F.relu(out)
        elif self.sigma is None:
            pass

        return (out, alpha) if return_alpha else out