import torch
import torch.nn as nn

class LSTMOnlyModel(nn.Module):
    def __init__(self, M=2, hidden_size=128, num_layers=2):
        super().__init__()
        self.M = M
        self.N = M * 6
        self.input_dim = M * 5 * 3   # (dx, dy, valid)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.head = nn.Linear(hidden_size, self.N * 2)

    def forward(self, delta_in, frame_feats=None):
        # delta_in: (B,T,M,5,3)

        B, T, M, K, C = delta_in.shape

        x = delta_in.reshape(B, T, -1)   # (B,T,M*5*3)

        out, _ = self.lstm(x)            # (B,T,H)
        out = self.head(out)             # (B,T,N*2)

        out = out.view(B, T, self.N, 2)  # (B,T,N,2)
        return out