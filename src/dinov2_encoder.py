import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv2Encoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", out_dim=256, freeze=True, use_cls=False):
        super().__init__()

        self.use_cls = use_cls
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.proj = nn.Linear(hidden, out_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B,3,H,W), already a torch tensor
        returns: (B,out_dim)
        """
        # DINOv2 in transformers expects pixel_values.
        if x.ndim == 4:
            outputs = self.backbone(pixel_values=x)
            tokens = outputs.last_hidden_state   # (B, 1+num_patches, hidden)

            if self.use_cls:
                feat = tokens[:, 0]              # (B, hidden)
            else:
                feat = tokens[:, 1:].mean(dim=1) # (B, hidden)

            feat = self.proj(feat)               # (B, out_dim)
            return feat

        elif x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

            outputs = self.backbone(pixel_values=x)
            tokens = outputs.last_hidden_state   # (B*T, 1+num_patches, hidden)

            if self.use_cls:
                feat = tokens[:, 0]              # (B*T, hidden)
            else:
                feat = tokens[:, 1:].mean(dim=1) # (B*T, hidden)

            feat = self.proj(feat)               # (B*T, out_dim)
            feat = feat.reshape(B, T, -1)        # (B, T, out_dim)
            return feat

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")