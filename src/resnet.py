import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, out_dim=256, freeze=True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, out_dim)

        if freeze:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B,3,224,224)
        feat = self.feature_extractor(x)   # (B,2048,1,1)
        feat = feat.flatten(1)             # (B,2048)
        feat = self.proj(feat)             # (B,out_dim)
        return feat