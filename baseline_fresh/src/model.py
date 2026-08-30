import torch
import torch.nn as nn
from torchvision import models


class FERResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)  # 512-d, post-avgpool, post-ReLU (non-negative)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits
