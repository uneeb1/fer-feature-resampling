import torch
import torch.nn as nn
from torchvision import models


class FERResNet18(nn.Module):
    def __init__(self, num_classes=7, freeze_stages=None):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

        if freeze_stages:
            self._freeze(freeze_stages)

    def _freeze(self, stage_names):
        stage_map = {
            "conv1": [self.conv1],
            "bn1": [self.bn1],
            "layer1": [self.layer1],
            "layer2": [self.layer2],
            "layer3": [self.layer3],
            "layer4": [self.layer4],
        }
        for name in stage_names:
            for module in stage_map[name]:
                for p in module.parameters():
                    p.requires_grad = False

    def param_groups(self, base_lr, lr_mults=None):
        if lr_mults is None:
            return [{"params": self.parameters(), "lr": base_lr}]
        groups = []
        stage_params = {
            "layer3": self.layer3.parameters(),
            "layer4": self.layer4.parameters(),
            "fc": self.fc.parameters(),
        }
        assigned = set()
        for name, mult in lr_mults.items():
            params = list(stage_params[name])
            ids = {id(p) for p in params}
            assigned |= ids
            groups.append({"params": params, "lr": base_lr * mult})
        remaining = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]
        if remaining:
            groups.insert(0, {"params": remaining, "lr": base_lr})
        return groups

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits
