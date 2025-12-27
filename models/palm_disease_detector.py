import torch
import torch.nn as nn
from torchvision import models

# Image (224×224×3)
#     ↓
# ResNet50 Backbone (frozen)
#     ↓
# 2048 features
#     ↓
# Linear(2048→512) + ReLU + Dropout(0.5)
#     ↓
# 512 features
#     ↓
# Linear(512→128) + ReLU
#     ↓
# 128 features
#     ↓
# Linear(128→2)
#     ↓
# 2 logits [healthy, sick]


class PalmDiseaseDetector(nn.Module):
    def __init__(self):
        super(PalmDiseaseDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone weights frozen - only Head will be trained")
        num_features = self.backbone.fc.in_features  # should be 2048

        self.backbone.fc = nn.Sequential(
            # Layer 1: Compression from 2048 to 512
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 2: Additional compression from 512 to 128
            nn.Linear(512, 128),
            nn.ReLU(),
            # Layer 3: Final output - 2 neurons (healthy, sick)
            nn.Linear(128, 2),
        )
        print(f"Model initialized with {num_features} input features")
        print("Head architecture: 2048 -> 512 -> 128 -> 2")

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            # Apply argmax to get predicted class indices
            predictions = torch.argmax(logits, dim=1)
            return predictions
