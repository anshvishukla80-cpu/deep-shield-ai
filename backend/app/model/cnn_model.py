import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """
    Transfer learning model using pretrained Xception architecture.
    Xception provides state-of-the-art accuracy for deepfake detection tasks.
    We freeze the early layers to retain pretrained features and only
    train the later layers and the classification head.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model('xception', pretrained=pretrained)

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 blocks + classifier head
        blocks_to_unfreeze = ['block12', 'block11', 'fc']
        for name, param in self.backbone.named_parameters():
            if any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True

        # Replace the final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)