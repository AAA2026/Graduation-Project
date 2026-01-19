import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2_LSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=128, lstm_layers=1, dropout=0.5):
        super().__init__()

        try:
            base = mobilenet_v2(weights="DEFAULT")
        except:
            base = mobilenet_v2(pretrained=True)

        self.backbone = nn.Sequential(*list(base.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        feats = feats.view(B, T, self.feature_dim)

        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
