import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=512, dropout=0.1):
        super().__init__()

        def block(ic, oc, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, k, s, p, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout * 0.5),  # Light dropout for CNN layers
            )

        # 8-layer CNN backbone with residual shortcuts and dropout
        self.cnn = nn.Sequential(
            block(3, 64),
            block(64, 64),
            nn.MaxPool2d(2, 2),  # 32×240
            block(64, 128),
            block(128, 128),
            nn.MaxPool2d(2, 2),  # 16×120
            block(128, 256),
            block(256, 256),  # 16×120
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8×120
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4×120
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 3×119
        )

        # 4-layer Bi-LSTM with increased dropout
        self.rnn = nn.LSTM(
            512,
            hidden_size,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Add dropout before final classification layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):  # x: [B,3,64,480]
        f = self.cnn(x)  # [B,512,3,W]
        f = nn.functional.adaptive_avg_pool2d(f, (1, f.shape[-1]))
        f = f.squeeze(2).permute(0, 2, 1)  # [B,W,512]
        f, _ = self.rnn(f)  # [B,W,1024]
        f = self.dropout(f)  # Apply dropout before classification
        return self.fc(f).permute(1, 0, 2)  # [W,B,C] for CTC
