# entry_model.py
# -------------------------------------------------------------
# This script defines a minimal, scalable chess policy network
# using PyTorch Lightning. It includes:
# - A configurable ResNet with input conv, residual blocks, and head.
# - Training loop with smoothed accuracy/loss tracking.
# - Learning rate logging and scheduler setup.
# Ideal for beginner ML hobbyists to train and deploy chess AIs.
# -------------------------------------------------------------
from collections import deque

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)


class ChessResNet(pl.LightningModule):
    def __init__(self, num_blocks=20, num_channels=208, num_outputs=1968, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.acc_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=100)

        self.input_layer = nn.Sequential(
            nn.Conv2d(18, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_outputs),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.acc_window.append(acc.item())
        self.loss_window.append(loss.item())
        smoothed_acc = sum(self.acc_window) / len(self.acc_window)
        smoothed_loss = sum(self.loss_window) / len(self.loss_window)
        if not torch.isfinite(loss):
            print("NaN detected:", logits)
            raise ValueError("Loss is NaN")

        self.log("train_loss", smoothed_loss, prog_bar=True, on_step=True)
        self.log("train_acc", smoothed_acc, prog_bar=True, on_step=True)

        if self.trainer.lr_scheduler_configs:
            current_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.log("LR", current_lr, prog_bar=True, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500
            }
        }


if __name__ == "__main__":
    model = ChessResNet()
    dummy_input = torch.randn(1, 18, 8, 8)
    summary(model, input_data=dummy_input)
