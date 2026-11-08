import torch.nn as nn


class CatDogClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(64, 32, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(32, 16, kernel_size, stride, padding)

        self.act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = conv(x)
            x = self.act(x)
            x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x