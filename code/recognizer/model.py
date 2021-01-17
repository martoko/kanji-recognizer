from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class KanjiRecognizer(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int):
        super(KanjiRecognizer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)  # 32 -> 28
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)  # 14 -> 10
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)  # 10 -> 6
        self.linear = nn.Linear(64 * 6 * 6, output_dimensions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 6 * 6)
        x = self.linear(x)
        return x
