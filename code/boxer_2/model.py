from torch import nn
import torch.nn.functional as F


class KanjiBoxer(nn.Module):
    def __init__(self, output_dimensions: int):
        super(KanjiBoxer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv18 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.linear = nn.Linear(256 * 2 * 2, output_dimensions)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        before = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x += before

        before = x
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x += before

        before = x
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x += before

        before = x
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x += before

        before = x
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x += before

        x = self.pool(x)

        before = x
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x += before

        x = x.view(-1, 256 * 2 * 2)
        x = self.linear(x)

        return x
