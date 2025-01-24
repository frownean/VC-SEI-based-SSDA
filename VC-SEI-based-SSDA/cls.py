import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1152, 520),
            nn.Linear(520, 16),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Net()
    input = torch.randn((32, 1152))
    output = model(input)
    print(output.shape)
