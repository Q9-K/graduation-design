import torch.nn as nn
import torch.nn.functional as F
import torch



class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    model = Net()
    input = torch.randn(1, 1, 32, 32)
    output = model(input)
    print(output.shape)