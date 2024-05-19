from torch import nn


class ResNet(nn.Module):
    def __init__(self, blocks, layers, num_classes=100):
        super().__init__()

