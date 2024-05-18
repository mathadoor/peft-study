import torch
from torch import nn


class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, rank, init_type=0):
        super().__init__()

        self.A = nn.Parameter(torch.Tensor(in_features, rank))
        self.B = nn.Parameter(torch.Tensor(rank, out_features))

        self.reset_parameters(init_type)

    def reset_parameters(self, init_type):
        if init_type == 0:
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)
        elif init_type == 1:
            nn.init.zeros_(self.A)
            nn.init.normal_(self.B, mean=0, std=0.01)
        elif init_type == 2:
            nn.init.normal_(self.A, mean=0, std=0.01)
            nn.init.zeros_(self.B)
        else:
            nn.init.normal_(self.A, mean=0, std=0.01)
            nn.init.normal_(self.B, mean=0, std=0.01)

    def forward(self, x):
        x = torch.matmul(x, self.A)
        x = torch.matmul(x, self.B)
        return x


class LoRALinearLayer(nn.Module):

    def __init__(self, linear_layer, rank, init_type=0):
        super().__init__()
        self.lora = LoRALinear(linear_layer.in_features, linear_layer.out_features, rank, init_type)
        self.linear = linear_layer

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.lora(x)
        x = x1 + x2
        return x
