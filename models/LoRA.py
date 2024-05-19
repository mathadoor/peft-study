import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Linear layer for Low-Rank Adaptation. The decomposed matrices can be initialized as zeros or normal.
    """
    def __init__(self, in_features, out_features, rank, init_type=0):
        """
        :param in_features: number of input features
        :param out_features: number of output features
        :param rank: rank of the decomposition
        :param init_type: initialization type (0: zeros, 1: normal for B, 2: normal for A, 3: normal for both A and B)
        """
        super().__init__()

        self.A = nn.Parameter(torch.Tensor(rank, out_features))
        self.B = nn.Parameter(torch.Tensor(in_features, rank))

        self.reset_parameters(init_type)

    def reset_parameters(self, init_type):
        """
        Initialize the decomposed matrices.
        :param init_type: See __init__ for details
        """
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
        """
        Forward pass of the LoRALinear layer. A @ (B @ x); Number of operations is (out + in) * rank instead of in * out.
        :param x:
        :return:
        """
        x = torch.matmul(self.A, x)
        x = torch.matmul(self.B, x)
        return x


class LoRALinearLayer(nn.Module):
    """
    A wrapper for the LoRALinear layer. It combines the original linear layer with the LoRALinear layer(see above).
    """
    def __init__(self, linear_layer, rank, init_type=0):
        """
        Creates a LoRA Linear Layer for an input Linear Layer. The linear layer is frozen.
        """
        super().__init__()
        self.lora = LoRALinear(linear_layer.in_features, linear_layer.out_features, rank, init_type)
        self.linear = linear_layer
        self.rank = rank
        self.init_type = init_type

        self.freeze_layer(self.linear)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.lora(x)
        x = x1 + x2
        return x

    def freeze_layer(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def __repr__(self):
        return f"LoRALinear(in_features={self.linear.in_features}, in_features={self.linear.out_features}, rank=${self.rank}, init_type=${str(self.init_type)})"