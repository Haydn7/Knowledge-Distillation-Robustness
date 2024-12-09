import torch
from torch import nn
import math
from typing import Optional
from utils.tools import round_up_to_power_of_2


# Custom linear module
class GroupLoRALinear(nn.Module):

    __constants__ = ["in_features", "out_features", "block_size", "in_group_size", "rank", "in_group_size", "out_group_size"]
    in_features:        int
    out_features:       int
    rank:               int
    in_group_size:      int
    out_group_size:     int
    groups:             int
    scaling :           torch.Tensor
    weight:             nn.Parameter
    weights_lora_in:    nn.Parameter
    weights_lora_out:   nn.Parameter
    bias:               nn.Parameter

    def __init__(self, teacher: nn.Module, param_dtype: Optional[torch.dtype]=None,
                 block_size: int = 64, rank: int = 64) -> None:
        super().__init__()
        # Check whether it is possible to apply structured sparsity
        self.in_features, self.out_features, self.rank = teacher.in_features, teacher.out_features, rank
        self.groups = min(self.out_features // block_size, self.in_features // block_size)
        if self.out_features % self.groups != 0 or self.in_features % self.groups != 0:
            self.groups = round_up_to_power_of_2(self.groups)

        assert self.groups >= 4, "Expected at least 4 groups otherwise use nn.Linear"
        assert self.out_features % self.groups == 0, f"Out features {self.out_features} not divisible by groups {self.groups}"
        assert self.in_features % self.groups == 0, f"In features {self.out_features} not divisible by groups {self.groups}"

        self.in_group_size = self.in_features // self.groups
        self.out_group_size = self.out_features // self.groups
        dtype = param_dtype if param_dtype is not None else teacher.weight.dtype
        self.weight = nn.Parameter(torch.zeros((self.groups, self.out_group_size, self.in_group_size), dtype=dtype))

        # Copy the diagonal block of the teacher weights to these weights
        with torch.no_grad():
            teacher_ws = teacher.weight.view(self.groups, self.out_group_size, self.groups, self.in_group_size)
            for i in range(self.groups):
                self.weight[i] = teacher_ws[i, :, i, :]

        if teacher.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(teacher.bias.to(dtype).clone())

        # Init LoRA parameters and scaling
        self.scaling = torch.ones(1, dtype=dtype) / self.rank
        self.scaling = self.scaling.to(teacher.weight.device)
        self.lora_in = nn.Parameter(torch.randn(self.rank, self.in_features))
        self.lora_out = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_in, a=math.sqrt(5))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, groups={self.groups}, " + \
            f"in_group_size={self.in_group_size}, out_group_size={self.out_group_size}, bias={self.bias is not None}"

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Pass the input tensor through a grouped (multi-headed) linear and LoRA projections
        :param x_in         shape (batch_size, spatial_size, in_features)
        :returns            shape (batch_size, spatial_size, out_features)
        """
        x = x_in.to(self.weight.dtype)
        batch_size, spatial_size, in_features = x.shape
        x_group = x.view(batch_size, spatial_size, self.groups, self.in_group_size)
        out = torch.einsum("... bi,boi->... bo", x_group, self.weight)    # Perform block matrix multiplication
        lora_out = (x @ self.lora_in.T) @ self.lora_out.T * self.scaling
        out = out.reshape(batch_size, spatial_size, self.out_features) + lora_out
        if self.bias is not None:
            out = out + self.bias
        return out.to(x_in.dtype)
