# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/14 14:51
@Auth ： 吕鑫
@File ：modules.py
@IDE ：PyCharm
"""

from torch import nn
import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.用来避免除零操作，增加数值的稳定性
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # 将输入张量 x 转换为浮点数类型（x.float()），这是为了确保在计算过程中不会因整数类型导致精度问题
        # type_as(x)）。这种做法是为了避免因类型不匹配导致的计算问题。
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
