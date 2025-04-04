# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/14 14:51
@Auth ： 吕鑫
@File ：modules.py
@IDE ：PyCharm
"""

from torch import nn
import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from models.config import LMConfig


class RMSNorm(torch.nn.Module):
    """
    计算更高效、更稳定
    """

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
        super(RMSNorm, self).__init__()
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


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super(Attention, self).__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.embedding_dim // args.n_heads
        self.wq = nn.Linear(args.embedding_dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.embedding_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.embedding_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.embedding_dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 判断是否有flash_attn，如果存在则使用flash_attn进行加速，否则使用常规的attention计算。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        # 初始化上三角矩阵作为mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # register_buffer存储一些在训练过程中更新的（但不需要梯度的）状态或常量
        self.register_buffer("mask", mask, persistent=False)

    def apply_rotary_emb(self, xq, xk, pos_cis):
        """
        将位置编码（pos_cis）应用到查询Q和键K上，生成旋转嵌入。
        有效结合绝对位置和相对位置信息
        xq：查询张量（Query Tensor），形状是 (batch_size, seq_len, num_heads, head_dim)，表示查询向量。
        xk：键张量（Key Tensor），形状是 (batch_size, seq_len, num_kv_heads, head_dim)，表示键向量。
        pos_cis：位置编码（Position Encodings），形状是 (seq_len, head_dim)，表示每个位置的旋转编码。

        为什么要用虚数： 虚数乘法恰好能够表达旋转，
        """

        def unite_shape(pos_cis, x):
            """
            pos_cis 的形状调整为 (1, seq_len, 1, head_dim) 这样的形状，以便它能与 xq 和 xk 对应的维度进行 逐元素相乘。
            这样，位置编码就可以 作用到每一个头 上，而不是仅仅作用于某一个头。
            """
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert pos_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return pos_cis.view(*shape)

        # 转为虚数
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        # 将位置编码调整为合适的形状
        pos_cis = unite_shape(pos_cis, xq_)
        # 对查询和键进行旋转,并将虚数结果转换回实数形式
        xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        用于重复键K和值V，以便能够在多头注意力中进行计算。
        torch.repeat_interleave(x, dim=2, repeats=n_rep)
        """
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        """
        前向传播：计算注意力输出
        """
        bsz, seq_len, _ = x.shape

        # 计算查询Q，键K和值V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码，RoPE 的旋转机制使注意力权重自然地捕捉到相对位置信息。
        # 旋转位置编码（RoPE），能够有效结合绝对位置和相对位置信息，并且适用于线性自注意力机制。
        xq, xk = self.apply_rotary_emb(xq, xk, pos_cis)

        # kv_cache实现
        # 生成第 t 个 token 时，注意力机制需要用到前 t−1 token 的 Key 和 Value。
        # 如果没有 KV Cache，每次生成新 token 都需要重新计算所有的 K 和 V，这会导致重复计算。
        # 缓存机制：将前 t−1 步计算得到的 Key 和 Value 缓存在内存中。
        # 增量计算：在第 t 步，仅需为当前输入计算新的 Key 和 Value，并将其与缓存的 Key 和 Value 拼接起来。
        if past_key_value is not None:  # 判断是否有缓存
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None  # 更新缓存

        # 实现GQA（Grouped Query Attention）提高计算效率并减少显存占用, MHA与MQA（Mutil-Query Attention）的折中
        xq, xk, xv = (
            xq.transpose(1, 2),
            self.repeat_kv(xk, self.n_rep).transpose(1, 2),
            self.repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:  # flash attention
            # Flash Attention 是对标准缩放点积注意力公式的 硬件优化实现，
            # 通过 内存优化、并行化 和 低级硬件加速，显著提高了大规模模型和长序列输入的处理速度。
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv
