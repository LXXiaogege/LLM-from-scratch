# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/13 21:35
@Auth ： 吕鑫
@File ：model.py
@IDE ：PyCharm

从零开始实现LLM！
"""

from torch import nn
from models.modules import RMSNorm
import torch.nn.functional as F
from models.modules import Attention
from models.config import LMConfig


class LLMModel(nn.Module):
    def __init__(self, n_layers):
        super(LLMModel).__init__()
        # embedding layer
        self.embedding = PositionEmbedding()
        # transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(n_layers)]
        )
        # softmax layer
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            embedded = transformer_block(embedded)
        outputs = self.softmax(embedded)
        return outputs


class PositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(PositionEmbedding).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, inputs):
        return self.embedding(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, norm_type='rms'):
        super(TransformerBlock).__init__()
        self.args = LMConfig()
        self.attn_norm = RMSNorm(self.args.norm_dim)
        self.attention = Attention(args=self.args)
        self.ffn_norm = RMSNorm(self.args.norm_dim)
        self.feed_forward = FeedForwardLayer()

    def forward(self, inputs):
        att_inputs = self.attn_norm(inputs)
        att_outputs = self.attention(att_inputs)
        # residual connection
        att_outputs = inputs + att_outputs

        fead_forward_inputs = self.ffn_norm(att_outputs)
        feed_forward_output = self.feed_forward(fead_forward_inputs)
        # residual connection
        feed_forward_output = att_outputs + feed_forward_output
        return feed_forward_output


class FeedForwardLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SiLU 函数的优点在于它能够平滑地处理负值，并且在正值输入时提供较大的输出范围.它有助于保持梯度的流动，避免梯度消失问题。
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
