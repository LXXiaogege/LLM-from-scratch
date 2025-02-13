# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/13 21:35
@Auth ： 吕鑫
@File ：model.py
@IDE ：PyCharm

从零开始实现LLM！
"""

from torch import nn


class LLMModel(nn.Module):
    def __init__(self, n_layers):
        super(LLMModel).__init__()
        # embedding layer
        self.embedding = nn.Embedding()
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


class TransformerBlock(nn.Module):
    def __init__(self, ):
        super(TransformerBlock).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        self.norm = nn.LayerNorm(normalized_shape=768)
        self.feed_forward = nn.Linear()  # TODO: add feed forward layer

    def forward(self, inputs):
        attention_output = self.attention(inputs, inputs, inputs)
        normalized_output = self.norm(attention_output + inputs)
        feed_forward_output = self.feed_forward(normalized_output)
        return feed_forward_output
