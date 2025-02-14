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
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        if norm_type == 'rms':
            self.norm = RMSNorm(dim=768)
        else:
            self.norm = nn.LayerNorm(normalized_shape=768)
        self.feed_forward = FeedForwardLayer()

    def forward(self, inputs):
        attention_output = self.attention(inputs, inputs, inputs)
        normalized_output = self.norm(attention_output + inputs)
        feed_forward_output = self.feed_forward(normalized_output)
        return feed_forward_output


class FeedForwardLayer(nn.Module):
    def __init__(self, ):
        super(FeedForwardLayer).__init__()
        self.linear1 = nn.Linear(in_features=768, out_features=3072)
        self.linear2 = nn.Linear(in_features=3072, out_features=768)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.linear1(inputs)
        output = self.relu(output)
        output = self.linear2(output)
        return output
