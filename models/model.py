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
from transformers.modeling_outputs import CausalLMOutputWithPast


class LLMModel(nn.Module):
    def __init__(self, args: LMConfig):
        super(LLMModel).__init__()
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim)
        # transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.norm_dim)
        self.output = nn.Linear(args.embedding_dim, args.vocab_size)
        # 将模型的多个输出封装成一个对象，以便后续处理。
        self.OUT = CausalLMOutputWithPast()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            embedded = transformer_block(embedded)
        logits = self.output(self.norm(embedded))

        # 在模型的输出层，通常直接输出 logits，而不是通过 softmax 转换为概率。这是因为：
        # 1. softmax 通常在训练或推理时的后续步骤中应用，而不是模型内部。
        # 2. 使用 logits 更加灵活，尤其是对于生成任务（如语言模型），可以根据 logits 做不同的解码操作。
        # 3. 在训练时，CrossEntropyLoss 会自动处理 softmax，所以模型不需要显式地应用它。
        self.OUT.__setitem__('logits', logits)
        # self.OUT.__setitem__('aux_loss', aux_loss)
        # self.OUT.__setitem__('past_key_values', past_kvs)        __
        return self.OUT


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
    def __init__(self, args: LMConfig):
        super().__init__()
        self.ffn_hidden_dim = args.ffn_hidden_dim
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 4 * args.ffn_dim
            self.ffn_hidden_dim = int(2 * self.ffn_hidden_dim / 3)
            self.ffn_hidden_dim = args.multiple_of * ((self.ffn_hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.ffn_dim, self.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(args.ffn_hidden_dim, args.ffn_dim, bias=False)
        self.w3 = nn.Linear(args.ffn_dim, args.ffn_hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.ffn_dropout)

    def forward(self, x):
        # SiLU 函数的优点在于它能够平滑地处理负值，并且在正值输入时提供较大的输出范围.它有助于保持梯度的流动，避免梯度消失问题。
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RouterExpert(nn.Module):
    """
    路由专家
    """

    def __init__(self, args: LMConfig):
        super(RouterExpert).__init__()

    def forward(self, inputs):
        pass


class RouterModel(nn.Module):
    """
    路由层，决定将输入的token送给那个专家
    通常会选择top K的专家激活
    """

    def __init__(self, args: LMConfig):
        super(RouterModel).__init__()
        self.args = args

    def forward(self, inputs):
        pass


class MOElLayer(nn.Module):
    def __init__(self, args: LMConfig):
        super(MOElLayer).__init__()
        self.args = args
        #  路由专家
        self.router_experts = nn.ModuleList([FeedForwardLayer(args) for _ in range(args.n_router_experts)])
        # 共享专家
        self.shared_experts = nn.ModuleList([FeedForwardLayer(args) for _ in range(args.n_shared_experts)])

    def forward(self, inputs):
        pass
