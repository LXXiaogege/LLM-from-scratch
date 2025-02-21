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
import torch
import math
import torch.nn.init as init


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


class Router(nn.Module):
    """
    路由/门控层(Router/Gate)，决定将输入的token送给那个专家
    通常会选择top K的专家激活
    """

    def __init__(self, args: LMConfig):
        super().__init__()
        self.args = args
        # 选择评分最高的前 K 个专家。
        self.top_k = args.num_experts_per_tok
        # 路由专家数量
        self.n_routed_experts = args.n_routed_experts
        # 评分函数
        self.scoring_func = args.scoring_func
        # 用于辅助损失的系数。
        self.alpha = args.aux_loss_alpha
        # 是否使用序列级的辅助损失
        self.seq_aux = args.seq_aux

        # 是否对 top K 的权重进行归一化
        self.norm_topk_prob = args.norm_topk_prob
        #
        self.gating_dim = args.dim
        # 可学习的参数，用于生成每个专家的权重
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 初始化专家权重
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用 kaiming_uniform_ 方法初始化 weight 权重，通常是用来初始化神经网络权重的一种策略，适合 ReLU 激活函数。"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # (bsz * seq_len, h)
        # 使用一个全连接层（F.linear）计算每个 token 对每个专家的“评分”
        logits = F.linear(hidden_states, self.weight, None)  # (bsz * seq_len, n_routed_experts)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择出每个 token 评分最高的 top_k 个专家，并返回它们的权重和索引
        # topk_weight前 K 个专家的权重，形状是 (bsz * seq_len, top_k)， topk_idx 前 K 个专家的索引，形状是 (bsz * seq_len, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        #  Top-K 权重归一化， 使得每一行的权重和为 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 负载均衡策略： 使用辅助损失，让选择专家时更加均衡，不让模型总是选择部分专家
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


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
