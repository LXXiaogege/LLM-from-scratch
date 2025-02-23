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
        self.args = args
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim)
        self.embedded_dropout = nn.Dropout(args.dropout)
        # transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.embedding_dim, eps=args.norm_eps)
        self.output = nn.Linear(args.embedding_dim, args.vocab_size)
        # 将模型的多个输出封装成一个对象，以便后续处理。
        self.OUT = CausalLMOutputWithPast()

    def forward(self, input_ids=None, past_key_values=None, use_cache: bool = False, **args):
        """

        :param input_ids: (batch_size, seq_len)
        :param past_key_values:包含先前的计算结果，key 和 value（在自注意力机制中用于加速计算）。如果是第一次推理，past_key_values 为 None
        :param use_cache:
        :param args:
        :return:
        """

        # 用于标记每个 token 在序列中的位置。start_pos 确定了序列的起始位置，pos_cis 用来提取位置编码
        start_pos = args.get('start_pos', 0)
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]

        past_key_values = past_key_values or [None] * len(self.transformer_blocks)

        embedded = self.embedding(input_ids)
        # 小规模数据集：数据量不足时，embedding层参数量大（尤其是词表庞大的NLP任务），易过拟合。Dropout可有效抑制过拟合
        # 大规模预训练：数据充足时,embedding 后 Dropout可能损害语义完整性，主流大模型通常省略此处的Dropout。
        if self.args.use_embedding_dropout:
            embedded = self.embedded_dropout(embedded)

        past_kvs = []
        for idx, transformer_block in enumerate(self.transformer_blocks):
            embedded, past_kv = transformer_block(embedded, pos_cis=pos_cis, past_kv=past_key_values[idx],
                                                  use_cache=use_cache)
            past_kvs.append(past_kv)

        logits = self.output(self.norm(embedded))
        # MoE模型中用于控制专家激活的辅助损失,用于负载均衡
        aux_loss = sum(
            l.feed_forward.aux_loss for l in self.transformer_blocks if isinstance(l.feed_forward, MOElLayer))

        # 在模型的输出层，通常直接输出 logits，而不是通过 softmax 转换为概率。这是因为：
        # 1. softmax 通常在训练或推理时的后续步骤中应用，而不是模型内部。
        # 2. 使用 logits 更加灵活，尤其是对于生成任务（如语言模型），可以根据 logits 做不同的解码操作。
        # 3. 在训练时，CrossEntropyLoss 会自动处理 softmax，所以模型不需要显式地应用它。
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT


class TransformerBlock(nn.Module):
    def __init__(self, args: LMConfig):
        super(TransformerBlock).__init__()
        self.attn_norm = RMSNorm(dim=args.embedding_dim, eps=args.norm_eps)
        self.attention = Attention(args=self.args)
        self.ffn_norm = RMSNorm(dim=args.embedding_dim, eps=args.norm_eps)
        self.feed_forward = FeedForwardLayer(args) if not args.use_moe else MOElLayer(args)

    def forward(self, inputs, pos_cis, past_key_value=None, use_cache=False):
        """

        param inputs: 输入张量， `[batch_size, seq_len, hidden_dim]`
        param pos_cis: 位置编码
        param past_key_value: 前 t−1 token的 k v cache，用于加速自回归生成
        param use_cache: 是否使用缓存 k_v cache
        return: 输出张量，形状为 `[batch_size, seq_len, hidden_dim]`
        """
        att_inputs = self.attn_norm(inputs)
        att_outputs, past_kv = self.attention(att_inputs, pos_cis, past_key_value, use_cache)
        # residual connection
        att_outputs = inputs + att_outputs

        fead_forward_inputs = self.ffn_norm(att_outputs)
        feed_forward_output = self.feed_forward(fead_forward_inputs)
        # residual connection
        feed_forward_output = att_outputs + feed_forward_output
        return feed_forward_output, past_kv


class FeedForwardLayer(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.ffn_hidden_dim = args.ffn_hidden_dim
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 4 * args.embedding_dim
            self.ffn_hidden_dim = int(2 * self.ffn_hidden_dim / 3)
            self.ffn_hidden_dim = args.multiple_of * ((self.ffn_hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.embedding_dim, self.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(args.ffn_hidden_dim, args.embedding_dim, bias=False)
        self.w3 = nn.Linear(args.embedding_dim, args.ffn_hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # SiLU 函数的优点在于它能够平滑地处理负值，并且在正值输入时提供较大的输出范围.它有助于保持梯度的流动，避免梯度消失问题。
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


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
        self.gating_dim = args.embedding_dim
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
        self.router = Router(args)
        #  路由专家
        self.router_experts = nn.ModuleList([FeedForwardLayer(args) for _ in range(args.n_routed_experts)])
        # 共享专家
        if args.n_shared_experts is not None:
            self.shared_experts = nn.ModuleList([FeedForwardLayer(args) for _ in range(args.n_shared_experts)])
        self.aux_loss = None

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        在推理模式，模型只选择最优的专家进行计算。根据输入的专家索引和权重，选择合适的专家进行推理计算
        """
        # 用于存储最终的加权专家输出，初始化为与 x 相同形状的零张量
        expert_cache = torch.zeros_like(x)
        # 将 flat_expert_indices 按照升序排列，并返回排序后的索引
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的 token 数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个 token 所属的专家
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        # 迭代每个专家，进行推理计算
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:  # 跳过没有处理 token 的专家
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权输出
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加专家输出到 expert_cache, 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.router(x)  # topk_idx， topk_weight(bsz * seq_len, top_k)
        x = x.view(-1, x.shape[-1])  # (bsz * seq_len, h)
        flat_topk_idx = topk_idx.view(-1)  # (bsz * seq_len * top_k)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # （bsz * seq_len * top_k, h)
            y = torch.empty_like(x, dtype=torch.float16)  # (bsz * seq_len * top_k, h)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致,(bsz * seq_len * top_k, h)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 加权和所有专家的输出 (bsz * seq_len, h)
            y = y.view(*orig_shape)  # (bsz, seq_len, h)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y
