# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/16 20:46
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""


class LMConfig:
    def __init__(self, vaocab_size=6400,
                 max_seq_len=8192,
                 embedding_dim=512,
                 dropout=0.1,
                 use_embedding_dropout=True,
                 n_layers=8,
                 norm_eps=1e-5,
                 n_heads=8,
                 n_kv_heads=2,
                 flash_attn=False,
                 multiple_of=64,
                 ffn_hidden_dim=None,
                 use_moe=True,
                 n_routed_experts=100,
                 n_shared_experts=1,
                 num_experts_per_tok=8,
                 scoring_func="softmax",
                 aux_loss_alpha=0.01,
                 seq_aux=False,
                 norm_topk_prob=False):
        self.vocab_size = vaocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.use_embedding_dropout = use_embedding_dropout

        # block
        self.n_layers = n_layers

        # rms norm
        self.norm_eps = norm_eps

        # attention layer
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.flash_attn = flash_attn

        # feed forward layer
        self.multiple_of = multiple_of
        self.ffn_hidden_dim = ffn_hidden_dim

        # moe
        self.use_moe = use_moe
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
