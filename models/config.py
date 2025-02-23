# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/16 20:46
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""


class LMConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.max_seq_len = 8192
        self.embedding_dim = 512
        self.dropout = 0.1

        # block
        self.n_layers = 8

        # rms norm
        self.norm_eps = 1e-5

        # attention layer
        self.n_heads = 8
        self.n_kv_heads = 2
        self.flash_attn = False

        # feed forward layer
        self.multiple_of = 64
        self.ffn_hidden_dim = None

        # moe
        self.n_routed_experts = 100
        self.n_shared_experts = 1
        self.num_experts_per_tok = 8
        self.scoring_func = "softmax"
        self.aux_loss_alpha = 0.01
        self.seq_aux = False
        self.norm_topk_prob = False
