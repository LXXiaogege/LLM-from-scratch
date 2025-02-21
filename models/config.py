# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/16 20:46
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""


class LMConfig:
    def __init__(self):
        # attention layer
        self.n_heads = 12
        self.dim = 768
        self.dropout = 0.1
        self.max_seq_len = 1024
        self.n_kv_heads = None
        self.flash_attn = False
        self.norm_dim = 768

        self.vocab_size = 50257
        self.embedding_dim = 768
        self.n_layers = 12

        # feed forward layer
        self.multiple_of = 256
        self.ffn_hidden_dim = None
        self.ffn_dim = 256
        self.ffn_dropout = 0.1

        # moe
        self.n_routed_experts = 100
        self.n_shared_experts = 1
        self.num_experts_per_tok = 8
        self.scoring_func = "softmax"
        self.aux_loss_alpha = 0.01
        self.seq_aux = False
        self.norm_topk_prob = False