# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/16 20:46
@Auth ： 吕鑫
@File ：config.py
@IDE ：PyCharm
"""


class LMConfig:
    def __init__(self):
        self.n_heads = 12
        self.dim = 768
        self.dropout = 0.1
        self.max_seq_len = 1024
        self.n_kv_heads = None
        self.flash_attn = False
