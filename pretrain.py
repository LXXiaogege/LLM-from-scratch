# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/26 19:22
@Auth ： 吕鑫
@File ：pretrain.py
@IDE ：PyCharm

模型预训练
"""
import argparse
import torch
from transformers import AutoTokenizer
from models.model import LLMModel
from models.config import LMConfig
import torch.distributed as dist
import os


def Logger(content, ddp):
    """
    DDP分布式训练时，日志打印，只在主进程打印日志。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./models/tokenizer')
    model = LLMModel(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def train(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    # 是否使用分布式数据并行 (DDP)
    parser.add_argument("--ddp", action="store_true")
    # 梯度累积步数。在内存有限的情况下，可以通过梯度累积来模拟较大的批量训练。
    parser.add_argument("--accumulation_steps", type=int, default=8)
    # 梯度裁剪的阈值，默认是 1.0。在训练过程中，如果梯度的 L2 范数超过这个阈值，会进行裁剪，从而防止梯度爆炸。
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 学习率预热的迭代次数，默认是 0。预热是指在训练初期，学习率逐渐从小值增长到设定的值。
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    # 分布式训练中的进程编号，用于分布式训练时同步模型
    parser.add_argument('--local_rank', type=int, default=-1)

    # 模型设置
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    lm_config = LMConfig(embedding_dim=args.embedding_dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len,
                         use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 每个训练迭代（iteration）中，模型将处理的 token 总数
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    train(args)
