# -*- coding: utf-8 -*-
"""
@Time ： 2025/3/10 22:50
@Auth ： 吕鑫
@File ：train_full_sft.py
@IDE ：PyCharm
"""

from torch.nn.parallel import DistributedDataParallel
from torch import nn, optim
import time
from contextlib import nullcontext

import argparse
import os

import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoTokenizer
import torch
from models.model import LLMModel
from models.config import LMConfig
import math
from datasets.llm_dataset import SFTDataset


def Logger(content, ddp):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """Cosine Annealing 余弦退火, 让学习率平滑下降"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode(args):
    if not args.ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def init_model(lm_config, args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = LLMModel(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'/Users/lvxin/datasets/llm/pretrain_{lm_config.embedding_dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万', args.ddp)
    model = model.to(args.device)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="/Users/lvxin/datasets/llm/sft_mini_512.jsonl")
    parser.add_argument("--tokenizer", type=str, default="./models/tokenizer")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    args.wandb_run_name = f"MiniMind-sft_full_train-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    args.use_wandb = False

    if args.ddp:
        init_distributed_mode(args)
        args.device = torch.device(DEVICE)

    lm_config = LMConfig(embedding_dim=args.embedding_dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len,
                         use_moe=args.use_moe)
    model, tokenizer = init_model(lm_config, args)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 根据是否启用分布式训练，选择是否使用 DistributedSampler 来分配数据
    # 在分布式训练中，DistributedSampler 会将数据集划分成多个部分，每个训练进程只处理其中一部分。这样可以确保每个进程处理不同的数据，避免重复处理
    train_sampler = DistributedSampler(train_ds) if args.ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler  # 如果启用了分布式训练并且使用了 DistributedSampler，则数据将会被分配到不同的进程中
    )
    from pretrain import train
    train(model, train_loader, lm_config, args)


if __name__ == '__main__':
    main()
