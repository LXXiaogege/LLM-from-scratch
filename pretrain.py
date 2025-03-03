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
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from torch import nn, optim
from datasets.llm_dataset import PretrainDataset
from models.model import LLMModel
from models.config import LMConfig
import torch.distributed as dist
import os
import time
import math
from contextlib import nullcontext
import wandb


def Logger(content, ddp):
    """
    DDP分布式训练时，日志打印，只在主进程打印日志。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def init_model(lm_config, args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = LLMModel(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train(model, train_loader, lm_config, args):
    ddp = int(os.environ.get("RANK", -1)) != -1  # 读取环境变量 RANK，用来判断是否是分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    # 只有主进程（在DDP模式下）负责记录日志
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 每个训练迭代（iteration）中，模型将处理的 token 总数
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 设定随机种子，保证实验结果可复现
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 启用自动混合精度（AMP），提高计算速度并减少显存占用。
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 启用混合精度梯度缩放（GradScaler），，防止梯度下溢。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    iter_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(args.device)
            Y = Y.to(args.device)
            loss_mask = loss_mask.to(args.device)
            lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                # padding token 的损失对训练无意义，因此用 loss_mask 过滤掉无效部分。
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                # accumulation_steps 梯度累积步数。不除以 accumulation_steps，损失会比正常大 batch 训练的损失高 4 倍，导致梯度也放大 4 倍，训练不稳定。
                # 模拟更大 batch size（在显存受限时）。
                # 例如：
                # 设 batch_size=16，但想要 batch_size=64 的效果。
                # 通过 accumulation_steps=4，每 4 步累积一次梯度，再执行 optimizer.step()
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()  # 缩放 loss，防止 float16 计算导致数值不稳定（underflow）

            # 在梯度累积达到 accumulation_steps 次后，执行参数更新
            if (step + 1) % args.accumulation_steps == 0:
                # 反缩放梯度，确保梯度的数值在合适的范围，防止数值溢出
                scaler.unscale_(optimizer)
                # 梯度裁剪（Gradient Clipping），防止梯度爆炸（gradient explosion）
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # 执行参数更新，类似于 optimizer.step()，但 结合了 AMP 训练，确保数值稳定
                scaler.step(optimizer)
                # 调整 scaler 的缩放因子（loss scale），适应训练过程中的数值范围。
                # 如果 scaler.step() 发现梯度太小，scaler.update() 可能会增加 loss scale，防止数值 underflow。
                scaler.update()
                # 清空梯度，防止梯度累积到下一轮
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                Logger(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        args.epochs,
                        step,
                        iter_per_epoch,
                        loss.item() * args.accumulation_steps,
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
            if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
                model.eval()
                moe_path = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                torch.save(state_dict, ckp)
                model.train()


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
    parser.add_argument("--data_path", type=str, default="/Users/lvxin/datasets/llm/pretrain_hq.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./models/tokenizer")
    return parser.parse_args()


def main():
    args = parse_args()

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    lm_config = LMConfig(embedding_dim=args.embedding_dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len,
                         use_moe=args.use_moe)
    model, tokenizer = init_model(lm_config, args)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
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

    train(model, train_loader, lm_config, args)


if __name__ == '__main__':
    main()
