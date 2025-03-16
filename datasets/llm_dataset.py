# -*- coding: utf-8 -*-
"""
@Time ： 2025/2/27 23:24
@Auth ： 吕鑫
@File ：llm_dataset.py
@IDE ：PyCharm
"""

from torch.utils.data import Dataset
import json
import torch


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        # 生成一个损失掩码，标记哪些 token 是有效的，哪些是填充（pad） token， [True, True, True, True, False, False, False]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.as_tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.as_tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.as_tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _generate_loss_mask(self, input_ids):
        # 生成 loss_mask，让 LLM 训练时 忽略 prompt，只计算回答部分的 Loss
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 超长的截断
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))  # 前面截断了，不会担心出现负值
        loss_mask = self._generate_loss_mask(input_ids)

        # X = torch.as_tensor(input_ids[:-1], dtype=torch.long)
        # Y = torch.as_tensor(input_ids[1:], dtype=torch.long)
        # loss_mask = torch.as_tensor(loss_mask[1:], dtype=torch.long)
        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        return X, Y, loss_mask


if __name__ == '__main__':
    from pretrain import init_model, parse_args
    from models.config import LMConfig

    args = parse_args()
    args.tokenizer_path = "../models/tokenizer"
    llm_config = LMConfig()
    model, tokenizer = init_model(llm_config, args)
    datasets = SFTDataset(data_path="/Users/lvxin/datasets/llm/sft_test.jsonl", tokenizer=tokenizer)
    # samples = datasets.load_data("/Users/lvxin/datasets/llm/sft_test.jsonl")
    # datasets._create_chat_prompt(samples[0]['conversations'])
    print(datasets.__getitem__(0))
