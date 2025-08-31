# deepdiver_SFT.py
##################################################

import os
import json
import torch
import random
import time
import re
import html
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    modeling_utils,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import copy

#################### Config ####################
class DeepDiverConfig:
    def __init__(self):
        # 模型设置 - 使用4-bit量化
        self.model_name = "Qwen/Qwen2.5-3B" 
        self.use_4bit = False
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # LoRA配置
        self.lora_rank = 32  # 降低rank节省显存
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        # 训练参数
        self.batch_size = 2  # 减小批量大小
        self.gradient_accumulation_steps = 8  # 梯度累积
        self.learning_rate = 1e-5
        self.num_train_epochs = 2  # 减少训练轮次
        self.max_seq_length = 1024  # 减少序列长度节省显存
        # 路径设置
        self.dataset_path = "webpuzzle_dataset_3735.jsonl"  # 3735条数据
        self.output_dir = "./deepdiver_output"
        self.sft_output_dir = os.path.join(self.output_dir, "sft_model")
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)

# 新增: 将TokenizedDataset类提升到模块级别
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        # Ensure labels are correctly assigned
        self.encodings["labels"] = self.encodings["input_ids"].clone()
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            # 确保返回张量而不是切片操作
            tensor = val[idx].clone().detach()
            item[key] = tensor
        return item

# WebPuzzleDataset类（简化版，仅包含SFT所需功能）
class WebPuzzleDataset(Dataset):
    def __init__(self, config, tokenizer, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = self.load_data()
        
    def load_data(self):
        data = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        
        # 划分训练集和验证集 (80% train, 20% val)
        random.seed(42)
        random.shuffle(data)
        split_idx = int(0.8 * len(data))
        
        if self.mode == 'train':
            return data[:split_idx]
        else:
            return data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def format_for_sft(self, item):
        """格式化数据用于SFT训练"""
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")
        return f"{prompt}\n答案: {answer}\n"

#################### Cold Start SFT ####################
def run_sft_training(config):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_kwargs = {}
    if config.use_4bit:
        model_kwargs["quantization_config"] = config.bnb_config
    model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config explicitly
    model_config = AutoConfig.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        **model_kwargs
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    # 修复1: 确保模型准备正确
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 修复2: 禁用缓存并启用输入梯度要求
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    
    # 准备数据集（只使用训练集）
    train_dataset = WebPuzzleDataset(config, tokenizer, mode='train')
    sft_data = [train_dataset.format_for_sft(item) for item in train_dataset.data]
    
    # 对数据集进行分词
    tokenized_data = tokenizer(
        sft_data,
        max_length=config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 使用定义在模块级别的TokenizedDataset类
    train_dataset = TokenizedDataset(tokenized_data)
    
    # 准备验证集（用于评估）
    val_dataset = WebPuzzleDataset(config, tokenizer, mode='val')
    val_data = [val_dataset.format_for_sft(item) for item in val_dataset.data]
    tokenized_val_data = tokenizer(
        val_data,
        max_length=config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Fix: Convert validation data to a PyTorch Dataset
    val_dataset = TokenizedDataset(tokenized_val_data)
    
    # 创建数据加载器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 配置训练参数（优化显存使用）
    training_args = TrainingArguments(
        output_dir=config.sft_output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",  # 分页优化器减少内存峰值
        gradient_checkpointing=True,  # 梯度检查点节省显存
        remove_unused_columns=False,  # 防止移除梯度计算所需的列
        dataloader_num_workers=4,    # 添加数据加载器工作线程
    )
    
    # 修复4: 确保模型设置为训练模式
    model.train()
    
    # Fix: Ensure labels are passed correctly during training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save the model
    print("Saving model and configuration...")
    # 修复：保存完整模型而不仅仅是适配器
    model.save_pretrained(config.sft_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(config.sft_output_dir)
    model.config.to_json_file(os.path.join(config.sft_output_dir, "config.json"))
    print(f"SFT model saved to {config.sft_output_dir}")

def main():
    # Init config
    config = DeepDiverConfig()
    
    print("=" * 50)
    print("Starting Supervised Fine-Tuning (SFT)")
    print("=" * 50)
    run_sft_training(config)
    
    print("\nOpen DeepDiver SFT training completed!")

if __name__ == "__main__":
    main()
