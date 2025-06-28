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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import numpy as np

# New imports for modularized components
from dataset_module import WebPuzzleDataset
from search_module import DuckDuckGoSearcher
from reward_module import calculate_reward
from agent_module import DeepDiverAgent

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
        
        # PPO参数
        self.ppo_batch_size = 1  # PPO批量设为1
        self.ppo_epochs = 1
        self.ppo_steps = 500  # 减少PPO步数
        self.gamma = 0.99
        self.lam = 0.95
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        
        # 路径设置
        self.dataset_path = "webpuzzle_dataset_3735.jsonl"  # 3735条数据
        self.output_dir = "./deepdiver_output"
        self.sft_output_dir = os.path.join(self.output_dir, "sft_model")
        self.ppo_output_dir = os.path.join(self.output_dir, "ppo_model")
        self.val_dataset_path = os.path.join(self.output_dir, "val_dataset.jsonl")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)
        os.makedirs(self.ppo_output_dir, exist_ok=True)

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Add gate/up/down for MLP layers
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
    
    # 创建数据加载器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
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
        report_to="wandb",
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
    model.save_pretrained(config.sft_output_dir)
    tokenizer.save_pretrained(config.sft_output_dir)
    print(f"SFT model saved to {config.sft_output_dir}")

# 5. 强化学习训练（PPO）
def run_ppo_training(config):
    # 加载SFT模型
    tokenizer = AutoTokenizer.from_pretrained(config.sft_output_dir)
    
    # Load config explicitly
    model_config = AutoConfig.from_pretrained(config.sft_output_dir)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.sft_output_dir,
        config=model_config,
        quantization_config=config.bnb_config if config.use_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
    )
    
    # 准备数据集（只使用训练集）
    train_dataset = WebPuzzleDataset(config, tokenizer, mode='train')
    
    # 修改1: PPO数据应包含查询、响应、奖励等信息
    ppo_data = []
    for item in train_dataset.data:
        formatted = train_dataset.format_for_ppo(item)
        ppo_data.append({
            "query": formatted,
            "prompt": item["prompt"],
            "answer": item["answer"],
            "difficulty": item["difficulty"]
        })
    
    # 创建PPO配置
    ppo_config = PPOConfig(
        batch_size=config.ppo_batch_size,
        mini_batch_size=1,
        ppo_epochs=config.ppo_epochs,
        learning_rate=config.learning_rate,
        log_with=None,
        steps=config.ppo_steps,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        optimize_cuda_cache=True,
    )
    
    # 创建PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=ppo_data
    )
    
    # 生成设置 - 更严格的参数控制
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,  # 减少生成长度节省显存
        "output_scores": True,  # 需要输出分数用于奖励计算
        "return_dict_in_generate": True  # 返回字典格式
    }
    
    # 初始化搜索工具
    searcher = DuckDuckGoSearcher(max_results=2)  # 减少搜索结果节省资源
    
    # 奖励函数 - 根据论文设计（优化版）
    def reward_function(responses, prompts, answers, difficulties):
        rewards = []
        for response, prompt in zip(responses, prompts):
            # 1. 基本奖励：回答是否包含正确答案
            if answer and answer in response:
                reward = 1.0
            else:
                reward = 0.0
                
            # 2. 搜索强度奖励：搜索操作的数量
            search_count = response.count("搜索：")
            if search_count > 0:
                reward += min(search_count * 0.2, 1.0)
                
            # 3. 反思奖励：反思操作的数量
            reflection_count = response.count("反思：")
            reward += min(reflection_count * 0.3, 1.5)
            
            # 4. 难度奖励：困难问题额外奖励
            if difficulty == "hard":
                reward += 0.5
            elif difficulty == "outlier":
                reward += 1.0
                
            # 5. 效率惩罚：过长响应惩罚
            if len(response) > 500:
                reward -= 0.3
                
            rewards.append(reward)
            
        return torch.tensor(rewards, dtype=torch.float)
    
    # PPO训练循环（完整实现）
    print("Starting PPO training...")
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=min(config.ppo_steps, len(ppo_trainer.dataloader))):
        if step >= config.ppo_steps:
            break
            
        # 获取查询
        queries = batch["query"]
        prompts = [item["prompt"] for item in ppo_data if item["query"] in queries]
        answers = [item["answer"] for item in ppo_data if item["query"] in queries]
        difficulties = [item["difficulty"] for item in ppo_data if item["query"] in queries]
        
        # 确保数据在正确的设备上
        query_tensors = [torch.tensor(tokenizer.encode(q, add_special_tokens=False)).to(model.device) for q in queries]
        
        # 生成响应 - 使用更安全的方式
        response_tensors = []
        responses = []
        for query_tensor in query_tensors:
            response = model.generate(
                input_ids=query_tensor.unsqueeze(0),
                **generation_kwargs
            )
            # 提取生成的内容
            response_tensor = response.sequences[0][query_tensor.shape[0]:]
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            # 处理搜索请求
            if "搜索：" in response_text:
                # 提取搜索查询
                search_query = response_text.split("搜索：")[1].split("\n")[0].strip()
                
                # 执行实际搜索
                try:
                    search_results = searcher.search(search_query)
                    formatted_results = searcher.format_results(search_results)
                    # 添加到响应中
                    response_text += f"\n搜索结果：{formatted_results}\n"
                except Exception as e:
                    print(f"Search error: {e}")
            
            response_tensors.append(response_tensor)
            responses.append(response_text)
        
        # 计算奖励
        rewards = reward_function(responses, prompts, answers, difficulties)
        
        # 确保张量在正确的设备上
        rewards = rewards.to(model.device)
        
        # PPO更新 - 更健壮的错误处理
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
        except Exception as e:
            print(f"PPO step error: {e}")
            continue
        
        # 定期保存检查点
        if step % 100 == 0:
            model.save_pretrained(f"{config.ppo_output_dir}/step_{step}")
            print(f"Saved checkpoint at step {step}")
            
            # 在验证集上进行评估
            evaluate_ppo_model(config, model, tokenizer, step)
    
    # 保存最终模型
    model.save_pretrained(config.ppo_output_dir)
    tokenizer.save_pretrained(config.ppo_output_dir)
    print(f"PPO model saved to {config.ppo_output_dir}")
    
    # 最终评估
    evaluate_ppo_model(config, model, tokenizer, "final")

def evaluate_ppo_model(config, model, tokenizer, step):
    """在验证集上评估PPO模型"""
    print(f"Evaluating PPO model at step {step}...")
    
    # 准备验证集
    val_dataset = WebPuzzleDataset(config, tokenizer, mode='val')
    
    # 初始化搜索工具
    searcher = DuckDuckGoSearcher(max_results=2)
    
    # 评估指标
    total_reward = 0.0
    correct_count = 0
    total_search_count = 0
    
    # 评估函数
    def reward_function(response, prompt, answer, difficulty):
        # 1. 基本奖励：回答是否包含正确答案
        if answer and answer in response:
            reward = 1.0
        else:
            reward = 0.0
            
        # 2. 搜索强度奖励：搜索操作的数量
        search_count = response.count("搜索：")
        if search_count > 0:
            reward += min(search_count * 0.2, 1.0)
            
        # 3. 反思奖励：反思操作的数量
        reflection_count = response.count("反思：")
        reward += min(reflection_count * 0.3, 1.5)
        
        # 4. 难度奖励：困难问题额外奖励
        if difficulty == "hard":
            reward += 0.5
        elif difficulty == "outlier":
            reward += 1.0
            
        # 5. 效率惩罚：过长响应惩罚
        if len(response) > 500:
            reward -= 0.3
            
        return reward, search_count
    
    # 评估验证集中的样本
    for item in tqdm(val_dataset.data[:50], desc="Evaluating"):  # 只评估50个样本
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(model.device)
        response = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = tokenizer.decode(response[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 处理搜索请求
        if "搜索：" in response_text:
            # 提取搜索查询
            search_query = response_text.split("搜索：")[1].split("\n")[0].strip()
            
            # 执行实际搜索
            search_results = searcher.search(search_query)
            formatted_results = searcher.format_results(search_results)
            
            # 添加到响应中
            response_text += f"\n搜索结果：{formatted_results}\n"
        
        # 计算奖励
        reward, search_count = reward_function(
            response_text,
            item['prompt'],
            item['answer'],
            item['difficulty']
        )
        
        total_reward += reward
        total_search_count += search_count
        
        # 检查是否正确
        if item['answer'] in response_text:
            correct_count += 1
    
    # 计算平均指标
    avg_reward = total_reward / len(val_dataset.data[:50])
    accuracy = correct_count / len(val_dataset.data[:50])
    avg_search = total_search_count / len(val_dataset.data[:50])
    
    print(f"Step {step} Evaluation Results:")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Searches: {avg_search:.2f}")

# 7. 最终评估模块
def evaluate_agent(config, test_data_path=None):
    """评估DeepDiver代理"""
    # 加载验证集
    val_dataset = WebPuzzleDataset(config, None, mode='val')
    
    # 初始化代理
    agent = DeepDiverAgent(config)
    
    results = []
    for item in tqdm(val_dataset.data, desc="Evaluating"):
        response = agent.generate(item["prompt"])
        
        # 检查是否正确
        correct = item["answer"] in response["response"]
        
        results.append({
            "id": item.get("id", "N/A"),
            "question": item["question"],
            "answer": item["answer"],
            "response": response["response"],
            "correct": correct,
            "search_count": response["search_count"],
            "search_queries": response["search_queries"],
            "steps": response["steps"]
        })
    
    # 计算指标
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_searches = sum(r["search_count"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    
    print(f"Final Evaluation Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Searches: {avg_searches:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # 保存结果
    with open(os.path.join(config.output_dir, "final_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

#################### CLI ####################
import argparse
def main():
    parser = argparse.ArgumentParser(description='DeepDiver Training CLI')
    parser.add_argument('--stage', choices=['sft', 'ppo', 'eval'], required=True,
                        help='Choose the stage to run: sft (Supervised Fine-Tuning), '
                             'ppo (Proximal Policy Optimization), or eval (Final Evaluation)')
    args = parser.parse_args()

    # Init config
    config = DeepDiverConfig()

    if args.stage == 'sft':
        print("=" * 50)
        print("Starting Supervised Fine-Tuning (SFT)")
        print("=" * 50)
        run_sft_training(config)

    elif args.stage == 'ppo':
        print("\n" + "=" * 50)
        print("Starting Proximal Policy Optimization (PPO)")
        print("=" * 50)
        run_ppo_training(config)

    elif args.stage == 'eval':
        print("\n" + "=" * 50)
        print("Starting Final Evaluation")
        print("=" * 50)
        evaluate_agent(config)

    print("\nOpen DeepDiver training completed!")

if __name__ == "__main__":
    main()
