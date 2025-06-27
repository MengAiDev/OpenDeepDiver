import os
import json
import torch
import random
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
import re

# 1. 配置参数（显存优化版）
class DeepDiverConfig:
    def __init__(self):
        # 模型设置 - 使用4-bit量化
        self.model_name = "Qwen/Qwen2.5-3B"  # 更小模型节省显存
        self.use_4bit = True
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
        
        # PPO参数
        self.ppo_batch_size = 1  # PPO批量设为1
        self.ppo_epochs = 1
        self.ppo_steps = 500  # 减少PPO步数
        self.gamma = 0.99
        self.lam = 0.95
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        
        # 路径设置
        self.dataset_path = "your_dataset.jsonl"  # 3735条数据
        self.output_dir = "./deepdiver_output"
        self.sft_output_dir = os.path.join(self.output_dir, "sft_model")
        self.ppo_output_dir = os.path.join(self.output_dir, "ppo_model")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)
        os.makedirs(self.ppo_output_dir, exist_ok=True)

# 2. 数据处理
class WebPuzzleDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """加载并预处理3735条数据"""
        data = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                item = json.loads(line)
                
                # 构建训练样本
                prompt = (
                    "你是一个信息搜索专家，需要解决以下问题：\n"
                    f"问题：{item['question']}\n"
                    "请按步骤进行推理和搜索：\n"
                )
                
                # 添加难度信息
                if 'difficulty' in item:
                    prompt += f"问题难度：{item['difficulty']}\n"
                
                data.append({
                    "prompt": prompt,
                    "answer": item.get("answer", ""),
                    "difficulty": item.get("difficulty", "medium"),
                    "question": item.get("question", "")
                })
        
        print(f"Loaded {len(data)} samples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def format_for_sft(self, item):
        """为监督微调格式化数据"""
        text = item['prompt'] + "\n推理过程："
        return text
    
    def format_for_ppo(self, item):
        """为PPO训练格式化数据"""
        return item['prompt']

# 3. DuckDuckGo 搜索工具
class DuckDuckGoSearcher:
    def __init__(self, max_results=3, min_delay=1.0):
        self.max_results = max_results
        self.min_delay = min_delay
        self.last_search_time = 0
        
    def search(self, query):
        """执行DuckDuckGo搜索"""
        # 遵守速率限制
        current_time = time.time()
        if current_time - self.last_search_time < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_search_time))
        
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        "title": r.get('title', ''),
                        "url": r.get('href', ''),
                        "snippet": r.get('body', '')
                    })
                self.last_search_time = time.time()
                return results
        except Exception as e:
            print(f"搜索出错: {str(e)}")
            return [{"error": f"搜索服务不可用: {str(e)}"}]
    
    def format_results(self, results):
        """格式化搜索结果"""
        if not results:
            return "没有找到相关信息"
        
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            if "error" in result:
                return result["error"]
            
            # 提取关键信息
            snippet = result.get('snippet', '')
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."
                
            formatted.append(f"{i}. {result.get('title', '无标题')}\n   {snippet}")
        
        return "\n".join(formatted)

# 4. 冷启动监督微调（SFT）
def run_sft_training(config):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=config.bnb_config if config.use_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 准备数据集
    dataset = WebPuzzleDataset(config, tokenizer)
    sft_data = [dataset.format_for_sft(item) for item in dataset.data]
    
    # 对数据集进行分词
    def tokenize_function(examples):
        return tokenizer(
            examples,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length"
        )
    
    tokenized_data = tokenize_function(sft_data)
    
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
        report_to="none",
        optim="paged_adamw_8bit",  # 分页优化器减少内存峰值
        gradient_checkpointing=True  # 梯度检查点节省显存
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("Starting SFT training...")
    trainer.train()
    
    # 保存模型
    model.save_pretrained(config.sft_output_dir)
    tokenizer.save_pretrained(config.sft_output_dir)
    print(f"SFT model saved to {config.sft_output_dir}")

# 5. 强化学习训练（PPO）
def run_ppo_training(config):
    # 加载SFT模型
    tokenizer = AutoTokenizer.from_pretrained(config.sft_output_dir)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.sft_output_dir,
        quantization_config=config.bnb_config if config.use_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
    )
    
    # 准备数据集
    dataset = WebPuzzleDataset(config, tokenizer)
    ppo_data = [dataset.format_for_ppo(item) for item in dataset.data]
    
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
    
    # 生成设置
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,  # 减少生成长度节省显存
    }
    
    # 初始化搜索工具
    searcher = DuckDuckGoSearcher(max_results=2)  # 减少搜索结果节省资源
    
    # 奖励函数 - 根据论文设计（优化版）
    def reward_function(responses, prompts):
        rewards = []
        for response, prompt in zip(responses, prompts):
            # 1. 基本奖励：回答是否包含正确答案
            answer = next((item["answer"] for item in dataset.data if item["prompt"] == prompt), "")
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
            difficulty = next((item["difficulty"] for item in dataset.data if item["prompt"] == prompt), "medium")
            if difficulty == "hard":
                reward += 0.5
            elif difficulty == "outlier":
                reward += 1.0
                
            # 5. 效率惩罚：过长响应惩罚
            if len(response) > 500:
                reward -= 0.3
                
            rewards.append(reward)
            
        return torch.tensor(rewards, dtype=torch.float)
    
    # PPO训练循环（简化版）
    print("Starting PPO training...")
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=min(config.ppo_steps, len(ppo_trainer.dataloader))):
        if step >= config.ppo_steps:
            break
            
        # 获取查询
        queries = batch["query"]
        
        # 生成响应
        response_tensors = []
        responses = []
        for query in queries:
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            response = model.generate(**inputs, **generation_kwargs)
            response_tensor = response.squeeze()[-generation_kwargs["max_new_tokens"]:]
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            # 处理搜索请求（模拟）
            if "搜索：" in response_text:
                search_query = response_text.split("搜索：")[1].split("\n")[0].strip()
                # 在实际训练中，我们模拟搜索结果
                search_result = f"模拟搜索结果: {search_query}"
                response_text += f"\n搜索结果：{search_result}\n"
            
            response_tensors.append(response_tensor)
            responses.append(response_text)
        
        # 计算奖励
        rewards = reward_function(responses, queries)
        
        # PPO更新
        stats = ppo_trainer.step(response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        # 定期保存检查点
        if step % 100 == 0:
            model.save_pretrained(f"{config.ppo_output_dir}/step_{step}")
            print(f"Saved checkpoint at step {step}")
    
    # 保存最终模型
    model.save_pretrained(config.ppo_output_dir)
    tokenizer.save_pretrained(config.ppo_output_dir)
    print(f"PPO model saved to {config.ppo_output_dir}")

# 6. 搜索推理模块（集成DuckDuckGo）
class DeepDiverAgent:
    def __init__(self, config):
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(config.ppo_output_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.ppo_output_dir,
            quantization_config=config.bnb_config if config.use_4bit else None,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        # 初始化搜索工具
        self.searcher = DuckDuckGoSearcher(max_results=3)
        self.max_steps = 5
        self.max_search_queries = 5  # 限制搜索查询次数
        
    def generate(self, prompt):
        """生成推理过程（集成真实搜索）"""
        full_response = ""
        search_count = 0
        search_queries = []
        
        for step in range(self.max_steps):
            # 准备输入
            input_text = prompt + full_response
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            # 生成下一步
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True  # 提前停止节省资源
            )
            
            # 解码响应
            new_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            full_response += new_text
            
            # 检查是否需要搜索
            if "搜索：" in new_text and search_count < self.max_search_queries:
                search_count += 1
                # 提取搜索查询
                search_query = new_text.split("搜索：")[1].split("\n")[0].strip()
                search_queries.append(search_query)
                
                # 执行真实搜索
                search_results = self.searcher.search(search_query)
                formatted_results = self.searcher.format_results(search_results)
                
                # 添加到上下文
                full_response += f"\n搜索结果：{formatted_results}\n"
            
            # 检查是否给出最终答案
            if "答案：" in new_text:
                break
        
        return {
            "response": full_response,
            "search_count": search_count,
            "search_queries": search_queries,
            "steps": step + 1
        }

# 7. 评估模块
def evaluate_agent(config, test_data_path):
    """评估DeepDiver代理"""
    # 加载测试数据
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # 初始化代理
    agent = DeepDiverAgent(config)
    
    results = []
    for item in tqdm(test_data[:20], desc="Evaluating"):  # 只评估20个样本节省资源
        prompt = (
            "你是一个信息搜索专家，需要解决以下问题：\n"
            f"问题：{item['question']}\n"
            "请按步骤进行推理和搜索：\n"
        )
        
        response = agent.generate(prompt)
        
        # 简单评估
        correct = item["answer"] in response["response"]
        
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "response": response["response"],
            "correct": correct,
            "search_count": response["search_count"],
            "search_queries": response["search_queries"],
            "steps": response["steps"]
        })
    
    # 计算准确率
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_searches = sum(r["search_count"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    
    print(f"Evaluation Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Searches: {avg_searches:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # 保存结果
    with open(os.path.join(config.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# 8. 主流程
if __name__ == "__main__":
    # 初始化配置
    config = DeepDiverConfig()
    
    # 阶段1: 监督微调
    print("="*50)
    print("Starting Supervised Fine-Tuning (SFT)")
    print("="*50)
    run_sft_training(config)
    
    # 阶段2: 强化学习训练
    print("\n" + "="*50)
    print("Starting Proximal Policy Optimization (PPO)")
    print("="*50)
    run_ppo_training(config)
    
    # 阶段3: 评估
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)
    # evaluate_agent(config, "test_dataset.jsonl")
    
    print("\nOpen DeepDiver training completed!")
