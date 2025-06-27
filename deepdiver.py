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
        self.dataset_path = "webpuzzle_dataset_3735.jsonl.jsonl"  # 3735条数据
        self.output_dir = "./deepdiver_output"
        self.sft_output_dir = os.path.join(self.output_dir, "sft_model")
        self.ppo_output_dir = os.path.join(self.output_dir, "ppo_model")
        self.val_dataset_path = os.path.join(self.output_dir, "val_dataset.jsonl")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)
        os.makedirs(self.ppo_output_dir, exist_ok=True)

# 2. 数据处理
class WebPuzzleDataset(Dataset):
    def __init__(self, config, tokenizer, mode='train'):
        """
        mode: 'train' 或 'val'
        """
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """加载并预处理数据，按80%训练集和20%验证集划分"""
        # 首先检查验证集是否已经存在
        if self.mode == 'val' and os.path.exists(self.config.val_dataset_path):
            return self.load_val_dataset()
            
        # 加载完整数据集
        full_data = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading full dataset"):
                item = json.loads(line)
                full_data.append(item)
        
        print(f"Loaded {len(full_data)} samples in total")
        
        # 划分训练集和验证集 (80% 训练, 20% 验证)
        total_size = len(full_data)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # 随机打乱数据
        rng = np.random.default_rng(42)  # 固定随机种子确保可复现
        indices = rng.permutation(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 保存验证集到单独文件
        val_data = [full_data[i] for i in val_indices]
        with open(self.config.val_dataset_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved validation dataset with {len(val_data)} samples")
        
        # 根据模式返回相应数据
        if self.mode == 'train':
            data = [full_data[i] for i in train_indices]
            print(f"Using {len(data)} samples for training")
        else:
            data = val_data
            print(f"Using {len(data)} samples for validation")
        
        # 格式化数据
        formatted_data = []
        for item in data:
            # 构建训练样本
            prompt = (
                "你是一个信息搜索专家，需要解决以下问题：\n"
                f"问题：{item['question']}\n"
                "请按步骤进行推理和搜索：\n"
            )
            
            # 添加难度信息
            if 'difficulty' in item:
                prompt += f"问题难度：{item['difficulty']}\n"
            
            formatted_data.append({
                "prompt": prompt,
                "answer": item.get("answer", ""),
                "difficulty": item.get("difficulty", "medium"),
                "question": item.get("question", "")
            })
        
        return formatted_data
    
    def load_val_dataset(self):
        """从文件加载验证集"""
        data = []
        with open(self.config.val_dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading validation dataset"):
                item = json.loads(line)
                
                # 构建验证样本
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
        
        print(f"Loaded {len(data)} validation samples")
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

# 3. DuckDuckGo 搜索工具（完整实现）
class DuckDuckGoSearcher:
    def __init__(self, max_results=3, min_delay=1.0, max_retries=3):
        self.max_results = max_results
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.last_search_time = 0
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def search(self, query):
        """执行DuckDuckGo搜索"""
        # 遵守速率限制
        current_time = time.time()
        if current_time - self.last_search_time < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_search_time))
        
        retries = 0
        while retries < self.max_retries:
            try:
                with DDGS() as ddgs:
                    results = []
                    for r in ddgs.text(query, max_results=self.max_results):
                        # 获取更完整的页面内容
                        full_content = self.get_page_content(r.get('href', ''))
                        if full_content:
                            # 从页面内容中提取更相关的摘要
                            summary = self.extract_relevant_content(full_content, query)
                            if summary:
                                r['snippet'] = summary
                        
                        results.append({
                            "title": r.get('title', ''),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', r.get('snippet', ''))
                        })
                    
                    self.last_search_time = time.time()
                    return results
            except Exception as e:
                print(f"搜索出错 (尝试 {retries+1}/{self.max_retries}): {str(e)}")
                retries += 1
                time.sleep(2)
        
        print(f"搜索失败: {query}")
        return [{"error": "搜索服务不可用，请稍后再试"}]
    
    def get_page_content(self, url, timeout=5):
        """获取网页内容"""
        if not url.startswith('http'):
            return ""
        
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"获取页面内容失败: {url} - {str(e)}")
            return ""
    
    def tag_visible(self, element):
        """检查HTML元素是否可见"""
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True
    
    def extract_relevant_content(self, html_content, query, max_length=500):
        """从HTML内容中提取与查询相关的部分"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取所有可见文本
            texts = soup.findAll(string=True)
            visible_texts = filter(self.tag_visible, texts)
            text = " ".join(t.strip() for t in visible_texts if t.strip())
            
            # 清理文本
            text = re.sub(r'\s+', ' ', text)
            
            # 查找与查询相关的部分
            query_words = set(query.lower().split())
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                if query_words & sentence_words:  # 如果有共同词汇
                    relevant_sentences.append(sentence)
            
            # 选择最相关的部分
            if relevant_sentences:
                # 按包含查询词的数量排序
                relevant_sentences.sort(
                    key=lambda s: len(set(re.findall(r'\b\w+\b', s.lower())) & query_words),
                    reverse=True
                )
                result = " ".join(relevant_sentences[:3])[:max_length]
            else:
                # 没有找到直接相关的内容，返回开头部分
                result = text[:max_length]
            
            # 添加省略号如果内容被截断
            if len(result) == max_length:
                result += "..."
                
            return result
        except Exception as e:
            print(f"提取内容失败: {str(e)}")
            return ""
    
    def format_results(self, results):
        """完整实现：格式化搜索结果"""
        if not results:
            return "没有找到相关信息"
        
        # 处理错误情况
        if "error" in results[0]:
            return f"搜索错误: {results[0]['error']}"
        
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            title = result.get('title', '无标题').strip()
            url = result.get('url', '无URL').strip()
            snippet = result.get('snippet', '无摘要').strip()
            
            # 清理和截断文本
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            
            if len(title) > 80:
                title = title[:77] + "..."
                
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            
            # 格式化结果
            formatted.append(f"结果 {i}: {title}\n链接: {url}\n摘要: {snippet}")
        
        return "\n\n".join(formatted)

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
        evaluation_strategy="epoch",  # 每个epoch后评估
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",  # 分页优化器减少内存峰值
        gradient_checkpointing=True  # 梯度检查点节省显存
    )
    
    # 创建Trainer（包含验证集）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_val_data,
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
    
    # 准备数据集（只使用训练集）
    train_dataset = WebPuzzleDataset(config, tokenizer, mode='train')
    ppo_data = [train_dataset.format_for_ppo(item) for item in train_dataset.data]
    
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
            answer = next((item["answer"] for item in train_dataset.data if item["prompt"] == prompt), "")
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
            difficulty = next((item["difficulty"] for item in train_dataset.data if item["prompt"] == prompt), "medium")
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
        
        # 生成响应
        response_tensors = []
        responses = []
        for query in queries:
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            response = model.generate(**inputs, **generation_kwargs)
            response_tensor = response.squeeze()[-generation_kwargs["max_new_tokens"]:]
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            # 处理搜索请求
            if "搜索：" in response_text:
                # 提取搜索查询
                search_query = response_text.split("搜索：")[1].split("\n")[0].strip()
                
                # 执行实际搜索
                search_results = searcher.search(search_query)
                formatted_results = searcher.format_results(search_results)
                
                # 添加到响应中
                response_text += f"\n搜索结果：{formatted_results}\n"
            
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
    
    # 阶段3: 最终评估
    print("\n" + "="*50)
    print("Starting Final Evaluation")
    print("="*50)
    evaluate_agent(config)
    
    print("\nOpen DeepDiver training completed!")
