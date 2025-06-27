import os
import json
import numpy as np
from tqdm import tqdm

class WebPuzzleDataset:
    def __init__(self, config, tokenizer, mode='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        if self.mode == 'val' and os.path.exists(self.config.val_dataset_path):
            return self.load_val_dataset()
            
        full_data = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading full dataset"):
                item = json.loads(line)
                full_data.append(item)
        
        print(f"Loaded {len(full_data)} samples in total")
        
        total_size = len(full_data)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        rng = np.random.default_rng(42)
        indices = rng.permutation(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        val_data = [full_data[i] for i in val_indices]
        with open(self.config.val_dataset_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved validation dataset with {len(val_data)} samples")
        
        if self.mode == 'train':
            data = [full_data[i] for i in train_indices]
            print(f"Using {len(data)} samples for training")
        else:
            data = val_data
            print(f"Using {len(data)} samples for validation")
        
        formatted_data = []
        for item in data:
            prompt = (
                "你是一个信息搜索专家，需要解决以下问题：\n"
                f"问题：{item['question']}\n"
                "请按步骤进行推理和搜索：\n"
            )
            
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
        data = []
        with open(self.config.val_dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading validation dataset"):
                item = json.loads(line)
                
                prompt = (
                    "你是一个信息搜索专家，需要解决以下问题：\n"
                    f"问题：{item['question']}\n"
                    "请按步骤进行推理和搜索：\n"
                )
                
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
        text = item['prompt'] + "\n推理过程："
        return text
    
    def format_for_ppo(self, item):
        return item['prompt']