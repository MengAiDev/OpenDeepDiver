from transformers import AutoTokenizer, AutoModelForCausalLM
from search_module import DuckDuckGoSearcher
import torch

class DeepDiverAgent:
    def __init__(self, config):
        from transformers import modeling_utils
        if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

        # Fix: Ensure tokenizer uses local_files_only=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.sft_output_dir,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.ppo_output_dir, # Changed from "./sft_model" to use same path as tokenizer
            quantization_config=config.bnb_config if config.use_4bit else None,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True  # Ensure local files only
        )
        self.model.eval()
        self.searcher = DuckDuckGoSearcher(max_results=3)
        self.max_steps = 5
        self.max_search_queries = 5
        
    def generate(self, prompt):
        full_response = ""
        search_count = 0
        search_queries = []
        
        for step in range(self.max_steps):
            input_text = prompt + full_response
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            new_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            full_response += new_text
            
            if "搜索：" in new_text and search_count < self.max_search_queries:
                search_count += 1
                search_query = new_text.split("搜索：")[1].split("\n")[0].strip()
                search_queries.append(search_query)
                
                search_results = self.searcher.search(search_query)
                formatted_results = self.searcher.format_results(search_results)
                
                full_response += f"\n搜索结果：{formatted_results}\n"
            
            if "答案：" in new_text:
                break
        
        return {
            "response": full_response,
            "search_count": search_count,
            "search_queries": search_queries,
            "steps": step + 1
        }
