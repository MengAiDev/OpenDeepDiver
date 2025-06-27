import time
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import html
from duckduckgo_search import DDGS

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
        current_time = time.time()
        if current_time - self.last_search_time < self.min_delay:
            time.sleep(self.min_delay - (current_time - self.last_search_time))
        
        retries = 0
        while retries < self.max_retries:
            try:
                with DDGS() as ddgs:
                    results = []
                    for r in ddgs.text(query, max_results=self.max_results):
                        full_content = self.get_page_content(r.get('href', ''))
                        if full_content:
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
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True
    
    def extract_relevant_content(self, html_content, query, max_length=500):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            texts = soup.findAll(string=True)
            visible_texts = filter(self.tag_visible, texts)
            text = " ".join(t.strip() for t in visible_texts if t.strip())
            text = re.sub(r'\s+', ' ', text)
            
            query_words = set(query.lower().split())
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                if query_words & sentence_words:
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                relevant_sentences.sort(
                    key=lambda s: len(set(re.findall(r'\b\w+\b', s.lower())) & query_words),
                    reverse=True
                )
                result = " ".join(relevant_sentences[:3])[:max_length]
            else:
                result = text[:max_length]
            
            if len(result) == max_length:
                result += "..."
                
            return result
        except Exception as e:
            print(f"提取内容失败: {str(e)}")
            return ""
    
    def format_results(self, results):
        if not results:
            return "没有找到相关信息"
        
        if "error" in results[0]:
            return f"搜索错误: {results[0]['error']}"
        
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            title = result.get('title', '无标题').strip()
            url = result.get('url', '无URL').strip()
            snippet = result.get('snippet', '无摘要').strip()
            
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            
            if len(title) > 80:
                title = title[:77] + "..."
                
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            
            formatted.append(f"结果 {i}: {title}\n链接: {url}\n摘要: {snippet}")
        
        return "\n\n".join(formatted)