import requests
import json
from typing import Dict, List
import logging

class Qwen3Client:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1, enable_thinking: bool = False) -> str:
        """Gọi API của llama-server"""
        
        # Thêm instruction để disable thinking nếu cần
        if not enable_thinking:
            prompt = "/no_think\n" + prompt
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "[/INST]"],
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/completion",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "").strip()
        except Exception as e:
            logging.error(f"Error calling Qwen3 API: {e}")
            return ""