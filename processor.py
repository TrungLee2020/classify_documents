from client import Qwen3Client
import logging
import fitz
from typing import Dict, List
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self, qwen_client: Qwen3Client):
        self.qwen_client = qwen_client
    
    def extract_first_page(self, pdf_path: str) -> str:
        """Trích xuất text từ trang đầu PDF bằng PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                page = doc.load_page(0)  # Trang đầu tiên (index 0)
                text = page.get_text()
                return text.strip()
            return ""
        except Exception as e:
            logging.error(f"Error extracting PDF with PyMuPDF: {e}")
            return ""
    
    def summarize_text(self, text: str) -> str:
        """Tóm tắt văn bản sử dụng Qwen3"""
        if len(text) < 100:
            return text
        
        # Cắt text nếu quá dài (tránh vượt quá context length)
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        prompt = f"""Hãy tóm tắt nội dung văn bản sau thành 2-3 câu ngắn gọn, tập trung vào thông tin chính:

Văn bản:
{text}

Tóm tắt:"""
        
        summary = self.qwen_client.generate_text(prompt, max_tokens=200, temperature=0.1)
        return summary if summary else text[:500]