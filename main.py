from client import Qwen3Client
from processor import DocumentProcessor
from classify import DocumentClassifier
import json
import os
from typing import Dict, List
import logging
import requests
import time
import re

class DocumentAnalyzer:
    def __init__(self, llama_server_url: str = "http://localhost:8080"):
        self.qwen_client = Qwen3Client(llama_server_url)
        self.processor = DocumentProcessor(self.qwen_client)
        self.classifier = DocumentClassifier(self.qwen_client)
    
    def analyze_document(self, pdf_path: str) -> Dict:
        """Phân tích và phân loại tài liệu PDF"""
        
        # Bước 1: Trích xuất trang đầu
        print("Đang trích xuất trang đầu...")
        first_page_text = self.processor.extract_first_page(pdf_path)
        
        if not first_page_text:
            return {
                "error": "Không thể trích xuất nội dung từ PDF",
                "category": "Lỗi",
                "confidence": 0.0
            }
        
        # Bước 2: Tóm tắt
        print("Đang tóm tắt nội dung...")
        summary = self.processor.summarize_text(first_page_text)
        
        # Bước 3: Phân loại
        print("Đang phân loại văn bản...")
        classification_result = self.classifier.classify_document(summary)
        
        # Thêm thông tin gốc
        classification_result["original_text_length"] = len(first_page_text)
        classification_result["processing_steps"] = [
            "Trích xuất trang đầu",
            "Tóm tắt nội dung", 
            "Phân loại văn bản"
        ]
        
        return classification_result

# Ví dụ sử dụng
analyzer = DocumentAnalyzer("http://localhost:8080")
result = analyzer.analyze_document("document.pdf")
print(json.dumps(result, ensure_ascii=False, indent=2))