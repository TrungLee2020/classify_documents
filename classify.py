from client import Qwen3Client
import logging
from typing import Dict
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentClassifier:
    def __init__(self, qwen_client: Qwen3Client):
        self.qwen_client = qwen_client
        self.categories = {
            0: "Thông báo",
            1: "Tài chính"
        }
    
    def classify_document(self, summary_text: str) -> Dict:
        """Phân loại văn bản dựa trên tóm tắt"""
        
        prompt = f"""Phân loại văn bản sau thuộc loại nào:
- Loại 0: Thông báo (thông báo nội bộ, công văn, hướng dẫn, quy định)
- Loại 1: Tài chính (báo cáo tài chính, bảng cân đối kế toán, báo cáo doanh thu, lợi nhuận)

Nội dung cần phân loại:
{summary_text}

Hãy trả lời theo format chính xác:
Loại: [0 hoặc 1]
Độ tin cậy: [số từ 0.0 đến 1.0]
Lý do: [giải thích ngắn gọn]"""

        response = self.qwen_client.generate_text(prompt, max_tokens=150, temperature=0.0)
        
        # Parse response
        category, confidence, reason = self._parse_classification_response(response)
        
        return {
            "category": self.categories.get(category, "Không xác định"),
            "category_id": category,
            "confidence": confidence,
            "reason": reason,
            "summary": summary_text
        }
    
    def _parse_classification_response(self, response: str) -> tuple:
        """Parse response từ Qwen3"""
        try:
            lines = response.strip().split('\n')
            category = 0
            confidence = 0.5
            reason = "Không có lý do"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Loại:"):
                    try:
                        category = int(line.split(":")[1].strip())
                    except:
                        category = 0
                elif line.startswith("Độ tin cậy:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith("Lý do:"):
                    reason = line.split(":", 1)[1].strip()
            
            return category, confidence, reason
            
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return 0, 0.5, "Lỗi phân tích"