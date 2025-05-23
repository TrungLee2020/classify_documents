import json
from main import DocumentAnalyzer
import os


def test_classification():
    analyzer = DocumentAnalyzer("http://localhost:8080")
    
    # Test với file mẫu
    test_files = [
        "baocao_taichih.pdf",  # Báo cáo tài chính
        "thongbao_noibo.pdf"   # Thông báo nội bộ
    ]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n=== Phân tích {pdf_file} ===")
            result = analyzer.analyze_document(pdf_file)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"File {pdf_file} không tồn tại")

if __name__ == "__main__":
    test_classification()