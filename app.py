from flask import Flask, request, jsonify, render_template_string
import os
from werkzeug.utils import secure_filename
import tempfile

from processor import DocumentProcessor
from classify import DocumentClassifier
from main import DocumentAnalyzer
import requests
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Khởi tạo analyzer
analyzer = DocumentAnalyzer("http://localhost:8080")

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phân loại Văn bản</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>Phân loại Văn bản Tài chính vs Thông báo</h1>
        <form action="/classify" method="post" enctype="multipart/form-data">
            <input type="file" name="document" accept=".pdf" required>
            <br><br>
            <button type="submit">Phân loại</button>
        </form>
    </body>
    </html>
    ''')

@app.route('/classify', methods=['POST'])
def classify_document():
    try:
        if 'document' not in request.files:
            return jsonify({"error": "Không có file được upload"}), 400
        
        file = request.files['document']
        if file.filename == '':
            return jsonify({"error": "Không có file được chọn"}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            # Lưu file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                file.save(tmp_file.name)
                
                # Phân tích
                result = analyzer.analyze_document(tmp_file.name)
                
                # Xóa file tạm
                os.unlink(tmp_file.name)
                
                return jsonify(result)
        else:
            return jsonify({"error": "Chỉ chấp nhận file PDF"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Kiểm tra trạng thái llama-server"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            return jsonify({"status": "healthy", "llama_server": "running"})
        else:
            return jsonify({"status": "unhealthy", "llama_server": "error"})
    except:
        return jsonify({"status": "unhealthy", "llama_server": "offline"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)