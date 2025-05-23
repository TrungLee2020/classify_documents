import requests
import json
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)

class Qwen3Client:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Gọi API của llama-server với disable thinking"""
        
        # Disable thinking mode
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
                timeout=100
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "").strip()
        except Exception as e:
            logging.error(f"Error calling Qwen3 API: {e}")
            return ""

class DocumentProcessor:
    def __init__(self, qwen_client: Qwen3Client):
        self.qwen_client = qwen_client
    
    def summarize_text(self, text: str) -> str:
        """Tóm tắt văn bản sử dụng Qwen3"""
        if len(text) < 100:
            return text
        
        # Cắt text nếu quá dài
        if len(text) > 1500:
            text = text[:1500] + "..."
        
        prompt = f"""Hãy tóm tắt nội dung văn bản sau thành 2-3 câu ngắn gọn, tập trung vào thông tin chính:

        Văn bản:
        {text}

        Tóm tắt:"""
        
        summary = self.qwen_client.generate_text(prompt, max_tokens=200, temperature=0.1)
        
        return summary if summary else text[:300]

class DocumentClassifier:
    def __init__(self, qwen_client: Qwen3Client):
        self.qwen_client = qwen_client
        self.categories = {
            0: "Thông báo",
            1: "Tài chính"
        }
    
    def classify_document(self, text: str) -> Dict:
        """Phân loại văn bản dựa trên nội dung"""
        
        prompt = f"""Phân loại văn bản sau thuộc loại nào:
- Loại 0: Thông báo (thông báo nội bộ, công văn, hướng dẫn, quy định, thông báo sự kiện)
- Loại 1: Tài chính (báo cáo tài chính, doanh thu, lợi nhuận, đầu tư, thuế, ngân hàng, bảo hiểm)

Nội dung cần phân loại:
{text}

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
            "raw_response": response
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

def test_classification_with_sample_data():
    """Test phân loại với dữ liệu mẫu"""
    
    # Dữ liệu mẫu - Thông báo (Label: 0)
    thong_bao_samples = [
        "Kính gửi toàn thể nhân viên phòng Kinh doanh, Trưởng phòng yêu cầu một cuộc họp khẩn cấp vào lúc 14h00 chiều nay, ngày [Ngày/Tháng/Năm], tại phòng họp số 2. Nội dung: Giải quyết vấn đề phát sinh từ hợp đồng khách hàng X. Yêu cầu tất cả các thành viên có mặt đầy đủ và đúng giờ.",
        
        "Công ty Cổ phần ABC trân trọng thông báo đến toàn thể Quý Đối tác và Khách hàng lịch nghỉ lễ Giỗ Tổ Hùng Vương (Mùng 10/3 Âm lịch) như sau: Thời gian nghỉ: Thứ Năm, ngày [Ngày/Tháng/Năm]. Công ty sẽ hoạt động trở lại bình thường vào Thứ Sáu, ngày [Ngày/Tháng/Năm]. Kính chúc Quý vị một kỳ nghỉ lễ vui vẻ!",
        
        "Kính gửi Quý Khách hàng, kể từ ngày [Ngày/Tháng/Năm], Văn phòng Giao dịch của Công ty TNHH XYZ sẽ chính thức chuyển về địa điểm mới tại: Tầng 5, Tòa nhà Central Park, Số 123 Đường DEF, Quận GHI, Thành phố JKL. Mọi thông tin liên hệ khác không thay đổi. Rất mong tiếp tục nhận được sự ủng hộ của Quý vị.",
        
        "Công ty Phát triển Công nghệ Alpha đang có nhu cầu tuyển dụng 02 vị trí Lập trình viên Java với kinh nghiệm từ 2 năm trở lên. Ứng viên quan tâm vui lòng gửi CV về địa chỉ email: tuyendung@alpha-tech.vn trước ngày [Ngày/Tháng/Năm]. Chi tiết mô tả công việc vui lòng xem tại website của công ty.",
        
        "Thông báo về việc bảo trì hệ thống máy chủ: Để nâng cao chất lượng dịch vụ, chúng tôi sẽ tiến hành bảo trì hệ thống từ 23h00 ngày [Ngày/Tháng/Năm] đến 05h00 ngày [Ngày/Tháng/Năm]. Trong thời gian này, các dịch vụ trực tuyến có thể bị gián đoạn. Mong Quý khách hàng thông cảm cho sự bất tiện này."
    ]
    
    # Dữ liệu mẫu - Tài chính (Label: 1)
    tai_chinh_samples = [
        "Dự án đầu tư vào nhà máy sản xuất Y sau một năm vận hành đã cho thấy tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE) đạt 18%, vượt 3% so với mục tiêu ban đầu. Dòng tiền thuần từ hoạt động kinh doanh dương ổn định, cho thấy tiềm năng tăng trưởng bền vững và khả năng thu hồi vốn nhanh.",
        
        "Kết thúc Quý III/2023, Công ty Cổ phần Z ghi nhận doanh thu thuần đạt 250 tỷ đồng, tăng 15% so với cùng kỳ năm ngoái. Lợi nhuận sau thuế đạt 25 tỷ đồng, hoàn thành 110% kế hoạch quý. Kết quả này chủ yếu đến từ sự tăng trưởng mạnh mẽ của mảng sản phẩm chủ lực và kiểm soát chi phí hiệu quả.",
        
        "Để giảm thiểu rủi ro và tối ưu hóa lợi nhuận trong bối cảnh thị trường biến động, nhà đầu tư nên cân nhắc đa dạng hóa danh mục đầu tư. Việc phân bổ vốn vào các loại tài sản khác nhau như cổ phiếu, trái phiếu, bất động sản và vàng có thể giúp cân bằng rủi ro và nắm bắt cơ hội tăng trưởng từ nhiều nguồn.",
        
        "Theo Nghị định 123/2023/NĐ-CP, kể từ ngày 01/01/2024, một số quy định về thuế thu nhập doanh nghiệp sẽ được điều chỉnh, bao gồm việc giảm thuế suất cho doanh nghiệp nhỏ và vừa. Các doanh nghiệp cần cập nhật thông tin để đảm bảo tuân thủ và tận dụng các ưu đãi (nếu có).",
        
        "Biến động tỷ giá USD/VND trong tháng qua chủ yếu do tác động từ chính sách tiền tệ của Cục Dự trữ Liên bang Mỹ (FED) và dòng vốn đầu tư nước ngoài. Các doanh nghiệp xuất nhập khẩu cần theo dõi sát diễn biến tỷ giá để có chiến lược phòng ngừa rủi ro phù hợp."
    ]
    
    # Khởi tạo components
    qwen_client = Qwen3Client("http://localhost:8080")
    processor = DocumentProcessor(qwen_client)
    classifier = DocumentClassifier(qwen_client)
    
    # Test với tóm tắt
    def test_with_summarization(samples, expected_label, category_name):
        print(f"\n{'='*60}")
        print(f"TESTING {category_name.upper()} - With Summarization")
        print(f"{'='*60}")
        
        correct = 0
        total = len(samples)
        
        for i, text in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"Original text: {text[:100]}...")
            
            # Tóm tắt
            summary = processor.summarize_text(text)
            print(f"Summary: {summary}")
            
            # Phân loại
            result = classifier.classify_document(summary)
            print(f"Predicted: {result['category']} (ID: {result['category_id']})")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reason: {result['reason']}")
            
            if result['category_id'] == expected_label:
                correct += 1
                print("✅ CORRECT")
            else:
                print("❌ WRONG")
        
        accuracy = correct / total * 100
        print(f"\n{category_name} Accuracy: {correct}/{total} = {accuracy:.1f}%")
        return accuracy
    
    # Test trực tiếp không tóm tắt
    def test_direct_classification(samples, expected_label, category_name):
        print(f"\n{'='*60}")
        print(f"TESTING {category_name.upper()} - Direct Classification")
        print(f"{'='*60}")
        
        correct = 0
        total = len(samples)
        
        for i, text in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"Text: {text[:100]}...")
            
            # Phân loại trực tiếp
            result = classifier.classify_document(text)
            print(f"Predicted: {result['category']} (ID: {result['category_id']})")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reason: {result['reason']}")
            
            if result['category_id'] == expected_label:
                correct += 1
                print("✅ CORRECT")
            else:
                print("❌ WRONG")
        
        accuracy = correct / total * 100
        print(f"\n{category_name} Accuracy: {correct}/{total} = {accuracy:.1f}%")
        return accuracy
    
    # Kiểm tra kết nối server
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        print("✅ Llama-server is running")
    except:
        print("❌ Cannot connect to llama-server. Make sure it's running on localhost:8080")
        return
    
    # Chạy test
    print("🚀 STARTING CLASSIFICATION TESTS")
    
    # Test với tóm tắt
    tb_acc_summ = test_with_summarization(thong_bao_samples, 0, "THÔNG BÁO")
    tc_acc_summ = test_with_summarization(tai_chinh_samples, 1, "TÀI CHÍNH")
    
    # Test trực tiếp
    tb_acc_direct = test_direct_classification(thong_bao_samples, 0, "THÔNG BÁO")
    tc_acc_direct = test_direct_classification(tai_chinh_samples, 1, "TÀI CHÍNH")
    
    # Tổng kết
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"With Summarization:")
    print(f"  - Thông báo: {tb_acc_summ:.1f}%")
    print(f"  - Tài chính: {tc_acc_summ:.1f}%")
    print(f"  - Overall: {(tb_acc_summ + tc_acc_summ)/2:.1f}%")
    
    print(f"\nDirect Classification:")
    print(f"  - Thông báo: {tb_acc_direct:.1f}%")
    print(f"  - Tài chính: {tc_acc_direct:.1f}%")
    print(f"  - Overall: {(tb_acc_direct + tc_acc_direct)/2:.1f}%")

if __name__ == "__main__":
    test_classification_with_sample_data()