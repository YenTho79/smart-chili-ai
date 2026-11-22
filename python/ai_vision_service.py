import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class AIVisionService:
    # 1. Khởi tạo và khai báo 4 lớp nhận diện bệnh học cây ớt
    CLASS_NAMES = [
        "Healthy",              # Lá xanh tươi, sinh trưởng tốt
        "Wilting",              # Lá héo rũ, mất nước nghiêm trọng
        "Leaf Spot",            # Đốm lá vi khuẩn / nấm hoại tử
        "Nutrient Deficiency"   # Thiếu hụt dinh dưỡng, vàng lá gân xanh
    ]

    def __init__(self, model_weight_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.CLASS_NAMES
        self.num_classes = len(self.classes)

        # 1. Khởi tạo kiến trúc MobileNetV3 Small siêu nhẹ, tối ưu biên nhúng
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, self.num_classes)
        
        if model_weight_path:
            try:
                self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            except Exception as e:
                print(f"Lưu ý: Khởi tạo trọng số cơ sở (chưa nạp custom weight: {e})")

        self.model.to(self.device)
        self.model.eval()

        # 2. Tiền xử lý chuẩn hóa ảnh ImageNet (Resize 224x224, Normalize)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def check_image_quality(self, cv_img):
        """2. Kiểm tra chất lượng hình ảnh (độ sáng quá tối/cháy sáng và độ mờ/nhiễu)"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        is_dark = mean_brightness < 40.0
        is_overexposed = mean_brightness > 230.0
        is_blurry = laplacian_var < 55.0

        quality_status = "PASS"
        warnings = []
        if is_dark:
            quality_status = "FAIL"
            warnings.append("Ảnh quá tối (độ sáng < 40)")
        elif is_overexposed:
            quality_status = "FAIL"
            warnings.append("Ảnh cháy sáng (độ sáng > 230)")
        if is_blurry:
            quality_status = "WARN"
            warnings.append("Ảnh bị mờ/rung lắc (độ nét Laplacian < 55)")

        return {
            "status": quality_status,
            "brightness": round(float(mean_brightness), 1),
            "sharpness": round(float(laplacian_var), 1),
            "warnings": warnings
        }

    def predict(self, input_image):
        """3. Hàm predict(): Đọc ảnh, chạy model, tính Softmax, Confidence và trả kết quả"""
        start_time = time.perf_counter()

        # Đọc ảnh từ đường dẫn hoặc mảng numpy
        if isinstance(input_image, str):
            cv_img = cv2.imread(input_image)
            if cv_img is None:
                raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {input_image}")
        elif isinstance(input_image, np.ndarray):
            cv_img = input_image
        else:
            raise TypeError("input_image phải là filepath (str) hoặc numpy.ndarray BGR")

        # Kiểm tra chất lượng quang học trước khi suy luận
        quality = self.check_image_quality(cv_img)

        # Tiền xử lý RGB -> PIL -> Tensor chuẩn hóa
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Suy luận Deep Learning MobileNetV3
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        pred_class = self.classes[predicted_idx.item()]
        conf_score = round(confidence.item() * 100.0, 2)

        # Trả kết quả dự đoán chi tiết phân phối xác suất
        prob_dist = {
            self.classes[i]: round(probabilities[i].item() * 100.0, 2)
            for i in range(self.num_classes)
        }

        return {
            "prediction": pred_class,
            "confidence": conf_score,
            "class_index": int(predicted_idx.item()),
            "probabilities": prob_dist,
            "inference_time_ms": inference_time_ms,
            "device": str(self.device),
            "quality_check": quality
        }

if __name__ == "__main__":
    service = AIVisionService()
    print("Khởi tạo MobileNetV3 Deep Learning thành công!")
    print(f"Các lớp phân loại: {service.classes}")
