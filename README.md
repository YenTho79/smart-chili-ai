# 🌶️ SMART CHILI AI - HỆ THỐNG VƯỜN ỚT THÔNG MINH

> Đề tài: **Hệ Thống Chăm Sóc Vườn Ớt Tự Động Kết Hợp Thị Giác Máy Tính AI**  
> Nhóm tác giả: **Lê Quốc Thắng - Nguyễn Thị Yến Thơ - Danh Thị Hồng Hạnh**  
> Kiến trúc vận hành: **SENSE - SEE - THINK - ACT - LEARN**

📖 **Tài liệu hướng dẫn sử dụng chi tiết toàn tập:** Vui lòng xem tại [HUONG_DAN_SU_DUNG.md](file:///e:/SMART%20CHILI%20AI/HUONG_DAN_SU_DUNG.md).

---

## ⚡ Hướng dẫn khởi chạy nhanh (Quick Start)

### 1. Khởi chạy Server AI & Web Dashboard:
```bash
cd "e:\SMART CHILI AI\python"
pip install -r requirements.txt
python server.py
```
Truy cập giao diện Web tại: **`http://localhost:8000`**

### 2. Sơ đồ chân kết nối ESP32-S3:
- Cảm biến độ ẩm đất: **GPIO 4** (Analog ADC)
- Cảm biến nhiệt ẩm DHT22: **GPIO 1**
- Cảm biến ánh sáng BH1750: **GPIO 11** (SDA), **GPIO 12** (SCL)
- Phao chống cạn nước: **GPIO 5**
- Điều khiển Bơm nước: **GPIO 6**
- Điều khiển Quạt: **GPIO 7**
- Nút bấm khẩn cấp E-Stop: **GPIO 0** (Nút BOOT)

### 3. Lưu ý khi nạp Firmware Arduino IDE:
Trước khi bấm nút Nạp code (Upload) trong Arduino IDE, hãy vào Web Dashboard bấm nút **"Ngắt để nạp code"** để giải phóng cổng COM6, tránh lỗi `PermissionError: COM6 Access is denied`. Nạp xong bấm **"Kết nối lại"**.
