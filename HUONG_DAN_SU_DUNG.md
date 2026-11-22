# 🌶️ HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
## SÁCH HƯỚNG DẪN CÀI ĐẶT, VẬN HÀNH & BÁO CÁO KỸ THUẬT TOÀN DIỆN

---

### 👥 THÔNG TIN DỰ ÁN & THÀNH VIÊN THỰC HIỆN
- **Tên đề tài:** Hệ Thống Chăm Sóc Vườn Ớt Tự Động Kết Hợp Thị Giác Máy Tính AI (Smart Chili AI System)
- **Kiến trúc vận hành:** **SENSE** (Cảm biến) ➔ **SEE** (Camera & YOLOv8) ➔ **THINK** (Logic mờ Fuzzy & Decision Tree) ➔ **ACT** (Bơm, Quạt, Relay) ➔ **LEARN** (Lưu trữ Database & Re-train AI)
- **Thành viên nhóm:**
  1. **Lê Quốc Thắng**
  2. **Nguyễn Thị Yến Thơ**
  3. **Danh Thị Hồng Hạnh**

---

## 📌 MỤC LỤC
1. [Sơ Đồ Đấu Nối Phần Cứng (ESP32-S3 Pinout)](#1-sơ-đồ-đấu-nối-phần-cứng-esp32-s3-pinout)
2. [Hướng Dẫn Nạp Firmware Cho ESP32-S3](#2-hướng-dẫn-nạp-firmware-cho-esp32-s3)
   - *Cách khắc phục lỗi nạp code "PermissionError: COM6 Access is denied"*
3. [Hướng Dẫn Cài Đặt Môi Trường & Khởi Chạy Server AI](#3-hướng-dẫn-cài-đặt-môi-trường--khởi-chạy-server-ai)
4. [Hướng Dẫn Vận Hành Web Dashboard](#4-hướng-dẫn-vận-hành-web-dashboard)
5. [Hệ Thống Thị Giác Máy Tính (YOLOv8 + Segmentation)](#5-hệ-thống-thị-giác-máy-tính-yolov8--segmentation)
6. [Hệ Thống Ra Quyết Định Thông Minh (Fuzzy Weather & ML)](#6-hệ-thống-ra-quyết-định-thông-minh-fuzzy-weather--ml)
7. [Kịch Bản Trình Diễn Thực Tế (Demo Test Scenarios)](#7-kịch-bản-trình-diễn-thực-tế-demo-test-scenarios)
8. [Bảng Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#8-bảng-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. SƠ ĐỒ ĐẤU NỐI PHẦN CỨNG (ESP32-S3 PINOUT)

> [!IMPORTANT]
> Toàn bộ chân kết nối được thiết kế chuẩn 100% theo tài liệu kỹ thuật của dự án, đảm bảo an toàn điện và độ nhạy tín hiệu tối đa.

| STT | Thiết bị / Module | Loại tín hiệu | Chân ESP32-S3 | Chân Module / Relay | Ghi chú kỹ thuật |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Cảm biến độ ẩm đất** | Analog ADC | **GPIO 4** | Chân **A0** | Tích hợp thuật toán lọc trung vị (Median Filter) 10 mẫu để chống nhiễu |
| 2 | **Cảm biến DHT22** | Digital 1-Wire | **GPIO 1** | Chân **DATA** | Đọc nhiệt độ (°C) và độ ẩm không khí (%RH) |
| 3 | **Cảm biến ánh sáng BH1750** | I2C Data | **GPIO 11** | Chân **SDA** | Đo cường độ sáng Lux (0 - 65,535 lx) |
| 4 | **Cảm biến ánh sáng BH1750** | I2C Clock | **GPIO 12** | Chân **SCL** | Xung nhịp I2C chuẩn 100kHz |
| 5 | **Phao báo mức nước bồn** | Digital Input | **GPIO 5** | Cực đóng ngắt | Cấu hình `INPUT_PULLUP`. Báo cạn ngắt bơm để bảo vệ motor |
| 6 | **Relay Bơm tưới nước** | Digital Output | **GPIO 6** | Chân **IN1 (L298N/Relay)** | Kích mức HIGH (1) để bơm, LOW (0) để ngắt |
| 7 | **Relay Quạt làm mát** | Digital Output | **GPIO 7** | Chân **IN2 (L298N/Relay)** | Tự động bật khi nhiệt độ cao hoặc nông dân bật tay |
| 8 | **Nút bấm khẩn cấp (E-Stop)** | Hardware Interrupt | **GPIO 0** | Nút **BOOT** | Bấm để ngắt toàn bộ relay ngay lập tức (bảo vệ an toàn) |
| 9 | **Nguồn cấp cảm biến** | Power | **3.3V / 5V** | VCC | Cấp nguồn riêng cho L298N/Relay và động cơ bơm |
| 10| **Mass chung** | Ground | **GND** | GND | Nối chung toàn bộ Mass của nguồn và vi điều khiển |

---

## 2. HƯỚNG DẪN NẠP FIRMWARE CHO ESP32-S3

### 2.1. Chuẩn bị phần mềm Arduino IDE
1. Tải và cài đặt **Arduino IDE 2.x**.
2. Thêm URL Board ESP32: `https://espressif.github.io/arduino-esp32/package_esp32_index.json` vào Preferences.
3. Trong Board Manager, cài đặt gói **esp32 by Espressif Systems**.
4. Cài đặt các thư viện cần thiết trong Library Manager:
   - `DHT sensor library` by Adafruit
   - `Adafruit Unified Sensor`
   - `BH1750` by Christopher Laws
   - `ArduinoJson` (phiên bản 6.x hoặc 7.x)

### 2.2. Thiết lập cấu hình Board
- **Board:** `ESP32S3 Dev Module`
- **Port:** Chọn đúng cổng COM của mạch (ví dụ `COM6`)
- **USB CDC On Boot:** `Enabled` (Để đọc Serial qua cổng USB cắm trực tiếp)
- **Flash Size:** `8MB` hoặc `16MB`
- **Upload Speed:** `921600` hoặc `115200`

### 2.3. Quy trình nạp code & Xử lý lỗi "PermissionError: COM6 Access is denied"
> [!WARNING]
> Trên hệ điều hành Windows, một cổng COM chỉ cho phép DUY NHẤT một phần mềm mở tại một thời điểm. Nếu máy chủ Python (`server.py`) đang kết nối cổng COM để nhận dữ liệu, Arduino IDE sẽ báo lỗi:
> `A fatal error occurred: Could not open COM6, the port is busy or doesn't exist. PermissionError(13, 'Access is denied.')`

**Các bước nạp code an toàn:**
1. **Cách 1 (Nhanh nhất từ Dashboard Web):**
   - Mở giao diện Web tại `http://localhost:8000`.
   - Tại mục **Cổng Serial ESP32**, bấm nút **"Ngắt để nạp code"**.
   - Server Python sẽ lập tức nhả cổng COM6.
   - Quay lại Arduino IDE, bấm nút **Upload (Mũi tên phải ➔)** để nạp code.
   - Sau khi nạp xong 100%, trên giao diện Web bấm **"Kết nối lại"**.
2. **Cách 2 (Bằng dòng lệnh/Tắt server):**
   - Đóng cửa sổ chạy server Python hoặc tắt tiến trình `python server.py`.
   - Nạp code trên Arduino IDE.
   - Khởi động lại `python server.py`.

---

## 3. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG & KHỞI CHẠY SERVER AI

### 3.1. Cài đặt thư viện Python
Mở Terminal / PowerShell tại thư mục dự án và chạy:
```bash
cd "e:\SMART CHILI AI\python"
pip install fastapi uvicorn opencv-python ultralytics pyserial requests scikit-fuzzy joblib scikit-learn
```

### 3.2. Khởi chạy máy chủ Backend & AI
Chạy lệnh khởi động máy chủ:
```bash
cd "e:\SMART CHILI AI\python"
python server.py
```
Khi màn hình hiển thị:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
➔ Máy chủ đã chạy thành công! Truy cập giao diện tại: **`http://localhost:8000`** hoặc **`http://127.0.0.1:8000`**.

---

## 4. HƯỚNG DẪN VẬN HÀNH WEB DASHBOARD

Giao diện Web được thiết kế theo phong cách hiện đại (Pastel Garden UI), trực quan và hỗ trợ đầy đủ các tính năng:

```
+---------------------------------------------------------------------------------------+
|  🌶️ SMART CHILI AI - HỆ THỐNG VƯỜN ỚT THÔNG MINH                                     |
|  [Trạng thái: KẾT NỐI ESP32 (COM6)]  [Chế độ: AUTO / MANUAL]  [E-STOP: BÌNH THƯỜNG]   |
+---------------------------------------------------------------------------------------+
| THÔNG SỐ CẢM BIẾN THỜI GIAN THỰC:                                                    |
|  [Độ ẩm đất: 45%]   [Nhiệt độ: 28.5°C]   [Độ ẩm KK: 72%]   [Ánh sáng: 8500 Lux]      |
|  [Mực nước: ĐẦY BỒN] [Dự báo mưa: 15% (Trời nắng ráo)]                                |
+---------------------------------------------------+-----------------------------------+
|  CAMERA FULL HD & AI NHẬN DIỆN THỜI GIAN THỰC    |  BẢNG THỐNG KÊ AI (6 LỚP ĐỐI TƯỢNG)|
|  [ LIVE STREAM MJPEG (30 FPS) ]                   |  🔴 Ớt chín đỏ:     3 trái        |
|  * Khung bao Bounding Box & Mặt nạ Segmentation   |  🟢 Ớt non xanh:    5 trái        |
|  * Tỷ lệ tin cậy AI (Confidence %): 80% - 95%     |  🟤 Ớt hỏng/hư:     0 trái        |
|  * Phân biệt rõ từng loại quả và từng dạng lá     |  ⭐ Tổng số trái:   8 trái        |
|  * Cảnh báo lá héo (thiếu nước) & lá sâu bệnh     |  🌿 Lá xanh khỏe:   12 lá         |
|  * Nhận diện và đếm 6 lớp theo thời gian thực     |  🍂 Lá héo rũ:      0 lá          |
|                                                   |  🍁 Lá bệnh/vàng:   0 lá          |
+---------------------------------------------------+-----------------------------------+
|  BẢNG ĐIỀU KHIỂN & BẬT TẮT THIẾT BỊ:                                                  |
|  [ Nút BẬT/TẮT BƠM ]   [ Nút BẬT/TẮT QUẠT ]   [ NÚT DỪNG KHẨN CẤP E-STOP ]            |
|  [ Nút NGẮT CỔNG COM (Để nạp code Arduino) ]  [ Nút KẾT NỐI LẠI CỔNG COM ]            |
+---------------------------------------------------------------------------------------+
|  LOG HỆ THỐNG SERIAL MONITOR (Nhật ký gửi nhận JSON theo chu kỳ 2 giây)               |
+---------------------------------------------------------------------------------------+
```

### Các nút chức năng chính:
1. **Chế độ Vận Hành:**
   - **AUTO (Tự động):** Hệ thống dùng thuật toán Logic mờ (Fuzzy Logic) kết hợp thời tiết trực tuyến để tự động kích hoạt bơm và quạt.
   - **MANUAL (Thủ công):** Nông dân được quyền chủ động bật/tắt bơm và quạt bằng các nút nhấn trên Web.
2. **Bộ Cài Đặt Ngưỡng Tự Động (Chỉnh Tay Linh Hoạt):**
   - 💧 **Đất khô kích hoạt tưới (Min Soil %):** Kéo thanh trượt từ 10% - 60% (Mặc định 35%). Khi độ ẩm đất tụt xuống dưới ngưỡng này, hệ thống tự động kích hoạt tưới.
   - 🛑 **Đất đủ ẩm ngắt tưới (Max Soil %):** Kéo thanh trượt từ 40% - 90% (Mặc định 65%). Ngắt bơm ngay khi đất đạt độ ẩm mong muốn.
   - 🌀 **Nhiệt độ bật quạt (Fan Temp °C):** Kéo thanh trượt từ 25°C - 45°C (Mặc định 32°C). Tự động bật quạt giải nhiệt khi chuồng/vườn quá nóng.
   - 🌧️ **Xác suất mưa hoãn tưới (Rain Delay %):** Kéo thanh trượt từ 20% - 90% (Mặc định 60%). Tiết kiệm nước tưới nếu trời sắp đổ mưa.
   - 🎯 **Độ nhạy AI Camera (Confidence %):** Kéo thanh trượt từ 5% - 80% (Mặc định 15%). Lọc khung nhận diện YOLO nhạy hơn hoặc khắt khe hơn theo ý muốn.
   - Có nút **"Lưu Cấu Hình Ngưỡng"** (tự lưu vĩnh viễn vào file `thresholds.json`) và nút **"Mặc Định"** để khôi phục nhanh.
3. **Nút Dừng Khẩn Cấp (E-STOP):**
   - Lập tức ngắt toàn bộ rơ le bơm và quạt, đồng thời khóa lệnh tự động để đảm bảo an toàn tuyệt đối.
4. **Bộ Quản Lý Cổng Nạp Serial:**
   - Hỗ trợ đổi cổng COM linh hoạt và nút **"Ngắt để nạp code"** giúp người dùng nạp firmware mà không phải tắt server.

---

## 5. HỆ THỐNG THỊ GIÁC MÁY TÍNH (YOLOv8 + SEGMENTATION 6 LỚP)

### 5.1. Bộ dữ liệu huấn luyện & 6 Lớp Đối Tượng
Mô hình YOLOv8 được huấn luyện toàn diện trên **622 bức ảnh thực tế** trong thư mục `cay ot/` (gồm 4,972 nhãn đối tượng) chia làm 6 lớp phân loại:
1. **Phân loại trái ớt (3 lớp):**
   - `ot_do` (Ớt chín đỏ - Class 0): Trái ớt đã chín, màu đỏ tươi/đậm, sẵn sàng thu hoạch.
   - `ot_xanh` (Ớt non xanh - Class 1): Trái ớt non đang lớn, dáng hình ống trụ thuôn dài đặc thù.
   - `ot_hong` (Ớt hỏng/thối - Class 2): Trái ớt bị đốm thối đen/nâu, héo khô hoặc hư hỏng trên cây.
2. **Nhận diện trạng thái lá (3 lớp):**
   - `la_xanh` (Lá xanh khỏe mạnh - Class 3): Phiến lá xanh tươi, quang hợp tốt, mép lá căng phẳng.
   - `la_heo` (Lá héo rũ - Class 4): Lá bị mất nước, rũ gục thẳng đứng, cuống mềm (dấu hiệu cây đang thiếu nước).
   - `la_benh` (Lá bệnh / vàng lá - Class 5): Phiến lá xuất hiện đốm vàng, cháy rìa lá, nấm bệnh hoặc thiếu vi lượng.

### 5.2. Giải pháp kỹ thuật nhận diện & phân đoạn (Instance Segmentation)
> [!NOTE]
> Nhóm tích hợp mạng học sâu YOLOv8 kết hợp xử lý phân đoạn hình thái:
> - **Phân đoạn đối tượng (Polygon Segmentation Masking):** Tự động phát hiện đường biên viền thực của từng quả ớt và phiến lá, phủ màu bán trong suốt (Alpha Overlay 42%) giúp người dùng quan sát rõ vị trí và hình khối từng quả trên cành.
> - **Loại trừ nhầm lẫn hình học:** Trái ớt có tỷ lệ dài/rộng $\ge 2.2$ và độ đặc $\ge 0.72$, phân biệt hoàn toàn với hình dáng bản dẹt bầu dục của phiến lá.
> - **Chẩn đoán y sinh học thực vật:** Hệ thống tự động đếm số lá héo, số lá bệnh và quả thối để phát ra cảnh báo tương ứng trên Web Dashboard (ví dụ: phát hiện lá héo thì kích hoạt tăng ưu tiên tưới nước).

### 5.3. Công Cụ Đánh Nhãn & MLOps Active Learning

Hệ thống hỗ trợ 2 phương thức đánh nhãn linh hoạt:

#### Cách 1: Sử dụng Tab "🏷️ AI Labeling Studio" tích hợp sẵn trên Web Dashboard
1. Truy cập Web Dashboard tại `http://localhost:8000`.
2. Bấm vào Tab **"🏷️ AI Labeling Studio (BBox & Segmentation)"** ở thanh trên cùng.
3. Các tính năng tích hợp:
   - **Chế độ vẽ linh hoạt:** Chuyển đổi giữa **Bounding Box (BBox)** (phím tắt `B`) và **Polygon Segmentation** (phím tắt `P`).
   - **Bảng chọn 6 lớp:** Phím tắt số `1` đến `6` cho 6 lớp tương ứng (`ot_do`, `ot_xanh`, `ot_hong`, `la_xanh`, `la_heo`, `la_benh`).
   - **Chụp ảnh mới từ Camera (Snapshot):** Bấm nút 📸 để chụp ngay hình ảnh thực tế từ cây ớt ngoài vườn và nạp vào canvas đánh nhãn.
   - **AI Gợi Ý Nhanh (Auto-Detect):** Bấm nút 🤖 để AI quét và tự động khoanh trước các đối tượng, người dùng chỉ cần kéo chỉnh lại.
   - **Lưu Nhãn (Save):** Bấm nút 💾 hoặc nhấn `Ctrl + S` để lưu nhãn chuẩn YOLO vào thư mục `dataset/labels`.
   - **Huấn Luyện Lại (Re-train):** Bấm nút 🚀 để chạy huấn luyện ngầm mô hình YOLOv8 và cập nhật mô hình mới vào luồng camera trực tiếp.

#### Cách 2: Sử dụng Label Studio chuyên nghiệp (Độc lập)
Máy tính đã được cài đặt sẵn gói `label-studio`. Bạn có thể mở CMD và khởi động:
```bash
label-studio start --port 8081
```
Sau đó truy cập **`http://localhost:8081`** theo đúng tài liệu hướng dẫn (nhập ảnh, chọn template BBox hoặc Polygon Segmentation, xuất dữ liệu dạng YOLO with Images).

---

## 6. HỆ THỐNG RA QUYẾT ĐỊNH THÔNG MINH (FUZZY WEATHER & ML)

Hệ thống kết hợp 2 tầng quyết định:
1. **Tầng 1: Cảnh báo khẩn cấp phần cứng (Hardware Safety Override)**
   - Nếu cảm biến báo độ ẩm đất $< 15\%$: Lập tức kích hoạt tưới khẩn cấp (Emergency Irrigation) bất kể thời tiết để chống héo rễ.
   - Nếu phao báo cạn nước trong bồn: Lập tức ngắt bơm và báo chuông cảnh báo, ngăn ngừa cháy motor.
2. **Tầng 2: Hệ Logic Mờ Mamdani (Mamdani Fuzzy Weather System)**
   - **Đầu vào 1:** Độ ẩm đất hiện tại từ cảm biến (0 - 100%).
   - **Đầu vào 2:** Nhiệt độ môi trường (°C).
   - **Đầu vào 3:** Tỷ lệ dự báo mưa từ OpenWeatherMap API (0 - 100%).
   - **Quy tắc mẫu:**
     - *NẾU Đất Ẩm VỪA VÀ Trời SẮP MƯA (Tỷ lệ mưa > 70%) ➔ KHÔNG TƯỚI (Tiết kiệm nước và tránh ngập úng).*
     - *NẾU Đất KHÔ VÀ Trời NẮNG GẮT (Không mưa) ➔ TƯỚI LƯỢNG NƯỚC LỚN (Thời gian bơm dài).*
     - *NẾU Nhiệt độ $> 33^\circ C$ ➔ TỰ ĐỘNG BẬT QUẠT LÀM MÁT.*

---

## 7. KỊCH BẢN TRÌNH DIỄN THỰC TẾ (DEMO TEST SCENARIOS)

Khi trình bày báo cáo hoặc chấm thi đồ án, nhóm thực hiện theo 4 kịch bản chuẩn sau:

### 🌟 Kịch bản 1: Nhận diện quả ớt và đếm tự động (Computer Vision Demo)
- **Thao tác:** Hướng camera vào chậu cây ớt (có cả ớt chín đỏ và ớt non xanh).
- **Hiện tượng quan sát:** 
  - Khung hình Camera trên Web xuất hiện các viền màu: Màu đỏ cho quả chín, màu xanh ngọc cho quả xanh.
  - Bảng thống kê bên cạnh cập nhật chính xác số lượng từng loại.
  - Các lá cây xanh xung quanh không bị đóng khung nhầm.

### 🌟 Kịch bản 2: Tự động tưới thông minh & Tiết kiệm nước (Smart Irrigation Demo)
- **Thao tác:** Rút cảm biến độ ẩm đất ra khỏi chậu (hoặc để đất khô ráo, giá trị $\approx 20\%$).
- **Hiện tượng quan sát:** 
  - Hệ thống kiểm tra thời tiết online: Nếu trời không mưa, server gửi lệnh kích hoạt Relay Bơm (`{"pump": 1}`).
  - Đèn trên mạch Relay sáng, motor bơm bắt đầu tưới nước vào chậu.
  - Khi cắm lại cảm biến vào đất ẩm ($> 65\%$), bơm tự động tắt (`{"pump": 0}`).

### 🌟 Kịch bản 3: Tình huống bảo vệ cạn nước (Safety Interlock Demo)
- **Thao tác:** Nâng cần phao nước lên (mô phỏng trạng thái bồn chứa bị cạn kiệt nước).
- **Hiện tượng quan sát:** 
  - Trên Web hiển thị cảnh báo đỏ **"CẢNH BÁO: BỒN HẾT NƯỚC"**.
  - Ngay cả khi độ ẩm đất rất khô, lệnh bơm vẫn bị khóa chặt để bảo vệ motor không bị cháy do chạy không tải.

### 🌟 Kịch bản 4: Dừng khẩn cấp bằng phần cứng và phần mềm (E-STOP Demo)
- **Thao tác:** Khi bơm đang chạy, bấm nút **BOOT (GPIO 0)** trên ESP32-S3 hoặc bấm nút **E-STOP** đỏ rực trên Web.
- **Hiện tượng quan sát:** 
  - Ngay lập tức ngắt toàn bộ rơ le trong vòng 10 miligiây.
  - Màn hình chuyển trạng thái **EMERGENCY STOPPED**, hệ thống an toàn tuyệt đối.

---

## 8. BẢNG XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

| Hiện tượng lỗi | Nguyên nhân gốc | Cách khắc phục triệt để |
|:---|:---|:---|
| **Arduino báo `PermissionError: COM6 Access is denied`** | Server Python đang chiếm giữ cổng COM để đọc dữ liệu | Bấm nút **"Ngắt để nạp code"** trên Web Dashboard, sau đó tiến hành nạp lại trên Arduino IDE. |
| **Camera đen màn hình hoặc không lên hình** | Windows chọn nhầm index camera ảo hoặc bị chiếm dụng | Kiểm tra cổng cắm USB của Webcam. Mở file `python/server.py` kiểm tra `CAMERA_INDEX` (thông thường là index `0` hoặc `1`). |
| **Giá trị độ ẩm đất nhảy liên tục hoặc hiển thị 0%** | Dây cắm A0 lỏng hoặc chưa tiếp xúc tốt với đất | Kiểm tra lại dây GPIO 4. Đảm bảo 2 thanh kim loại của cảm biến cắm ngập sâu vào đất. |
| **Web Dashboard không mở được (`localhost:8000`)** | Chưa khởi chạy file `server.py` | Mở Terminal gõ `cd "e:\SMART CHILI AI\python"` và chạy `python server.py`. |
| **Bơm không quay dù relay đã có tiếng nhảy cạch** | Nguồn ngoài cấp cho motor bơm chưa được bật hoặc yếu pin | Bơm cần nguồn ngoài (5V-12V) cấp qua chân nguồn của mạch rơ-le / L298N. Kiểm tra giắc cắm nguồn adapter. |

---

> 🎉 **CHÚC NHÓM BẢO VỆ ĐỒ ÁN VÀ THỰC HIỆN BUỔI DEMO THÀNH CÔNG RỰC RỠ!**
