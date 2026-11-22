import os
import io
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.lexers import PythonLexer, CppLexer
from pygments.formatters import ImageFormatter

OUTPUT_DIR = r"d:\smart-chili-ai\minh_chung_cau_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def render_code_card(code_text, lexer, filename, title_text, output_path):
    """Vẽ ảnh chụp code phong cách IDE hiện đại (Dark theme, line numbers, mac-style buttons)"""
    formatter = ImageFormatter(
        font_size=15,
        line_numbers=True,
        line_number_bg="#1e1e24",
        line_number_fg="#6e7681",
        style="monokai"
    )
    code_img_bytes = highlight(code_text.strip(), lexer, formatter)
    code_img = Image.open(io.BytesIO(code_img_bytes)).convert("RGB")

    header_height = 48
    padding = 24
    footer_height = 36

    total_w = code_img.width + padding * 2
    total_h = code_img.height + header_height + footer_height + padding

    card = Image.new("RGB", (total_w, total_h), color="#1e1e24")
    draw = ImageDraw.Draw(card)

    # 3 nút Mac OS: Đỏ, Vàng, Xanh
    draw.ellipse([padding, 18, padding + 13, 31], fill="#ff5f56")
    draw.ellipse([padding + 22, 18, padding + 35, 31], fill="#ffbd2e")
    draw.ellipse([padding + 44, 18, padding + 57, 31], fill="#27c93f")

    # Tiêu đề file
    title = f"{filename} — {title_text}"
    draw.text((padding + 75, 16), title, fill="#abb2bf")

    # Đường kẻ ngăn cách header
    draw.line([(0, header_height), (total_w, header_height)], fill="#2b2d3a", width=1)

    # Dán ảnh code vào giữa
    card.paste(code_img, (padding, header_height + 12))

    # Footer chú thích
    draw.text((padding, total_h - 26), "SMART CHILI AI — MINH CHỨNG CÂU 3", fill="#5c6370")

    card.save(output_path, quality=95)
    print(f"Đã lưu: {os.path.basename(output_path)}")

# ================= 1. CÂU 3.1a: AI / DEEP LEARNING (ai_vision_service.py) =================
code_31a_1 = """# 1. Khởi tạo mô hình MobileNetV3 và khai báo 4 lớp nhận diện bệnh học cây ớt
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

    # Khởi tạo kiến trúc MobileNetV3 Small siêu nhẹ, tối ưu thiết bị nhúng
    self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = self.model.classifier[3].in_features
    # Tinh chỉnh lớp phân loại cuối cùng cho 4 lớp của vườn ớt
    self.model.classifier[3] = nn.Linear(in_features, self.num_classes)
    
    self.model.to(self.device)
    self.model.eval()"""

code_31a_2 = """# 2. Tiền xử lý ảnh (Resize 224x224, Normalize) và kiểm tra chất lượng ảnh
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def check_image_quality(self, cv_img):
    \"\"\"Kiểm tra chất lượng quang học (độ sáng quá tối/cháy sáng và độ mờ/rung lắc)\"\"\"
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_dark = mean_brightness < 40.0
    is_overexposed = mean_brightness > 230.0
    is_blurry = laplacian_var < 55.0

    quality_status = "PASS"
    warnings = []
    if is_dark:
        quality_status = "FAIL"; warnings.append("Ảnh quá tối (độ sáng < 40)")
    elif is_overexposed:
        quality_status = "FAIL"; warnings.append("Ảnh cháy sáng (độ sáng > 230)")
    if is_blurry:
        quality_status = "WARN"; warnings.append("Ảnh bị mờ/rung lắc (Laplacian < 55)")

    return {"status": quality_status, "brightness": round(mean_brightness, 1),
            "sharpness": round(laplacian_var, 1), "warnings": warnings}"""

code_31a_3 = """# 3. Hàm predict() – Đọc ảnh, đưa vào model MobileNetV3, Softmax, Confidence và trả kết quả
def predict(self, input_image):
    start_time = time.perf_counter()
    if isinstance(input_image, str):
        cv_img = cv2.imread(input_image)
    else:
        cv_img = input_image

    # Bước 1: Kiểm tra chất lượng ảnh đầu vào
    quality = self.check_image_quality(cv_img)

    # Bước 2: Chuyển đổi màu và chuẩn hóa Tensor 224x224
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

    # Bước 3: Đưa qua mạng MobileNetV3 và tính phân phối Softmax
    with torch.no_grad():
        outputs = self.model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    pred_class = self.classes[predicted_idx.item()]
    conf_score = round(confidence.item() * 100.0, 2)

    return {
        "prediction": pred_class,
        "confidence": conf_score,
        "class_index": int(predicted_idx.item()),
        "probabilities": {self.classes[i]: round(probabilities[i].item()*100, 2) for i in range(self.num_classes)},
        "inference_time_ms": inference_time_ms,
        "quality_check": quality
    }"""

# ================= 2. CÂU 3.1b: BỘ SUY LUẬN FUZZY LOGIC (fuzzy_engine.py) =================
code_31b_4 = """# 4. Khai báo các biến đầu vào và đầu ra hệ mờ (Fuzzy Antecedents & Consequents)
# 4 Biến đầu vào:
self.soil = ctrl.Antecedent(np.arange(0, 101, 1), 'Soil')           # Độ ẩm đất (%: 0 - 100)
self.temp = ctrl.Antecedent(np.arange(15, 46, 1), 'Temperature')   # Nhiệt độ không khí (°C: 15 - 45)
self.rain = ctrl.Antecedent(np.arange(0, 101, 1), 'Rain')           # Xác suất mưa dự báo (%: 0 - 100)
self.wilt = ctrl.Antecedent(np.arange(0, 101, 1), 'WiltRisk')       # Nguy cơ héo rũ từ AI Vision (%: 0 - 100)

# 2 Biến đầu ra:
self.pump = ctrl.Consequent(np.arange(0, 16, 0.5), 'PumpDuration')  # Thời gian tưới (Giây: 0 - 15s)
self.fan  = ctrl.Consequent(np.arange(0, 256, 1), 'FanPWM')         # Tốc độ quạt làm mát (PWM: 0 - 255)"""

code_31b_5 = """# 5. Thiết kế các hàm thuộc (Membership Functions: Hình thang & Tam giác)
# Độ ẩm đất (Soil)
self.soil['Very_Dry'] = fuzz.trapmf(self.soil.universe, [0, 0, 15, 25])
self.soil['Dry']      = fuzz.trimf(self.soil.universe, [20, 35, 50])
self.soil['Optimal']  = fuzz.trimf(self.soil.universe, [45, 60, 75])
self.soil['Wet']      = fuzz.trapmf(self.soil.universe, [70, 85, 100, 100])

# Nhiệt độ (Temperature)
self.temp['Cool']   = fuzz.trapmf(self.temp.universe, [15, 15, 22, 26])
self.temp['Normal'] = fuzz.trimf(self.temp.universe, [24, 28, 33])
self.temp['Hot']    = fuzz.trapmf(self.temp.universe, [31, 36, 45, 45])

# Xác suất mưa (Rain)
self.rain['Low']    = fuzz.trapmf(self.rain.universe, [0, 0, 20, 40])
self.rain['Medium'] = fuzz.trimf(self.rain.universe, [30, 50, 75])
self.rain['High']   = fuzz.trapmf(self.rain.universe, [65, 80, 100, 100])

# Đầu ra Thời gian bơm (Pump Duration)
self.pump['Off']    = fuzz.trapmf(self.pump.universe, [0, 0, 0.5, 2.0])
self.pump['Short']  = fuzz.trimf(self.pump.universe, [1.5, 4.0, 7.0])
self.pump['Medium'] = fuzz.trimf(self.pump.universe, [5.5, 8.5, 12.0])
self.pump['Long']   = fuzz.trapmf(self.pump.universe, [10.0, 12.5, 15.0, 15.0])

# Đầu ra Quạt làm mát (Fan PWM)
self.fan['Off']    = fuzz.trapmf(self.fan.universe, [0, 0, 30, 60])
self.fan['Low']    = fuzz.trimf(self.fan.universe, [50, 110, 170])
self.fan['Medium'] = fuzz.trimf(self.fan.universe, [140, 190, 230])
self.fan['Max']    = fuzz.trapmf(self.fan.universe, [210, 240, 255, 255])"""

code_31b_6 = """# 6. Tập luật mờ (Rule Base) và hàm evaluate() tính toán giải mờ Centroid
rules = [
    # Đất ẩm hoặc ướt -> Luôn TẮT BƠM chống ngập úng
    ctrl.Rule(self.soil['Optimal'] | self.soil['Wet'], self.pump['Off']),
    # Đất rất khô -> Tưới thời gian dài
    ctrl.Rule(self.soil['Very_Dry'] & self.rain['Low'], self.pump['Long']),
    # Đất khô + Dự báo mưa cao -> HOÃN TƯỚI (Tiết kiệm nước)
    ctrl.Rule(self.soil['Dry'] & self.rain['High'], self.pump['Off']),
    ctrl.Rule(self.soil['Dry'] & self.rain['Medium'], self.pump['Short']),
    ctrl.Rule(self.soil['Dry'] & self.rain['Low'] & self.temp['Hot'], self.pump['Long']),
    # Kết hợp AI: Nguy cơ héo rũ cao -> Tăng cường tưới bù ẩm
    ctrl.Rule(self.wilt['High'] & ~self.soil['Wet'] & self.rain['Low'], self.pump['Long']),
    # Điều khiển quạt làm mát theo nhiệt độ
    ctrl.Rule(self.temp['Cool'], self.fan['Off']),
    ctrl.Rule(self.temp['Hot'], self.fan['Max'])
]
self.sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def evaluate(self, soil_pct, temp_c, rain_pct=0.0, wilt_risk_pct=0.0):
    self.sim.input['Soil'] = np.clip(soil_pct, 0, 100)
    self.sim.input['Temperature'] = np.clip(temp_c, 15, 45)
    self.sim.input['Rain'] = np.clip(rain_pct, 0, 100)
    self.sim.input['WiltRisk'] = np.clip(wilt_risk_pct, 0, 100)
    self.sim.compute()

    raw_pump = float(self.sim.output.get('PumpDuration', 0.0))
    raw_fan  = float(self.sim.output.get('FanPWM', 0.0))
    
    # Khóa an toàn ngưỡng đất ẩm và dự báo mưa
    if soil_pct >= 70.0:
        pump_dur = 0.0; dec = "KHÔNG TƯỚI (Đất đủ ẩm)"
    elif rain_pct >= 65.0 and soil_pct >= 30.0:
        pump_dur = 0.0; dec = f"HOÃN TƯỚI (Dự báo mưa cao {rain_pct}%)"
    else:
        pump_dur = round(min(12.0, raw_pump), 1)
        dec = f"BẬT TƯỚI {pump_dur}s"

    return {"pump_duration_sec": pump_dur, "fan_pwm": int(raw_fan), "decision": dec}"""

# ================= 3. CÂU 3.1b: FIRMWARE ESP32-S3 (SmartChili_Firmware.ino) =================
code_31b_7 = """// 7. Khai báo thư viện và chân kết nối phần cứng ESP32-S3
#include <Wire.h>
#include <Adafruit_AHTX0.h>
#include <BH1750.h>
#include <Adafruit_SSD1306.h>
#include <esp_task_wdt.h>

// 1. Cảm biến độ ẩm đất Analog
#define SOIL_PIN 18

// 2. Giao tiếp I2C chung (AHT20 nhiệt ẩm, BH1750 ánh sáng, SSD1306 OLED)
#define I2C_SDA 8
#define I2C_SCL 9

// 3. Driver L298N Kênh A: Quạt làm mát DC 5V (PWM ENA)
#define FAN_ENA 17   // Điều xung tốc độ quạt PWM
#define FAN_IN1 16
#define FAN_IN2 15

// 4. Driver L298N Kênh B: Máy bơm nước tưới DC 5V (PWM ENB)
#define PUMP_ENB 4   // Điều xung tốc độ bơm PWM
#define PUMP_IN3 6
#define PUMP_IN4 5

// 5. Nút dừng khẩn cấp E-STOP (BOOT Pin 0)
#define BTN_EMERGENCY 0"""

code_31b_8 = """// 8. Khai báo máy trạng thái hữu hạn FSM (Finite State Machine)
enum SystemState {
    STATE_INIT,      // Khởi tạo phần cứng và kiểm tra ngoại vi
    STATE_MONITOR,   // Giám sát cảm biến, phân tích môi trường
    STATE_WATERING,  // Đang kích hoạt bơm tưới có giới hạn thời gian
    STATE_SOAKING,   // Dừng bơm chờ nước thẩm thấu vào đất
    STATE_FAILSAFE   // Trạng thái khóa an toàn khi có lỗi phần cứng / E-STOP
};

SystemState currentState = STATE_INIT;

// Cấu hình an toàn và thời gian hoạt động FSM
const unsigned long MAX_PUMP_TIME = 8000;    // Dead-man Timer: Tưới tối đa 8s
const unsigned long SOAKING_TIME  = 15000;   // Chờ đất ngấm nước: 15s
const unsigned long PUMP_COOLDOWN = 20000;   // Anti-chatter: Nghỉ tối thiểu 20s
const int ADC_FAULT_LOW  = 200;              // Ngưỡng chập mạch cảm biến đất
const int ADC_FAULT_HIGH = 4050;             // Ngưỡng đứt dây cảm biến đất"""

code_31b_9 = """// 9. Hàm đọc cảm biến đất với bộ lọc trung vị (Median Filter) khử nhiễu gai
float readFilteredSoilMoisture() {
    const int NUM_SAMPLES = 7;
    int samples[NUM_SAMPLES];

    // Thu thập 7 mẫu liên tiếp cách nhau 5ms
    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = analogRead(SOIL_PIN);
        delay(5);
    }

    // Sắp xếp nổi bọt (Bubble Sort) để lấy giá trị trung vị
    for (int i = 0; i < NUM_SAMPLES - 1; i++) {
        for (int j = i + 1; j < NUM_SAMPLES; j++) {
            if (samples[i] > samples[j]) {
                int temp = samples[i];
                samples[i] = samples[j];
                samples[j] = temp;
            }
        }
    }
    rawSoilADC = samples[NUM_SAMPLES / 2];

    // Kiểm tra ngoại lệ Sensor Fault: đứt dây hoặc chập mạch cảm biến (SC01)
    if (rawSoilADC < ADC_FAULT_LOW || rawSoilADC > ADC_FAULT_HIGH) {
        currentState = STATE_FAILSAFE;
        failsafeReason = "Sensor Fault: Dut/chap cam bien do am dat!";
        return -1.0;
    }

    // Chuyển đổi tuyến tính thành phần trăm độ ẩm [0% - 100%]
    float pct = (float)(ADC_AIR - rawSoilADC) / (float)(ADC_AIR - ADC_WATER) * 100.0;
    return constrain(pct, 0.0, 100.0);
}"""

code_31b_10 = """// 10. Hàm setup() khởi tạo hệ thống ngoại vi và chuyển trạng thái MONITOR
void setup() {
    Serial.begin(115200);

    // Cấu hình chân điều khiển Driver L298N và nút dừng khẩn cấp
    pinMode(FAN_IN1, OUTPUT);  pinMode(FAN_IN2, OUTPUT);
    pinMode(PUMP_IN3, OUTPUT); pinMode(PUMP_IN4, OUTPUT);
    pinMode(BTN_EMERGENCY, INPUT_PULLUP);

    digitalWrite(PUMP_IN3, LOW); digitalWrite(PUMP_IN4, LOW);
    digitalWrite(FAN_IN1, LOW);  digitalWrite(FAN_IN2, LOW);

    // Khởi tạo I2C, Màn hình OLED SSD1306, Cảm biến AHT20 & BH1750
    Wire.begin(I2C_SDA, I2C_SCL);
    display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    display.clearDisplay();
    display.setTextSize(1); display.setTextColor(WHITE);
    display.setCursor(0, 10);
    display.println("SMART CHILI AI S3");
    display.println("Khoi tao he thong...");
    display.display();

    aht.begin();
    lightMeter.begin();

    // Cấu hình Hardware Watchdog 10s bảo vệ chống treo vi điều khiển
    esp_task_wdt_init(10, true);
    esp_task_wdt_add(NULL);

    // Chuyển sang trạng thái giám sát chính MONITOR
    currentState = STATE_MONITOR;
    Serial.println("[FSM] Khoi tao hoan tat -> Chuyen sang trang thai MONITOR");
}"""

code_31b_11 = """// 11. Xử lý trạng thái WATERING (Bơm có Dead-man timer) và SOAKING (Đất ngấm)
void handleStateWatering() {
    // Dead-man Timer: Giới hạn bơm tối đa 8s chống tràn ngập
    unsigned long duration = millis() - stateStartTime;
    if (duration >= MAX_PUMP_TIME || soilPercent >= 70.0) {
        // Tắt bơm và kích hoạt trạng thái SOAKING
        digitalWrite(PUMP_IN3, LOW);
        digitalWrite(PUMP_IN4, LOW);
        ledcWrite(PUMP_ENB, 0);

        stateStartTime = millis();
        currentState = STATE_SOAKING;
        Serial.println("[FSM] Ngat bom -> Chuyen sang SOAKING (cho nuoc ngam)");
        return;
    }

    // Kích hoạt bơm nước với PWM 220
    digitalWrite(PUMP_IN3, HIGH);
    digitalWrite(PUMP_IN4, LOW);
    ledcWrite(PUMP_ENB, 220);
}

void handleStateSoaking() {
    // Đảm bảo bơm luôn tắt trong thời gian đất ngấm
    digitalWrite(PUMP_IN3, LOW);
    digitalWrite(PUMP_IN4, LOW);

    // Anti-chatter: Chờ đủ thời gian ngấm 15s trước khi quay về MONITOR
    if (millis() - stateStartTime >= SOAKING_TIME) {
        lastWateringTime = millis();
        currentState = STATE_MONITOR;
        Serial.println("[FSM] Ket thuc SOAKING -> Quay ve trang thai MONITOR");
    }
}"""

code_31b_12 = """// 12. Trạng thái FAILSAFE: Khóa an toàn, tắt toàn bộ bơm/quạt và cảnh báo
void handleStateFailsafe() {
    // 1. Tắt toàn bộ tải công suất (Bơm nước & Quạt làm mát) ngay lập tức
    digitalWrite(PUMP_IN3, LOW);
    digitalWrite(PUMP_IN4, LOW);
    ledcWrite(PUMP_ENB, 0);

    digitalWrite(FAN_IN1, LOW);
    digitalWrite(FAN_IN2, LOW);
    ledcWrite(FAN_ENA, 0);

    // 2. Hiển thị cảnh báo lỗi nghiêm trọng lên màn hình OLED
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("!! FAILSAFE MODE !!");
    display.println("TAT TOAN BO BOM/QUAT");
    display.println(failsafeReason);
    display.println("Nhan RESET de phuc hoi");
    display.display();

    // 3. Xuất log cảnh báo chu kỳ 2s qua cổng Serial
    static unsigned long lastLog = 0;
    if (millis() - lastLog > 2000) {
        Serial.printf("[FAILSAFE] He thong dung khan cap! Ly do: %s\\n", failsafeReason.c_str());
        lastLog = millis();
    }
}"""

code_31c_13 = """// Câu 3.1c – CÁC CƠ CHẾ AN TOÀN VÀ XỬ LÝ NGOẠI LỆ HỆ THỐNG
// 1. Dead-man Timer: Tự động ngắt tưới nếu quá 8 giây
if (wateringDuration >= MAX_PUMP_TIME) { stopPump(); enterSoaking(); }

// 2. Sensor Fault (SC01): Đứt dây hoặc chập mạch cảm biến đất
if (rawADC < 200 || rawADC > 4050) { enterFailsafe("Sensor Fault"); }

// 3. Hardware Watchdog Timer: Chống treo vi điều khiển
esp_task_wdt_init(10, true); esp_task_wdt_reset();

// 4. Anti-chatter (Cooldown): Nghỉ tối thiểu 20s giữa 2 lần tưới
if (millis() - lastWateringTime < PUMP_COOLDOWN) { canWater = false; }

// 5. Offline Fallback (SC02): Tự động tưới độc lập nếu mất kết nối PC
if (millis() - lastPcCommTime > PC_TIMEOUT) { runLocalSafetyFSM(); }"""

py_lex = PythonLexer()
cpp_lex = CppLexer()

# Tạo 12 ảnh code chính + 1 ảnh code an toàn
render_code_card(code_31a_1, py_lex, "ai_vision_service.py", "Ảnh 1: Khởi tạo MobileNetV3 & 4 Lớp Bệnh Học", os.path.join(OUTPUT_DIR, "anh_01_mobilenet_init.png"))
render_code_card(code_31a_2, py_lex, "ai_vision_service.py", "Ảnh 2: Tiền Xử Lý Ảnh & Kiểm Tra Chất Lượng Quang Học", os.path.join(OUTPUT_DIR, "anh_02_preprocess_quality.png"))
render_code_card(code_31a_3, py_lex, "ai_vision_service.py", "Ảnh 3: Hàm predict() Suy Luận & Softmax Confidence", os.path.join(OUTPUT_DIR, "anh_03_predict_softmax.png"))

render_code_card(code_31b_4, py_lex, "fuzzy_engine.py", "Ảnh 4: Khai Báo Biến Vào/Ra (Soil, Temp, Rain, Wilt, Pump, Fan)", os.path.join(OUTPUT_DIR, "anh_04_fuzzy_variables.png"))
render_code_card(code_31b_5, py_lex, "fuzzy_engine.py", "Ảnh 5: Thiết Kế Hàm Thuộc Membership Functions", os.path.join(OUTPUT_DIR, "anh_05_fuzzy_membership.png"))
render_code_card(code_31b_6, py_lex, "fuzzy_engine.py", "Ảnh 6: Tập Luật Rule Base & Hàm evaluate() Giải Mờ", os.path.join(OUTPUT_DIR, "anh_06_fuzzy_rules_eval.png"))

render_code_card(code_31b_7, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 7: Khai Báo Thư Viện & Sơ Đồ Chân Kết Nối ESP32-S3", os.path.join(OUTPUT_DIR, "anh_07_esp32_pins.png"))
render_code_card(code_31b_8, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 8: Khai Báo Máy Trạng Thái Hữu Hạn FSM", os.path.join(OUTPUT_DIR, "anh_08_esp32_fsm_enum.png"))
render_code_card(code_31b_9, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 9: Đọc Cảm Biến Đất Với Median Filter Khử Nhiễu", os.path.join(OUTPUT_DIR, "anh_09_esp32_median_filter.png"))
render_code_card(code_31b_10, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 10: Hàm setup() Khởi Tạo & Chuyển Trạng Thái MONITOR", os.path.join(OUTPUT_DIR, "anh_10_esp32_setup_monitor.png"))
render_code_card(code_31b_11, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 11: Trạng Thái WATERING & Chờ Ngấm SOAKING", os.path.join(OUTPUT_DIR, "anh_11_esp32_watering_soaking.png"))
render_code_card(code_31b_12, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 12: Trạng Thái Khóa An Toàn FAILSAFE & Dừng Khẩn Cấp", os.path.join(OUTPUT_DIR, "anh_12_esp32_failsafe.png"))

render_code_card(code_31c_13, cpp_lex, "SmartChili_Firmware.ino", "Ảnh 13: Các Cơ Chế An Toàn (Dead-man, Watchdog, Anti-chatter)", os.path.join(OUTPUT_DIR, "anh_13_safety_exceptions.png"))

print("Đã tạo hoàn tất 13 ảnh code minh chứng!")
