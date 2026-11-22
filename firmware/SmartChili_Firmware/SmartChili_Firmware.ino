/*
 * SMART CHILI AI - ESP32-S3 FIRMWARE
 * Hệ thống IoT Vườn Ớt Thông Minh - Câu 3.1b & Câu 3.1c
 */

#include <Wire.h>
#include <Adafruit_AHTX0.h>
#include <BH1750.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include <esp_task_wdt.h>

// ================= 7. KHAI BÁO THƯ VIỆN VÀ CHÂN KẾT NỐI (PIN DEFINITIONS) =================
// 1. Cảm biến độ ẩm đất Analog (GPIO 18 / GPIO 7)
#define SOIL_PIN 18

// 2. Giao tiếp I2C chung (AHT20 nhiệt ẩm, BH1750 cường độ sáng, SSD1306 OLED)
#define I2C_SDA 8
#define I2C_SCL 9

// 3. Driver Motor L298N - Kênh A: Quạt làm mát / lưu thông không khí (Fan DC 5V)
#define FAN_ENA 17   // Chân điều xung PWM tốc độ quạt
#define FAN_IN1 16   // Hướng quay quạt
#define FAN_IN2 15   // Hướng quay quạt

// 4. Driver Motor L298N - Kênh B: Máy bơm nước tưới (Pump DC 5V)
#define PUMP_ENB 4   // Chân điều xung PWM tốc độ bơm
#define PUMP_IN3 6   // Hướng bơm
#define PUMP_IN4 5   // Hướng bơm

// 5. Nút dừng khẩn cấp vật lý E-STOP (BOOT Button)
#define BTN_EMERGENCY 0

// ================= CẤU HÌNH AN TOÀN & NGOẠI LỆ (SAFETY CONFIG) =================
const int ADC_AIR   = 3400;       // Đất khô 0%
const int ADC_WATER = 1200;       // Nhúng nước 100%
const int ADC_FAULT_LOW  = 200;   // Lỗi chập mạch cảm biến đất (SC01)
const int ADC_FAULT_HIGH = 4050;  // Lỗi hở mạch / đứt dây cảm biến đất

const unsigned long MAX_PUMP_TIME = 8000;    // Dead-man Timer: Giới hạn tưới tối đa 8s
const unsigned long SOAKING_TIME  = 15000;   // Thời gian chờ đất ngấm nước: 15s
const unsigned long PUMP_COOLDOWN = 20000;   // Anti-chatter: Nghỉ tối thiểu 20s giữa các lần tưới
const unsigned long WDT_TIMEOUT   = 10;      // Hardware Watchdog Timeout: 10 giây
const unsigned long PC_TIMEOUT    = 15000;   // Timeout mất kết nối máy chủ AI: 15s

// ================= 8. KHAI BÁO MÁY TRẠNG THÁI HỮU HẠN FSM =================
enum SystemState {
    STATE_INIT,      // Khởi tạo phần cứng và kiểm tra ngoại vi
    STATE_MONITOR,   // Giám sát cảm biến, phân tích môi trường
    STATE_WATERING,  // Đang kích hoạt bơm tưới có kiểm soát thời gian
    STATE_SOAKING,   // Dừng bơm chờ nước ngấm vào đất (chống ngập úng)
    STATE_FAILSAFE   // Trạng thái khóa an toàn khi có lỗi phần cứng / khẩn cấp
};

SystemState currentState = STATE_INIT;

// Khởi tạo các đối tượng ngoại vi
Adafruit_AHTX0 aht;
BH1750 lightMeter;
Adafruit_SSD1306 display(128, 64, &Wire, -1);

// Biến lưu trữ trạng thái hoạt động
float soilPercent = 0.0;
float airTemp = 0.0;
float airHum = 0.0;
float lux = 0.0;
int rawSoilADC = 0;

unsigned long stateStartTime = 0;
unsigned long lastWateringTime = 0;
unsigned long lastCommTime = 0;
bool isFailsafeTriggered = false;
String failsafeReason = "";

// ================= 9. HÀM ĐỌC CẢM BIẾN ĐẤT VỚI BỘ LỌC TRUNG VỊ (MEDIAN FILTER) =================
float readFilteredSoilMoisture() {
    const int NUM_SAMPLES = 7;
    int samples[NUM_SAMPLES];

    // Thu thập 7 mẫu liên tiếp cách nhau 5ms
    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = analogRead(SOIL_PIN);
        delay(5);
    }

    // Sắp xếp nổi bọt (Bubble Sort) để lấy giá trị trung vị lọc nhiễu gai điện áp
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

    // Kiểm tra Sensor Fault: đứt dây hoặc chập mạch cảm biến
    if (rawSoilADC < ADC_FAULT_LOW || rawSoilADC > ADC_FAULT_HIGH) {
        currentState = STATE_FAILSAFE;
        failsafeReason = "Sensor Fault (SC01): Đứt/chập cảm biến đất!";
        return -1.0;
    }

    // Chuyển đổi tuyến tính thành phần trăm độ ẩm đất [0% - 100%]
    float pct = (float)(ADC_AIR - rawSoilADC) / (float)(ADC_AIR - ADC_WATER) * 100.0;
    return constrain(pct, 0.0, 100.0);
}

// ================= 10. HÀM SETUP() KHỞI TẠO HỆ THỐNG VÀ CHUYỂN MONITOR =================
void setup() {
    Serial.begin(115200);

    // Cấu hình chân điều khiển Driver L298N và nút dừng khẩn cấp
    pinMode(FAN_IN1, OUTPUT);
    pinMode(FAN_IN2, OUTPUT);
    pinMode(PUMP_IN3, OUTPUT);
    pinMode(PUMP_IN4, OUTPUT);
    pinMode(BTN_EMERGENCY, INPUT_PULLUP);

    digitalWrite(FAN_IN1, LOW);
    digitalWrite(FAN_IN2, LOW);
    digitalWrite(PUMP_IN3, LOW);
    digitalWrite(PUMP_IN4, LOW);

    // Giao tiếp I2C & OLED SSD1306
    Wire.begin(I2C_SDA, I2C_SCL);
    display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 10);
    display.println("SMART CHILI AI S3");
    display.println("Khoi tao he thong...");
    display.display();

    // Khởi tạo các cảm biến
    aht.begin();
    lightMeter.begin();

    // Cấu hình Hardware Watchdog 10 giây chống treo vi điều khiển
    esp_task_wdt_init(WDT_TIMEOUT, true);
    esp_task_wdt_add(NULL);

    lastCommTime = millis();
    // Chuyển sang trạng thái giám sát chính MONITOR
    currentState = STATE_MONITOR;
    Serial.println("[FSM] Khoi tao hoan tat -> Chuyen sang trang thai MONITOR");
}

// ================= 11. TRẠNG THÁI WATERING VÀ SOAKING =================
void handleStateWatering() {
    // Dead-man Timer: Giới hạn thời gian bơm tối đa chống tràn nước
    unsigned long wateringDuration = millis() - stateStartTime;
    if (wateringDuration >= MAX_PUMP_TIME || soilPercent >= 70.0) {
        // Tắt bơm và chuyển sang trạng thái chờ đất ngấm SOAKING
        digitalWrite(PUMP_IN3, LOW);
        digitalWrite(PUMP_IN4, LOW);
        ledcWrite(PUMP_ENB, 0);

        stateStartTime = millis();
        currentState = STATE_SOAKING;
        Serial.println("[FSM] Ngat bom -> Chuyen sang trang thai SOAKING (cho nuoc ngam)");
        return;
    }

    // Kích hoạt bơm nước (PWM 220)
    digitalWrite(PUMP_IN3, HIGH);
    digitalWrite(PUMP_IN4, LOW);
    ledcWrite(PUMP_ENB, 220);
}

void handleStateSoaking() {
    // Đảm bảo bơm luôn tắt trong thời gian đất thẩm thấu
    digitalWrite(PUMP_IN3, LOW);
    digitalWrite(PUMP_IN4, LOW);

    // Anti-chatter: Chờ đủ thời gian ngấm nước 15s trước khi quay lại MONITOR
    if (millis() - stateStartTime >= SOAKING_TIME) {
        lastWateringTime = millis();
        currentState = STATE_MONITOR;
        Serial.println("[FSM] Ket thuc SOAKING -> Quay ve trang thai MONITOR");
    }
}

// ================= 12. TRẠNG THÁI FAILSAFE (KHÓA AN TOÀN VÀ XỬ LÝ NGOẠI LỆ) =================
void handleStateFailsafe() {
    // 1. Tắt toàn bộ tải động cơ (Bơm và Quạt) ngay lập tức
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

    // 3. Xuất log cảnh báo định kỳ qua UART
    static unsigned long lastLog = 0;
    if (millis() - lastLog > 2000) {
        Serial.printf("[FAILSAFE] He thong dung khan cap! Ly do: %s\n", failsafeReason.c_str());
        lastLog = millis();
    }
}

// ================= CHU TRÌNH ĐIỀU KHIỂN CHÍNH (MAIN LOOP) =================
void loop() {
    esp_task_wdt_reset(); // Reset Hardware Watchdog

    // Kiểm tra nút dừng khẩn cấp phần cứng
    if (digitalRead(BTN_EMERGENCY) == LOW) {
        currentState = STATE_FAILSAFE;
        failsafeReason = "Nut E-STOP vat ly duoc bam!";
    }

    // Đọc cảm biến môi trường
    soilPercent = readFilteredSoilMoisture();
    sensors_event_t humidity, temp;
    aht.getEvent(&humidity, &temp);
    airTemp = temp.temperature;
    airHum = humidity.relative_humidity;
    lux = lightMeter.readLightLevel();

    // Máy trạng thái hữu hạn FSM
    switch (currentState) {
        case STATE_MONITOR: {
            // Anti-chatter: Không tưới nếu chưa qua thời gian hồi nguội PUMP_COOLDOWN
            bool canWater = (millis() - lastWateringTime >= PUMP_COOLDOWN) || (lastWateringTime == 0);
            if (soilPercent >= 0 && soilPercent < 30.0 && canWater) {
                stateStartTime = millis();
                currentState = STATE_WATERING;
                Serial.println("[FSM] Dat kho -> Chuyen sang trang thai WATERING");
            }
            break;
        }
        case STATE_WATERING:
            handleStateWatering();
            break;
        case STATE_SOAKING:
            handleStateSoaking();
            break;
        case STATE_FAILSAFE:
            handleStateFailsafe();
            break;
        default:
            currentState = STATE_MONITOR;
            break;
    }

    delay(100);
}
