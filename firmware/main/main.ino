// ======================================================================================
// ĐỀ TÀI: HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
// FILE: main.ino
// VAI TRÒ: FIRMWARE VI ĐIỀU KHIỂN ESP32-S3 (TẦNG PHẦN CỨNG SENSE & ACT)
// CHỨC NĂNG:
//   1. Đọc dữ liệu từ cảm biến độ ẩm đất Analog (GPIO 7 hoặc GPIO 18).
//   2. Đọc cảm biến nhiệt độ & độ ẩm không khí AHT20 + cảm biến ánh sáng BH1750 qua bus I2C (SDA: 8, SCL: 9).
//   3. Hiển thị thông số và trạng thái Bơm/Quạt lên màn hình OLED SSD1306.
//   4. Điều khiển 2 kênh động cơ qua Driver cầu H L298N (Quạt: Kênh A, Bơm: Kênh B).
//   5. Khóa liên động an toàn phần cứng (Safety Interlocks) & ngắt khẩn cấp E-STOP qua nút BOOT (GPIO 0).
//   6. Giao tiếp UART 115200 truyền nhận JSON với máy tính/Python Server.
// ======================================================================================

#include <Wire.h>
#include <Adafruit_AHTX0.h>
#include <BH1750.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>

// ================= CHỌN MÔ HÌNH PHẦN CỨNG (THEO SƠ ĐỒ ĐẤU NỐI) =================
// Chọn mô hình bạn đang nạp code:
// #define MODEL_1  // Mô hình 1: Cảm biến đất ở GPIO 7
#define MODEL_2  // Mô hình 2: Cảm biến đất ở GPIO 18

#ifdef MODEL_1
  #define SOIL_PIN 7
  const char* MODEL_NAME = "MH1";
#else
  #define SOIL_PIN 18
  const char* MODEL_NAME = "MH2";
#endif

// ================= SƠ ĐỒ CHÂN ESP32-S3 CHUẨN 100% THEO TÀI LIỆU =================
// 1. Bus I2C chung 4 dây (AHT20 - 0x38, BH1750 - 0x23, OLED SSD1306 - 0x3C)
#define I2C_SDA 8
#define I2C_SCL 9

// 2. Driver L298N - Kênh A (Quạt DC 5V làm mát / hong khô)
#define FAN_ENA 17  // Chân PWM tốc độ quạt (tháo jumper ENA)
#define FAN_IN1 16  // Hướng quạt
#define FAN_IN2 15  // Hướng quạt

// 3. Driver L298N - Kênh B (Máy bơm DC 5V tưới nước)
#define PUMP_ENB 4  // Chân PWM tốc độ bơm (tháo jumper ENB)
#define PUMP_IN3 6  // Hướng bơm
#define PUMP_IN4 5  // Hướng bơm

// 4. Nút BOOT trên board dừng khẩn cấp (ALL OFF)
#define BTN_EMERGENCY 0

// Cấu hình PWM cho ESP32 Core v3.x (1 kHz, 8-bit)
#define PWM_FREQ 1000
#define PWM_RES 8

// ================= NGƯỠNG HIỆU CHUẨN & AN TOÀN =================
const int ADC_AIR = 3400;        // Đất khô ngoài không khí (0%)
const int ADC_WATER = 1200;      // Nhúng nước (100%)
const int ADC_FAULT_LOW = 200;   // Ngưỡng đứt/chập cảm biến độ ẩm đất (SC01)
const int ADC_FAULT_HIGH = 4050;

const float SOIL_CRITICAL_DRY = 20.0; // Khô nguy cấp -> ưu tiên tưới
const float SOIL_SAFE_WET = 75.0;     // Đủ ẩm -> khóa tưới tuyệt đối
const float TEMP_FAN_ON = 33.0;
const float TEMP_FAN_OFF = 31.0;

const unsigned long MAX_PUMP_DURATION = 10000; // Giới hạn tưới tối đa 10s/lần
const unsigned long PUMP_COOLDOWN = 20000;     // Nghỉ tối thiểu 20s giữa 2 lần bơm
const unsigned long PC_TIMEOUT = 15000;        // Mất kết nối PC > 15s -> Offline (SC02)

// Khởi tạo các đối tượng cảm biến & màn hình
Adafruit_AHTX0 aht;
BH1750 lightMeter;
Adafruit_SSD1306 display(128, 64, &Wire, -1);

bool emergencyStop = false;
bool isOfflineMode = true;
bool sensorFault = false;
String systemMode = "AUTO";

bool pumpState = false;
unsigned long pumpStartTime = 0;
unsigned long lastPumpStopTime = 0;
bool fanState = false;

unsigned long lastPcCommTime = 0;
unsigned long lastSensorReadTime = 0;
unsigned long lastOledUpdateTime = 0;

float airTemp = 0.0, airHum = 0.0, lux = 0.0, soilPercent = 0.0;
int soilRaw = 0;

// Điều khiển Máy Bơm (Kênh B - L298N)
void setPump(bool state, int speedPWM) {
  if (emergencyStop || sensorFault || soilPercent >= SOIL_SAFE_WET) {
    state = false;
  }

  if (state) {
    if (!pumpState && (millis() - lastPumpStopTime < PUMP_COOLDOWN) && lastPumpStopTime != 0) {
      return;
    }
    if (!pumpState) pumpStartTime = millis();
    digitalWrite(PUMP_IN3, HIGH);
    digitalWrite(PUMP_IN4, LOW);
    ledcWrite(PUMP_ENB, constrain(speedPWM, 0, 255));
    pumpState = true;
  } else {
    if (pumpState) lastPumpStopTime = millis();
    digitalWrite(PUMP_IN3, LOW);
    digitalWrite(PUMP_IN4, LOW);
    ledcWrite(PUMP_ENB, 0);
    pumpState = false;
  }
}

// Điều khiển Quạt Làm Mát (Kênh A - L298N)
void setFan(bool state, int speedPWM) {
  if (emergencyStop) state = false;

  if (state) {
    digitalWrite(FAN_IN1, HIGH);
    digitalWrite(FAN_IN2, LOW);
    ledcWrite(FAN_ENA, constrain(speedPWM, 0, 255));
    fanState = true;
  } else {
    digitalWrite(FAN_IN1, LOW);
    digitalWrite(FAN_IN2, LOW);
    ledcWrite(FAN_ENA, 0);
    fanState = false;
  }
}

// Đọc toàn bộ cảm biến môi trường
void readSensors() {
  sensors_event_t humidity, temp;
  if (aht.getEvent(&humidity, &temp)) {
    airTemp = temp.temperature;
    airHum = humidity.relative_humidity;
  }

  if (lightMeter.measurementReady()) {
    lux = lightMeter.readLightLevel();
  }

  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(SOIL_PIN);
    delay(2);
  }
  soilRaw = sum / 10;

  // Kiểm tra lỗi SC01 (đứt dây hoặc chập cảm biến)
  if (soilRaw < ADC_FAULT_LOW || soilRaw > ADC_FAULT_HIGH) {
    sensorFault = true;
    setPump(false, 0);
  } else {
    sensorFault = false;
    float pct = map(soilRaw, ADC_AIR, ADC_WATER, 0, 1000) / 10.0;
    soilPercent = constrain(pct, 0.0, 100.0);
  }
}

// Logic bảo vệ an toàn & tự trị khi mất kết nối máy tính
void runSafetyAndLocalLogic() {
  // Nút BOOT nhấn = Dừng khẩn cấp ALL OFF (chỉ kích hoạt sau khi board đã khởi động ổn định 1.5s để tránh kích hoạt lúc nạp ROM)
  if (millis() > 1500 && digitalRead(BTN_EMERGENCY) == LOW) {
    delay(50);
    if (digitalRead(BTN_EMERGENCY) == LOW) {
      emergencyStop = true;
      setPump(false, 0);
      setFan(false, 0);
    }
  }

  // Khóa bơm nếu chạy vượt quá 10 giây
  if (pumpState && (millis() - pumpStartTime >= MAX_PUMP_DURATION)) {
    setPump(false, 0);
  }

  // Phát hiện mất kết nối PC quá 15 giây (SC02)
  if (millis() - lastPcCommTime > PC_TIMEOUT) {
    isOfflineMode = true;
  }

  // Tự trị Offline an toàn
  if (isOfflineMode && systemMode == "AUTO" && !emergencyStop) {
    if (!sensorFault && soilPercent < SOIL_CRITICAL_DRY) {
      setPump(true, 200);
    } else if (soilPercent >= 45.0) {
      setPump(false, 0);
    }

    if (airTemp >= TEMP_FAN_ON) setFan(true, 220);
    else if (airTemp <= TEMP_FAN_OFF) setFan(false, 0);
  }
}

// Giao tiếp UART với chương trình Python (115200 baud)
void handleSerialComm() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    JsonDocument doc;
    if (!deserializeJson(doc, input)) {
      lastPcCommTime = millis();
      isOfflineMode = false;

      if (doc["reset_emergency"] == true) emergencyStop = false;
      if (doc["estop"] == true) {
        emergencyStop = true;
        setPump(false, 0);
        setFan(false, 0);
      }
      if (doc["mode"].is<const char*>()) systemMode = doc["mode"].as<String>();

      if (!emergencyStop) {
        if (doc["pump"].is<bool>()) setPump(doc["pump"], doc["pump_speed"] | 200);
        if (doc["fan"].is<bool>()) setFan(doc["fan"], doc["fan_speed"] | 200);
      }
    }
  }

  // Gửi telemetry lên máy tính mỗi 2 giây
  if (millis() - lastSensorReadTime >= 2000) {
    lastSensorReadTime = millis();
    readSensors();

    JsonDocument outDoc;
    outDoc["model"] = MODEL_NAME;
    outDoc["soil_raw"] = soilRaw;
    outDoc["soil_pct"] = round(soilPercent * 10) / 10.0;
    outDoc["temp"] = round(airTemp * 10) / 10.0;
    outDoc["hum"] = round(airHum * 10) / 10.0;
    outDoc["lux"] = round(lux);
    outDoc["pump"] = pumpState;
    outDoc["fan"] = fanState;
    outDoc["fault"] = sensorFault;
    outDoc["estop"] = emergencyStop;
    outDoc["offline"] = isOfflineMode;
    outDoc["mode"] = systemMode;

    serializeJson(outDoc, Serial);
    Serial.println();
  }
}

// Cập nhật màn hình OLED SSD1306
void updateOLED() {
  if (millis() - lastOledUpdateTime < 500) return;
  lastOledUpdateTime = millis();

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.printf("CHILI AI [%s] %s", MODEL_NAME, isOfflineMode ? "OFF" : "ON");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  if (emergencyStop) {
    display.setCursor(0, 22);
    display.setTextSize(2);
    display.println("ALL OFF!");
    display.setTextSize(1);
    display.println("Emergency Stopped");
    display.display();
    return;
  }

  if (sensorFault) {
    display.setCursor(0, 20);
    display.println("LOI CAM BIEN DAT!");
    display.printf("ADC Raw: %d\n", soilRaw);
    display.println("Khoa bom an toan");
    display.display();
    return;
  }

  display.setCursor(0, 15);
  display.printf("Dat: %.1f%% (%d)\n", soilPercent, soilRaw);
  display.setCursor(0, 27);
  display.printf("KK : %.1fC | %.0f%%\n", airTemp, airHum);
  display.setCursor(0, 39);
  display.printf("AS : %.0f lux\n", lux);

  display.drawLine(0, 50, 128, 50, SSD1306_WHITE);
  display.setCursor(0, 54);
  display.printf("BOM:%s | QUAT:%s", pumpState ? "ON " : "OFF", fanState ? "ON" : "OFF");
  display.display();
}

void setup() {
  Serial.begin(115200);
  pinMode(BTN_EMERGENCY, INPUT_PULLUP);
  pinMode(SOIL_PIN, INPUT);
  pinMode(FAN_IN1, OUTPUT);
  pinMode(FAN_IN2, OUTPUT);
  pinMode(PUMP_IN3, OUTPUT);
  pinMode(PUMP_IN4, OUTPUT);

  // Gắn kênh PWM cho quạt và bơm (ESP32 Arduino Core v3.x)
  ledcAttach(FAN_ENA, PWM_FREQ, PWM_RES);
  ledcAttach(PUMP_ENB, PWM_FREQ, PWM_RES);
  setPump(false, 0);
  setFan(false, 0);

  // Khởi động giao tiếp I2C và các cảm biến
  Wire.begin(I2C_SDA, I2C_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  aht.begin();
  lightMeter.begin();
}

void loop() {
  handleSerialComm();
  runSafetyAndLocalLogic();
  updateOLED();
}
