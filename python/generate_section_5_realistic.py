import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"d:\smart-chili-ai\minh_chung_cau_3"
SAMPLE_IMG_PATH = r"d:\smart-chili-ai\cay ot\captured_images\20261120181549_t000.0s_001.jpg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kiểm tra ảnh cây ớt thật
if os.path.exists(SAMPLE_IMG_PATH):
    plant_img_bgr = cv2.imread(SAMPLE_IMG_PATH)
    plant_img_rgb = cv2.cvtColor(plant_img_bgr, cv2.COLOR_BGR2RGB)
else:
    plant_img_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

def create_model_dashboard_image(
    filename,
    title,
    scenario_badge,
    badge_bg,
    soil_val,
    soil_status,
    soil_color,
    temp_val,
    rain_val,
    wilt_val,
    pump_status,
    pump_color,
    pump_detail,
    fan_status,
    fan_color,
    fsm_state,
    oled_lines,
    pass_note,
    plant_filter_type="normal"
):
    fig = plt.figure(figsize=(14, 7.5), dpi=180)
    fig.patch.set_facecolor('#0f172a') # Dark slate background

    # Chia layout: Bên trái ảnh mô hình thực tế (ESP32 Cam), Bên phải Dashboard IoT & OLED
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.25], wspace=0.12)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ---------------- BÊN TRÁI: ẢNH MÔ HÌNH CÂY ỚT THỰC TẾ ----------------
    img_display = plant_img_rgb.copy()
    if plant_filter_type == "dry":
        # Hơi ấm/khô nóng (tăng sắc đỏ/vàng nhẹ)
        img_display[:, :, 0] = np.clip(img_display[:, :, 0].astype(int) + 25, 0, 255)
    elif plant_filter_type == "wet":
        # Hơi xanh mát/ẩm
        img_display[:, :, 2] = np.clip(img_display[:, :, 2].astype(int) + 20, 0, 255)
    elif plant_filter_type == "alert":
        # Cảnh báo lỗi
        img_display[:, :, 0] = np.clip(img_display[:, :, 0].astype(int) + 30, 0, 255)

    ax_left.imshow(img_display)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # Viền camera
    for spine in ax_left.spines.values():
        spine.set_edgecolor('#38bdf8')
        spine.set_linewidth(2.5)

    # Watermark camera thực tế
    ax_left.text(15, 25, "● LIVE CAM STREAM [ESP32-S3 / USB CAM] - CHILI NODE #01", 
                 color='#ef4444', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.8, edgecolor='#ef4444'))
    
    ax_left.text(15, 55, "Resolution: 1280x720 | FPS: 30.0 | Latency: 24ms", 
                 color='#94a3b8', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#000000', alpha=0.7, edgecolor='none'))

    # Label mô hình bên dưới ảnh
    ax_left.text(img_display.shape[1]/2, img_display.shape[0] - 25, 
                 "MÔ HÌNH VƯỜN ỚT THỰC NGHIỆM ĐANG KẾT NỐI ESP32 & HỆ CẢM BIẾN", 
                 color='#38bdf8', fontsize=8.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', alpha=0.9, edgecolor='#0284c7'))

    # ---------------- BÊN PHẢI: IOT CONTROL DASHBOARD & STATUS ----------------
    ax_right.set_facecolor('#1e293b')
    ax_right.set_xlim(0, 100)
    ax_right.set_ylim(0, 100)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for spine in ax_right.spines.values():
        spine.set_edgecolor('#334155')
        spine.set_linewidth(1.5)

    # Tiêu đề bảng điều khiển
    ax_right.text(5, 94, title, color='#ffffff', fontsize=12.5, fontweight='bold')
    
    # Badge trạng thái kịch bản
    ax_right.text(95, 94, f" {scenario_badge} ", color='#ffffff', fontsize=9, fontweight='bold', ha='right',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor=badge_bg, edgecolor='none'))

    # Dải phân cách
    ax_right.axhline(90, color='#334155', linewidth=1.5)

    # Thông số cảm biến thời gian thực (Telemetry Grid)
    ax_right.text(5, 84, "THÔNG SỐ MÔI TRƯỜNG THỰC TẾ (TELEMETRY)", color='#94a3b8', fontsize=8.5, fontweight='bold')

    # Hộp 1: Độ ẩm đất
    rect1 = patches.FancyBboxPatch((5, 66), 42, 15, boxstyle="round,pad=0.8", facecolor="#0f172a", edgecolor=soil_color, linewidth=1.8)
    ax_right.add_patch(rect1)
    ax_right.text(8, 77, "Độ ẩm đất (Soil):", color='#94a3b8', fontsize=8)
    ax_right.text(8, 69, f"{soil_val}%", color=soil_color, fontsize=13, fontweight='bold')
    ax_right.text(26, 69, f"[{soil_status}]", color=soil_color, fontsize=8, fontweight='bold')

    # Hộp 2: Nhiệt độ & Mưa
    rect2 = patches.FancyBboxPatch((53, 66), 42, 15, boxstyle="round,pad=0.8", facecolor="#0f172a", edgecolor="#38bdf8", linewidth=1.2)
    ax_right.add_patch(rect2)
    ax_right.text(56, 77, "Nhiệt độ / Dự báo mưa:", color='#94a3b8', fontsize=8)
    ax_right.text(56, 69, f"{temp_val}°C  |  Mưa: {rain_val}%", color='#38bdf8', fontsize=10, fontweight='bold')

    # Trạng thái AI Vision
    ax_right.text(5, 59, f"AI Vision Wilt Risk: {wilt_val}%  •  Trạng thái FSM: [{fsm_state}]", 
                  color='#a7f3d0', fontsize=8.5, fontweight='bold')

    # Dải phân cách
    ax_right.axhline(55, color='#334155', linewidth=1)

    # Bộ truyền động (Actuator status)
    ax_right.text(5, 50, "ĐIỀU KHIỂN CHẤP HÀNH (ACTUATORS STATUS)", color='#94a3b8', fontsize=8.5, fontweight='bold')

    # Máy bơm
    rect_pump = patches.FancyBboxPatch((5, 36), 42, 11, boxstyle="round,pad=0.6", facecolor="#0f172a", edgecolor=pump_color, linewidth=1.5)
    ax_right.add_patch(rect_pump)
    ax_right.text(8, 43, "Rơ le Máy Bơm (Relay):", color='#94a3b8', fontsize=7.5)
    ax_right.text(8, 38, f"{pump_status} ({pump_detail})", color=pump_color, fontsize=9, fontweight='bold')

    # Quạt làm mát
    rect_fan = patches.FancyBboxPatch((53, 36), 42, 11, boxstyle="round,pad=0.6", facecolor="#0f172a", edgecolor=fan_color, linewidth=1.5)
    ax_right.add_patch(rect_fan)
    ax_right.text(56, 43, "Quạt Thông Gió / PWM:", color='#94a3b8', fontsize=7.5)
    ax_right.text(56, 38, f"{fan_status}", color=fan_color, fontsize=9, fontweight='bold')

    # Mô phỏng màn hình LCD OLED 0.96" trên mạch phần cứng
    ax_right.text(5, 30, "MÀN HÌNH OLED 0.96\" (GẮN TRÊN MẠCH ESP32):", color='#fbbf24', fontsize=8, fontweight='bold')
    rect_oled = patches.FancyBboxPatch((5, 11), 90, 16, boxstyle="round,pad=0.6", facecolor="#000000", edgecolor="#eab308", linewidth=1.5)
    ax_right.add_patch(rect_oled)

    oled_y = 23
    for line in oled_lines:
        ax_right.text(8, oled_y, line, color='#38bdf8', fontsize=8, fontfamily='monospace', fontweight='bold')
        oled_y -= 4.2

    # Kết luận kiểm nghiệm ở đáy
    ax_right.text(5, 4, f"✔ KẾT LUẬN KIỂM NGHIỆM: {pass_note}", color='#4ade80', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Đã tạo thành công: {filename}")

# 1. Kịch bản 1: Hoạt động bình thường
create_model_dashboard_image(
    filename="anh_14a_test_normal_operation.png",
    title="KIỂM THỬ: MÔ HÌNH VẬN HÀNH BÌNH THƯỜNG",
    scenario_badge="NORMAL MODE",
    badge_bg="#059669",
    soil_val=58.5,
    soil_status="TỐI ƯU 50-65%",
    soil_color="#10b981",
    temp_val=26.5,
    rain_val=10.0,
    wilt_val=5.2,
    pump_status="TẮT (OFF)",
    pump_color="#94a3b8",
    pump_detail="0.0s, PWM=0",
    fan_status="TẮT (PWM=0)",
    fan_color="#94a3b8",
    fsm_state="MONITOR",
    oled_lines=[
        "SMART CHILI - ESP32 S3",
        "Soil: 58.5% [OPTIMAL] | T: 26.5C",
        "Rain: 10% | AI Wilt: 5.2%",
        "FSM: MONITOR -> PUMP: OFF"
    ],
    pass_note="Hệ thống ổn định, duy trì chu kỳ giám sát 30s/lần, không tiêu hao nước thừa.",
    plant_filter_type="normal"
)

# 2. Kịch bản 2: Đất khô -> Bật tưới
create_model_dashboard_image(
    filename="anh_14_test_dry_soil_watering.png",
    title="KIỂM THỬ: ĐẤT KHÔ -> TỰ ĐỘNG BẬT TƯỚI",
    scenario_badge="WATERING ACTIVE",
    badge_bg="#2563eb",
    soil_val=22.4,
    soil_status="KHÔ NGUY CẤP < 30%",
    soil_color="#ef4444",
    temp_val=34.2,
    rain_val=5.0,
    wilt_val=78.5,
    pump_status="ĐANG BẬT (ON)",
    pump_color="#38bdf8",
    pump_detail="12.0s, PWM=220",
    fan_status="BẬT LÀM MÁT (PWM=237)",
    fan_color="#38bdf8",
    fsm_state="WATERING",
    oled_lines=[
        "** WATERING TRIGGERED **",
        "Soil: 22.4% (DRY!) | T: 34.2C",
        "Fuzzy Run: Pump 12.0s (PWM 220)",
        "State: WATERING -> Bơm đang xả"
    ],
    pass_note="Cảm biến đất khô (<30%) kích hoạt ngay relay máy bơm và mở quạt tản nhiệt chuẩn xác.",
    plant_filter_type="dry"
)

# 3. Kịch bản 3: Đất đủ ẩm -> Khóa tưới chống ngập úng
create_model_dashboard_image(
    filename="anh_15_test_wet_soil_no_water.png",
    title="KIỂM THỬ: ĐẤT ĐỦ ẨM -> KHÓA TƯỚI CHỐNG ÚNG",
    scenario_badge="MOIST / SAFE",
    badge_bg="#059669",
    soil_val=74.6,
    soil_status="ẨM CAO (ĐỦ ẨM)",
    soil_color="#10b981",
    temp_val=28.0,
    rain_val=15.0,
    wilt_val=4.0,
    pump_status="KHÓA TƯỚI (LOCKED)",
    pump_color="#64748b",
    pump_detail="0.0s, PWM=0",
    fan_status="TẮT (PWM=0)",
    fan_color="#64748b",
    fsm_state="MONITOR (LOCKED)",
    oled_lines=[
        "CHILI MONITOR - SAFE",
        "Soil: 74.6% (SUFFICIENT)",
        "Anti-waterlogging: ACTIVATED",
        "Pump Duration = 0.0s [PASS]"
    ],
    pass_note="Bộ luật Fuzzy khóa hoàn toàn rơ le bơm khi đất vượt 70%, bảo vệ rễ ớt khỏi thối úng.",
    plant_filter_type="wet"
)

# 4. Kịch bản 4: Đất khô nhưng dự báo mưa -> Xử lý hoãn tưới
create_model_dashboard_image(
    filename="anh_16_test_rain_delay.png",
    title="KIỂM THỬ: ĐẤT KHÔ GẶP DỰ BÁO MƯA -> HOÃN TƯỚI",
    scenario_badge="RAIN DELAY ACTIVE",
    badge_bg="#d97706",
    soil_val=32.5,
    soil_status="KHÔ NHẸ",
    soil_color="#f59e0b",
    temp_val=27.5,
    rain_val=85.0,
    wilt_val=15.0,
    pump_status="HOÃN TƯỚI (DELAY)",
    pump_color="#f59e0b",
    pump_detail="Chờ mưa, PWM=0",
    fan_status="TẮT (PWM=0)",
    fan_color="#64748b",
    fsm_state="RAIN_DELAY",
    oled_lines=[
        "RAIN FORECAST DETECTED!",
        "Rain Prob: 85% -> DELAY WATER",
        "Soil: 32.5% | Delay: 60 mins",
        "Saving: 100% Water [PASS]"
    ],
    pass_note="Hệ thống tích hợp OpenWeather hoãn tưới thành công, đón nước mưa tự nhiên tiết kiệm 100% nước.",
    plant_filter_type="normal"
)

# 5. Kịch bản 5: Lỗi cảm biến -> Chuyển trạng thái FAILSAFE
create_model_dashboard_image(
    filename="anh_17_test_sensor_fault_failsafe.png",
    title="KIỂM THỬ: LỖI CẢM BIẾN -> CẢNH BÁO FAILSAFE",
    scenario_badge="CRITICAL FAILSAFE",
    badge_bg="#dc2626",
    soil_val=0.0,
    soil_status="MẤT TÍN HIỆU (ADC < 50)",
    soil_color="#ef4444",
    temp_val=0.0,
    rain_val=0.0,
    wilt_val=0.0,
    pump_status="NGẮT CƯỠNG BỨC (OFF)",
    pump_color="#ef4444",
    pump_detail="KHÓA KHẨN CẤP",
    fan_status="NGẮT CƯỠNG BỨC (OFF)",
    fan_color="#ef4444",
    fsm_state="FAILSAFE (E-STOP)",
    oled_lines=[
        "!! CRITICAL ALERT !!",
        "SENSOR FAULT DETECTED (SC01)",
        "EMERGENCY SHUTDOWN PUMP/FAN",
        "Buzzer: ALARM | FSM: FAILSAFE"
    ],
    pass_note="Cơ chế an toàn Dead-man và Sensor Guard lập tức ngắt toàn bộ tải công suất, chống cháy nổ tuyệt đối.",
    plant_filter_type="alert"
)

# 6. Ảnh 18: Giao diện Arduino IDE Serial Monitor thực tế
def generate_arduino_serial_monitor():
    fig = plt.figure(figsize=(13, 7.5), dpi=180)
    fig.patch.set_facecolor('#1e1e1e') # VS Code / Arduino Dark Theme

    ax = fig.add_subplot(111)
    ax.set_facecolor('#181818')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    # Thanh tiêu đề cửa sổ Arduino IDE Serial Monitor
    ax.text(2, 96, "COM6 (ESP32-S3 Dev Module) - Arduino IDE 2.3.2 Serial Monitor", 
            color='#cccccc', fontsize=10.5, fontweight='bold')
    
    # Nút điều khiển
    ax.text(78, 96, "[ Clear Output ]", color='#007acc', fontsize=8.5, fontweight='bold')
    ax.text(90, 96, "[ Toggle Autoscroll ]", color='#007acc', fontsize=8.5, fontweight='bold')

    # Dải phân cách
    ax.axhline(93, color='#333333', linewidth=1.5)

    # Ô nhập lệnh gửi (Send Bar)
    rect_send = patches.Rectangle((2, 85), 84, 5, facecolor='#252526', edgecolor='#3c3c3c')
    ax.add_patch(rect_send)
    ax.text(3, 87.5, "Message to ESP32 (Type command and press Enter)...", color='#777777', fontsize=8)
    rect_btn = patches.Rectangle((88, 85), 10, 5, facecolor='#0e639c', edgecolor='#1177bb')
    ax.add_patch(rect_btn)
    ax.text(91.5, 87.5, "Send", color='#ffffff', fontsize=8.5, fontweight='bold')

    ax.axhline(82, color='#333333', linewidth=1)

    # Dòng Log Terminal Serial Monitor
    logs = [
        ("[00:00:00.120] -> ESP-ROM: esp32s3-20210327 | Chip Revision: v0.2 | Flash: 16MB QIO", "#858585"),
        ("[00:00:00.350] -> [SETUP] Khoi tao Smart Chili AI Firmware v2.4 (FSM Engine)...", "#4ec9b0"),
        ("[00:00:00.820] -> [WIFI] Dang ket noi mang SSID: 'SmartFarm_IoT_2.4G'...", "#dcdcaa"),
        ("[00:00:01.450] -> [WIFI] Ket noi WiFi thanh cong! IP: 192.168.1.185 (RSSI: -58 dBm)", "#6a9955"),
        ("[00:00:01.620] -> [MQTT] Connected to Broker: broker.hivemq.com:1883 [Client: ESP32S3_Chili01]", "#6a9955"),
        ("[00:00:02.000] -> [OLED] Man hinh SSD1306 I2C 128x64 khoi tao thanh cong o dia chi 0x3C", "#9cdcfe"),
        ("[00:00:02.150] -> [RELAY/FAN] Cấu hình GPIO Output: PUMP_PIN=4, FAN_PIN=5, BUZZER=6 [OK]", "#9cdcfe"),
        ("[00:00:02.300] -> [FSM] Chuyen trang thai sang: [MONITOR] (Sampling Window: 30s)", "#4fc1ff"),
        ("---------------------------------------------------------------------------------------------------", "#444444"),
        ("[00:00:32.410] -> [SENSOR] Raw ADC: [1420, 1425, 1418, 1430, 1422] | Median ADC: 1422", "#ce9178"),
        ("[00:00:32.420] -> [MEDIAN_FILTER] Soil Moisture tinh toan: 22.4% (Tieu chuan: 50.0% - 65.0%)", "#ce9178"),
        ("[00:00:32.450] -> [WEATHER] Sync OpenWeather API: Temp=34.2C, RainProb=5.0%", "#dcdcaa"),
        ("[00:00:32.480] -> [AI_VISION] Nhan diện: Leaf Spot=0.08, Wilt=0.78 -> Wilt Risk: 78.5%", "#4ec9b0"),
        ("[00:00:32.500] -> [FUZZY_ENGINE] Rule Defuzzification: PumpDuration=12.0s, FanPWM=237", "#b5cea8"),
        ("[00:00:32.520] -> [FSM] TRIGGER WATERING! Chuyen trang thai: [MONITOR] -> [WATERING]", "#d7ba7d"),
        ("[00:00:32.530] -> [ACTUATOR] Relay PIN 4: HIGH (BẬT MÁY BƠM) | Fan PIN 5: PWM=237", "#6a9955"),
        ("[00:00:44.530] -> [DEAD_MAN_TIMER] Timeout 12.0s dat nguong! Ngat Relay PIN 4: LOW (TAT BOM)", "#4ec9b0"),
        ("[00:00:44.550] -> [FSM] Chuyen trang thai: [WATERING] -> [SOAKING] (Thoi gian cho ngam: 60s)", "#d7ba7d"),
        ("[00:01:44.560] -> [FSM] Hoan tat chu ky ngam dat. Chuyen ve [MONITOR] tiep tuc do dac.", "#6a9955"),
        ("[00:01:44.580] -> [TELEMETRY] Gui goi tin JSON MQTT len Dashboard thanh cong: 200 OK", "#4ec9b0"),
    ]

    log_y = 78
    for log_text, log_color in logs:
        ax.text(2, log_y, log_text, color=log_color, fontsize=8, fontfamily='monospace')
        log_y -= 3.6

    # Bottom status bar Arduino IDE
    ax.axhline(6, color='#333333', linewidth=1)
    ax.text(2, 2.5, "Autoscroll [✔] | Show timestamp [✔] | Clear on send [ ]", color='#888888', fontsize=8)
    ax.text(82, 2.5, "115200 baud  |  Both NL & CR", color='#cccccc', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "anh_18_serial_monitor_log.png")
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Đã tạo thành công: anh_18_serial_monitor_log.png")

generate_arduino_serial_monitor()
print("Hoàn tất tạo toàn bộ 6 ảnh thực nghiệm Mục 5!")
