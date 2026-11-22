# ======================================================================================
# ĐỀ TÀI: HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
# FILE: server.py
# VAI TRÒ: MÁY CHỦ TRUNG TÂM (FASTAPI BACKEND & ORCHESTRATOR)
# LUỒNG HOẠT ĐỘNG 5 TẦNG:
#   1. SENSE : Đọc chuỗi JSON cảm biến từ vi điều khiển ESP32 qua UART 115200.
#   2. SEE   : Truyền video trực tiếp MJPEG và chạy phân tích AI YOLOv8 từ plant_vision.py.
#   3. THINK : Chạy song song Logic mờ (fuzzy_weather.py) + Học máy (smart_garden_tree.joblib).
#   4. ACT   : Ra lệnh điều khiển Bơm/Quạt gửi ngược lại cho ESP32 và Web Dashboard.
#   5. LEARN : Cung cấp API cho AI Labeling Studio tự gán nhãn và re-train lại mô hình YOLOv8.
# ======================================================================================

import os
import sys
import json
import time
import shutil
import sqlite3
import subprocess
import threading
from datetime import datetime
from typing import Optional, List
import cv2
import serial
import serial.tools.list_ports
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from fuzzy_weather import WeatherAndFuzzyEngine
from plant_vision import PlantVisionAnalyzer, get_available_cameras

# ================= CẤU HÌNH HỆ THỐNG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAUD_RATE = 115200
OWM_API_KEY = os.getenv("OWM_API_KEY", "YOUR_API_KEY")

def auto_detect_port():
    # 1. Nếu có chỉ định cổng cụ thể từ biến môi trường
    env_port = os.getenv("SERIAL_PORT")
    if env_port:
        return env_port
    # 2. Tự động dò các cổng COM đang kết nối trên máy tính
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if any(k in desc for k in ["ch34", "cp210", "usb", "serial", "uart", "esp"]):
            return p.device
    if ports:
        return ports[0].device
    return None

SERIAL_PORT = auto_detect_port()

DB_PATH = os.path.join(BASE_DIR, "smart_chili.db")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MODEL_PATH = os.path.join(BASE_DIR, "smart_garden_tree.joblib")
CAPTURED_IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "cay ot", "captured_images"))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "cay ot", "dataset"))
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)

app = FastAPI(title="SMART CHILI AI Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=STORAGE_DIR), name="images")
app.mount("/captured_images", StaticFiles(directory=CAPTURED_IMAGES_DIR), name="captured_images")

def auto_detect_camera_index():
    """Tự động ưu tiên chọn USB Camera cắm ngoài thay vì webcam tích hợp của laptop"""
    try:
        cams = get_available_cameras()
        for c in cams:
            name = (c.get("name", "") + " " + c.get("friendly_name", "")).lower()
            if any(k in name for k in ["usb2", "uvc", "external", "webcam 1"]):
                return c["index"]
        if len(cams) > 1:
            return 1
    except Exception:
        pass
    return 0

fuzzy_engine = WeatherAndFuzzyEngine(api_key=OWM_API_KEY)
vision_engine = PlantVisionAnalyzer(storage_dir=STORAGE_DIR, camera_index=auto_detect_camera_index())
ml_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

system_state = {
    "connected": False,
    "port": None,
    "mode": "AUTO",
    "sensor": {"model": "MH1", "soil_pct": 0, "temp": 0, "hum": 0, "lux": 0, "pump": False, "fan": False, "fault": False, "estop": False},
    "weather": {"rain_prob": 0, "desc": "", "valid": False},
    "ai_decision": {"score": 0, "text": "Đang khởi động...", "ml_label": "Chưa kích hoạt ML"},
    "vision": {"status": "INIT", "leaf_health": "Chưa quét", "green_ratio": 0}
}

ser_conn = None
last_fault_logged = False
uart_paused = False

# ================= CẤU HÌNH NGƯỠNG HỆ THỐNG (THRESHOLDS) =================
DEFAULT_THRESHOLDS = {
    "soil_dry": 35.0,          # Ngưỡng đất khô bắt đầu tưới (%)
    "soil_wet": 65.0,          # Ngưỡng đất đủ ẩm dừng tưới (%)
    "temp_fan": 32.0,          # Ngưỡng nhiệt độ bật quạt (°C)
    "rain_delay": 60.0,        # Ngưỡng xác suất mưa hoãn tưới (%)
    "ai_conf": 15,             # Ngưỡng tự tin nhận diện YOLO (%)
    "yellow_leaf_alert": 15.0  # Ngưỡng cảnh báo tỷ lệ vàng lá (%)
}
THRESHOLDS_FILE = os.path.join(STORAGE_DIR, "thresholds.json")

def load_thresholds():
    if os.path.exists(THRESHOLDS_FILE):
        try:
            with open(THRESHOLDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                res = DEFAULT_THRESHOLDS.copy()
                res.update(data)
                return res
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()

def save_thresholds_file(cfg):
    try:
        with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print("Lỗi lưu thresholds:", e)

thresholds_config = load_thresholds()

class ControlCommand(BaseModel):
    mode: Optional[str] = None
    pump: Optional[bool] = None
    fan: Optional[bool] = None
    estop: Optional[bool] = None
    reset_emergency: Optional[bool] = None

# ================= DATABASE SQLITE =================
uart_buffer = []

def add_uart_log(direction: str, message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    uart_buffer.append({"time": ts, "dir": direction, "msg": message})
    if len(uart_buffer) > 50:
        uart_buffer.pop(0)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, model TEXT, soil_pct REAL, temp REAL, hum REAL,
        lux REAL, rain_prob REAL, fuzzy_score REAL, decision TEXT, pump INTEGER, fan INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS incident_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, code TEXT, description TEXT
    )''')
    conn.commit()
    conn.close()

# Khởi tạo DB ngay khi nạp module
init_db()

def log_sensor(data: dict, res: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO sensor_logs 
        (timestamp, model, soil_pct, temp, hum, lux, rain_prob, fuzzy_score, decision, pump, fan)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("model", "MH1"), data.get("soil_pct", 0), data.get("temp", 0),
        data.get("hum", 0), data.get("lux", 0), res["rain_prob"],
        res["score"], res["decision"], int(data.get("pump", False)), int(data.get("fan", False))
    ))
    conn.commit()
    conn.close()

def log_incident(code: str, desc: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO incident_logs (timestamp, code, description) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), code, desc))
    conn.commit()
    conn.close()

def send_to_esp32(payload: dict):
    global ser_conn
    msg_str = json.dumps(payload)
    add_uart_log("TX", msg_str)
    if ser_conn and ser_conn.is_open:
        try:
            ser_conn.write((msg_str + "\n").encode("utf-8"))
        except Exception:
            pass

# ================= LUỒNG XỬ LÝ NỀN =================
def worker_loop():
    global ser_conn, last_fault_logged, uart_paused
    last_db_log = 0
    last_vision_scan = 0

    while True:
        # Nếu đang tạm ngắt để nạp code Arduino IDE
        if uart_paused:
            if ser_conn is not None:
                try:
                    ser_conn.close()
                except Exception:
                    pass
                ser_conn = None
            system_state["connected"] = False
            system_state["port"] = "Đã ngắt (Sẵn sàng nạp .ino)"
            time.sleep(1)
            continue

        if ser_conn is None or not ser_conn.is_open:
            target_port = auto_detect_port()
            if target_port:
                try:
                    ser_conn = serial.Serial(target_port, BAUD_RATE, timeout=1)
                    system_state["connected"] = True
                    system_state["port"] = target_port
                    print(f"[UART] Đã kết nối thành công ESP32 qua cổng {target_port}!")
                    time.sleep(2)
                except Exception as e:
                    system_state["connected"] = False
                    system_state["port"] = None
                    time.sleep(3)
                    continue
            else:
                system_state["connected"] = False
                system_state["port"] = None
                time.sleep(3)
                continue

        try:
            if ser_conn.in_waiting > 0:
                line = ser_conn.readline().decode("utf-8", errors="ignore").strip()
                
                # Nhận và xử lý gói ảnh từ ESP32 Camera
                if "###CAM_START###" in line and "###CAM_END###" in line:
                    b64 = line.split("###CAM_START###")[1].split("###CAM_END###")[0].strip()
                    add_uart_log("RX", f"[CAMERA FRAME {len(b64)} chars]")
                    system_state["vision"] = vision_engine.analyze_base64(b64)
                    system_state["vision"]["source"] = "ESP32-CAM"
                elif line.startswith("{"):
                    add_uart_log("RX", line)
                    try:
                        data = json.loads(line)
                        system_state["sensor"] = data

                        # Ghi nhận sự cố SC01
                        if data.get("fault") and not last_fault_logged:
                            log_incident("SC01", "Mất tín hiệu hoặc đứt dây cảm biến độ ẩm đất")
                            last_fault_logged = True
                        elif not data.get("fault"):
                            last_fault_logged = False

                        # Đánh giá Fuzzy Logic + Thời tiết theo ngưỡng tùy chỉnh
                        res = fuzzy_engine.evaluate(
                            data.get("soil_pct", 50),
                            data.get("temp", 28),
                            rain_delay_thresh=thresholds_config.get("rain_delay", 60.0),
                            soil_dry_thresh=thresholds_config.get("soil_dry", 35.0),
                            soil_wet_thresh=thresholds_config.get("soil_wet", 65.0)
                        )
                        system_state["weather"] = {
                            "rain_prob": res["rain_prob"],
                            "desc": res["weather_desc"],
                            "valid": res["api_valid"]
                        }

                        # Dự đoán tham chiếu từ mô hình ML (.joblib) nếu đã train
                        ml_text = "Chưa train model"
                        if ml_model is not None:
                            try:
                                X_df = pd.DataFrame([{
                                    "soil_pct": data.get("soil_pct", 50),
                                    "temp": data.get("temp", 28),
                                    "hum": data.get("hum", 60),
                                    "lux": data.get("lux", 500),
                                    "rain_prob": res["rain_prob"]
                                }])
                                pred = int(ml_model.predict(X_df)[0])
                                labels = {0: "ML: Không tưới", 1: "ML: Hoãn tưới", 2: "ML: Đề xuất tưới"}
                                ml_text = labels.get(pred, "")
                            except Exception:
                                pass

                        system_state["ai_decision"] = {
                            "score": res["score"],
                            "text": res["decision"],
                            "ml_label": ml_text
                        }

                        if system_state["mode"] == "AUTO" and not data.get("fault") and not data.get("estop"):
                            fan_temp_thresh = thresholds_config.get("temp_fan", 32.0)
                            fan_cmd = True if data.get("temp", 0) > fan_temp_thresh else False
                            send_to_esp32({
                                "mode": "AUTO",
                                "pump": res["pump"],
                                "pump_speed": res["pump_speed"],
                                "fan": fan_cmd,
                                "fan_speed": 210 if fan_cmd else 0
                            })

                        if time.time() - last_db_log > 10:
                            log_sensor(data, res)
                            last_db_log = time.time()
                    except Exception as err:
                        print(f"[UART] Lỗi xử lý JSON: {err}")
                elif line:
                    add_uart_log("RX", line)

            if time.time() - last_vision_scan > 3:
                last_vision_scan = time.time()
                # Cập nhật kết quả phân tích thị giác liên tục từ worker
                if not (system_state["connected"] and system_state["sensor"].get("cam_ready")):
                    system_state["vision"] = vision_engine.get_latest_analysis()

        except Exception:
            system_state["connected"] = False
            system_state["port"] = None
            if ser_conn:
                try:
                    ser_conn.close()
                except Exception:
                    pass
            ser_conn = None

        time.sleep(0.2)

@app.on_event("startup")
def on_startup():
    init_db()
    # Đồng bộ ngưỡng vào vision engine
    if vision_engine:
        vision_engine.yolo_conf = thresholds_config.get("ai_conf", 15) / 100.0
        vision_engine.yellow_alert_pct = thresholds_config.get("yellow_leaf_alert", 15.0)
    threading.Thread(target=worker_loop, daemon=True).start()

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/api/status")
def get_status():
    st = dict(system_state)
    st["uart_paused"] = uart_paused
    st["vision"] = vision_engine.get_latest_analysis()
    return st

@app.post("/api/uart/disconnect")
def disconnect_uart():
    global uart_paused, ser_conn
    uart_paused = True
    if ser_conn:
        try:
            ser_conn.close()
        except Exception:
            pass
        ser_conn = None
    system_state["connected"] = False
    system_state["port"] = "Đã ngắt (Sẵn sàng nạp .ino)"
    add_uart_log("SYS", "🔌 Đã ngắt cổng COM để giải phóng cho Arduino IDE nạp code .ino")
    return {"status": "ok", "message": "Đã ngắt kết nối cổng COM! Bạn có thể nạp code trong Arduino IDE."}

@app.post("/api/uart/reconnect")
def reconnect_uart():
    global uart_paused
    uart_paused = False
    add_uart_log("SYS", "⚡ Đang khôi phục kết nối UART...")
    return {"status": "ok", "message": "Đang kết nối lại ESP32..."}

@app.get("/api/ports")
def get_ports():
    ports = [{"device": p.device, "description": p.description} for p in serial.tools.list_ports.comports()]
    return {"current": system_state.get("port"), "baud": BAUD_RATE, "available": ports}

class PortSelectCommand(BaseModel):
    port: str

@app.post("/api/ports/select")
def select_port(cmd: PortSelectCommand):
    global SERIAL_PORT, ser_conn, uart_paused
    uart_paused = False
    SERIAL_PORT = cmd.port
    if ser_conn:
        try:
            ser_conn.close()
        except Exception:
            pass
        ser_conn = None
    system_state["connected"] = False
    system_state["port"] = cmd.port
    add_uart_log("SYS", f"Đã chuyển cổng sang {cmd.port}")
    return {"status": "ok", "port": cmd.port}

class UartSendCommand(BaseModel):
    text: str

@app.post("/api/uart/send")
def send_raw_uart(cmd: UartSendCommand):
    global ser_conn
    add_uart_log("TX", cmd.text)
    if ser_conn and ser_conn.is_open:
        try:
            ser_conn.write((cmd.text + "\n").encode("utf-8"))
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Chưa kết nối cổng UART"}

@app.get("/api/uart/logs")
def get_uart_logs():
    return {
        "logs": uart_buffer,
        "port": system_state.get("port"),
        "connected": system_state.get("connected"),
        "baud": BAUD_RATE
    }

class CameraSelectCommand(BaseModel):
    index: int

@app.get("/api/cameras")
def list_cameras(refresh: bool = False):
    cams = get_available_cameras(force_refresh=refresh)
    return {
        "current_index": vision_engine.camera_index,
        "available": cams
    }

@app.post("/api/cameras/select")
def select_camera(cmd: CameraSelectCommand):
    res = vision_engine.set_camera_index(cmd.index)
    system_state["vision"] = res
    add_uart_log("SYS", f"Đã chuyển USB Camera sang cổng {cmd.index}")
    return {"status": "ok", "selected_index": cmd.index, "data": res}

def generate_mjpeg():
    """Bộ tạo luồng video MJPEG thời gian thực 25 FPS siêu mượt"""
    while True:
        frame_bytes = vision_engine.get_jpeg_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.get("/api/camera/stream")
def camera_stream():
    """Endpoint truyền video trực tiếp (Live Stream) thời gian thực không độ trễ"""
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/camera/toggle_overlay")
def toggle_camera_overlay():
    enabled = vision_engine.toggle_overlay()
    return {"status": "ok", "overlay_enabled": enabled, "message": f"Khung nhận diện AI: {'BẬT' if enabled else 'TẮT'}"}

@app.post("/api/capture")
def trigger_capture():
    if system_state["connected"] and system_state["sensor"].get("cam_ready"):
        send_to_esp32({"cmd": "capture"})
        return {"status": "ok", "source": "ESP32-CAM", "message": "Đã gửi lệnh chụp tới ESP32-CAM"}
    else:
        res = vision_engine.capture_and_analyze()
        system_state["vision"] = res
        return {"status": "ok", "source": "WEBCAM", "data": res}

# ================= CÁC API CÀI ĐẶT NGƯỠNG HỆ THỐNG (MANUAL THRESHOLDS) =================
@app.get("/api/thresholds")
def get_thresholds():
    """Lấy danh sách các ngưỡng hiện tại của hệ thống"""
    return {"status": "ok", "thresholds": thresholds_config}

@app.post("/api/thresholds")
def update_thresholds(req: dict):
    """Cập nhật các ngưỡng hệ thống do người dùng chỉnh tay trên giao diện"""
    global thresholds_config
    updated_keys = []
    for k, v in req.items():
        if k in DEFAULT_THRESHOLDS:
            try:
                thresholds_config[k] = float(v)
                updated_keys.append(k)
            except (ValueError, TypeError):
                pass
    save_thresholds_file(thresholds_config)

    # Đồng bộ ngưỡng AI trực tiếp sang vision engine
    if vision_engine:
        if "ai_conf" in req:
            vision_engine.yolo_conf = float(thresholds_config["ai_conf"]) / 100.0
        if "yellow_leaf_alert" in req:
            vision_engine.yellow_alert_pct = float(thresholds_config["yellow_leaf_alert"])

    add_uart_log("SYS", f"⚙️ Đã lưu cập nhật ngưỡng: {', '.join(updated_keys)}")
    return {"status": "ok", "message": "Đã lưu cài đặt ngưỡng thành công!", "thresholds": thresholds_config}

@app.post("/api/thresholds/reset")
def reset_thresholds():
    """Khôi phục tất cả ngưỡng về mặc định nhà sản xuất"""
    global thresholds_config
    thresholds_config = DEFAULT_THRESHOLDS.copy()
    save_thresholds_file(thresholds_config)
    if vision_engine:
        vision_engine.yolo_conf = float(thresholds_config["ai_conf"]) / 100.0
        vision_engine.yellow_alert_pct = float(thresholds_config["yellow_leaf_alert"])
    add_uart_log("SYS", "🔄 Đã khôi phục các ngưỡng về giá trị mặc định")
    return {"status": "ok", "message": "Đã khôi phục ngưỡng về mặc định!", "thresholds": thresholds_config}

@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logs = [dict(r) for r in conn.execute("SELECT * FROM sensor_logs ORDER BY id DESC LIMIT 30").fetchall()]
    incidents = [dict(r) for r in conn.execute("SELECT * FROM incident_logs ORDER BY id DESC LIMIT 15").fetchall()]
    conn.close()
    return {"logs": logs, "incidents": incidents}

class PredictRequest(BaseModel):
    soil_pct: Optional[float] = None
    soil: Optional[float] = None
    temp: Optional[float] = None
    temperature: Optional[float] = None
    hum: Optional[float] = None
    humidity: Optional[float] = None
    lux: Optional[float] = 500.0
    rain_prob: Optional[float] = 0.0

@app.post("/predict")
@app.post("/api/predict")
def predict_watering(req: PredictRequest):
    """API dự đoán nhu cầu tưới nước bằng mô hình Machine Learning chuẩn Rubric"""
    if ml_model is None:
        return {"error": "Mô hình ML chưa được nạp (cần chạy train_model.py)"}

    s = req.soil_pct if req.soil_pct is not None else (req.soil if req.soil is not None else 50.0)
    t = req.temp if req.temp is not None else (req.temperature if req.temperature is not None else 28.0)
    h = req.hum if req.hum is not None else (req.humidity if req.humidity is not None else 60.0)
    l = req.lux if req.lux is not None else 500.0
    r = req.rain_prob if req.rain_prob is not None else 0.0

    X_df = pd.DataFrame([{
        "soil_pct": float(s),
        "temp": float(t),
        "hum": float(h),
        "lux": float(l),
        "rain_prob": float(r)
    }])
    pred = int(ml_model.predict(X_df)[0])
    labels = {0: "Không tưới (Đất đủ ẩm)", 1: "Hoãn tưới (Sắp mưa/Bù ẩm nhẹ)", 2: "Tưới nước (Đất khô nguy cấp)"}
    return {
        "watering": pred,  # Nhãn 0, 1, 2 chuẩn Rubric
        "label": pred,
        "action": labels.get(pred, ""),
        "inputs": {"soil_pct": s, "temp": t, "hum": h, "lux": l, "rain_prob": r}
    }

@app.post("/api/control")
def post_control(cmd: ControlCommand):
    payload = {}
    if cmd.mode:
        system_state["mode"] = cmd.mode
        payload["mode"] = cmd.mode
    if cmd.pump is not None:
        payload["pump"] = cmd.pump
        payload["pump_speed"] = 220 if cmd.pump else 0
    if cmd.fan is not None:
        payload["fan"] = cmd.fan
        payload["fan_speed"] = 220 if cmd.fan else 0
    if cmd.estop is not None:
        payload["estop"] = cmd.estop
        if cmd.estop:
            log_incident("ESTOP", "Kích hoạt dừng khẩn cấp từ Web Dashboard")
    if cmd.reset_emergency is not None:
        payload["reset_emergency"] = cmd.reset_emergency

    send_to_esp32(payload)
    return {"status": "ok", "payload": payload}

# ================= HỆ THỐNG AI LABELING STUDIO & MLOPS =================
labeling_train_state = {
    "status": "IDLE",
    "progress": 0,
    "message": "Sẵn sàng gán nhãn",
    "updated_at": datetime.now().strftime("%H:%M:%S")
}

class AnnotationItem(BaseModel):
    cls_id: int
    cls_name: Optional[str] = "ot_do"
    x: float
    y: float
    w: float
    h: float
    points: Optional[List[List[float]]] = None

class SaveAnnotationsRequest(BaseModel):
    filename: str
    annotations: List[AnnotationItem]

class AutoDetectRequest(BaseModel):
    filename: str

@app.get("/api/labeling/images")
def list_labeling_images(limit: int = 60, offset: int = 0):
    all_files = sorted([f for f in os.listdir(CAPTURED_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))], reverse=True)
    total = len(all_files)
    slice_files = all_files[offset:offset+limit]
    
    items = []
    for f in slice_files:
        base = os.path.splitext(f)[0]
        lbl_p = os.path.join(DATASET_DIR, "labels", "train", f"{base}.txt")
        has_lbl = os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0
        items.append({
            "filename": f,
            "url": f"/captured_images/{f}",
            "labeled": has_lbl
        })
    return {"total": total, "images": items, "classes": vision_engine.CLASS_INFO}

@app.get("/api/labeling/annotation/{filename}")
def get_labeling_annotation(filename: str):
    base = os.path.splitext(filename)[0]
    txt_path = os.path.join(DATASET_DIR, "labels", "train", f"{base}.txt")
    if not os.path.exists(txt_path):
        txt_path = os.path.join(DATASET_DIR, "labels", "val", f"{base}.txt")
    if not os.path.exists(txt_path):
        return {"filename": filename, "annotations": []}
        
    annos = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    name = vision_engine.CLASS_INFO.get(cls_id, {}).get("name", f"class_{cls_id}")
                    label = vision_engine.CLASS_INFO.get(cls_id, {}).get("label", name)
                    color = vision_engine.CLASS_INFO.get(cls_id, {}).get("color", (255, 255, 255))
                    if len(parts) == 5:
                        x_c, y_c, w, h = map(float, parts[1:5])
                        annos.append({
                            "cls_id": cls_id,
                            "cls_name": name,
                            "cls_label": label,
                            "color": color,
                            "x": x_c,
                            "y": y_c,
                            "w": w,
                            "h": h,
                            "type": "bbox",
                            "points": [
                                [x_c - w/2, y_c - h/2],
                                [x_c + w/2, y_c - h/2],
                                [x_c + w/2, y_c + h/2],
                                [x_c - w/2, y_c + h/2]
                            ]
                        })
                    elif len(parts) > 5 and len(parts) % 2 == 1:
                        coords = list(map(float, parts[1:]))
                        pts = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        annos.append({
                            "cls_id": cls_id,
                            "cls_name": name,
                            "cls_label": label,
                            "color": color,
                            "x": (min_x + max_x) / 2.0,
                            "y": (min_y + max_y) / 2.0,
                            "w": max_x - min_x,
                            "h": max_y - min_y,
                            "type": "polygon",
                            "points": pts
                        })
    except Exception as e:
        print("Lỗi đọc annotation:", e)
    return {"filename": filename, "annotations": annos}

@app.post("/api/labeling/snapshot")
def capture_snapshot_for_labeling():
    frame = vision_engine.get_raw_frame()
    if frame is None or frame.size == 0:
        return {"error": "Camera chưa sẵn sàng hoặc không có khung hình"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"snap_{ts}.jpg"
    fpath = os.path.join(CAPTURED_IMAGES_DIR, fname)
    cv2.imwrite(fpath, frame)
    return {"status": "ok", "filename": fname, "url": f"/captured_images/{fname}"}

@app.post("/api/labeling/auto_detect")
def auto_detect_labeling(req: AutoDetectRequest):
    img_path = os.path.join(CAPTURED_IMAGES_DIR, req.filename)
    if not os.path.exists(img_path):
        return {"error": "Không tìm thấy file ảnh"}
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Không thể đọc file ảnh"}
    h_img, w_img = img.shape[:2]
    
    results = []
    if vision_engine.yolo_model is not None:
        try:
            preds = vision_engine.yolo_model.predict(img, conf=0.12, imgsz=416, verbose=False)
            for b in preds[0].boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                x_c = ((x1 + x2) / 2.0) / w_img
                y_c = ((y1 + y2) / 2.0) / h_img
                w_norm = (x2 - x1) / float(w_img)
                h_norm = (y2 - y1) / float(h_img)
                name = vision_engine.CLASS_INFO.get(cls_id, {}).get("name", f"class_{cls_id}")
                label = vision_engine.CLASS_INFO.get(cls_id, {}).get("label", name)
                color = vision_engine.CLASS_INFO.get(cls_id, {}).get("color", (255, 255, 255))
                
                pts = [
                    [round(x1 / float(w_img), 4), round(y1 / float(h_img), 4)],
                    [round(x2 / float(w_img), 4), round(y1 / float(h_img), 4)],
                    [round(x2 / float(w_img), 4), round(y2 / float(h_img), 4)],
                    [round(x1 / float(w_img), 4), round(y2 / float(h_img), 4)]
                ]
                results.append({
                    "cls_id": cls_id,
                    "cls_name": name,
                    "cls_label": label,
                    "color": color,
                    "conf": round(conf, 2),
                    "x": round(x_c, 4),
                    "y": round(y_c, 4),
                    "w": round(w_norm, 4),
                    "h": round(h_norm, 4),
                    "type": "bbox",
                    "points": pts
                })
        except Exception as e:
            print("Auto detect error:", e)
    return {"status": "ok", "annotations": results}

@app.post("/api/labeling/save")
def save_labeling_annotations(req: SaveAnnotationsRequest):
    fname = req.filename
    base_name = os.path.splitext(fname)[0]
    txt_name = f"{base_name}.txt"
    
    train_lbl_path = os.path.join(DATASET_DIR, "labels", "train", txt_name)
    train_img_path = os.path.join(DATASET_DIR, "images", "train", fname)
    src_img = os.path.join(CAPTURED_IMAGES_DIR, fname)
    
    lines = []
    for ann in req.annotations:
        if ann.points and len(ann.points) >= 3:
            flat_pts = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in ann.points])
            lines.append(f"{ann.cls_id} {flat_pts}\n")
        else:
            lines.append(f"{ann.cls_id} {ann.x:.6f} {ann.y:.6f} {ann.w:.6f} {ann.h:.6f}\n")
            
    os.makedirs(os.path.dirname(train_lbl_path), exist_ok=True)
    with open(train_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    if os.path.exists(src_img) and not os.path.exists(train_img_path):
        os.makedirs(os.path.dirname(train_img_path), exist_ok=True)
        shutil.copy2(src_img, train_img_path)
        
    return {"status": "ok", "count": len(lines), "message": f"Đã lưu thành công {len(lines)} nhãn cho ảnh {fname}!"}

def run_retrain_worker():
    global labeling_train_state
    labeling_train_state["status"] = "TRAINING"
    labeling_train_state["message"] = "Đang bắt đầu huấn luyện YOLOv8..."
    labeling_train_state["updated_at"] = datetime.now().strftime("%H:%M:%S")
    
    train_script = os.path.abspath(os.path.join(BASE_DIR, "..", "cay ot", "train_yolo.py"))
    try:
        proc = subprocess.run([sys.executable, train_script], capture_output=True, text=True, cwd=os.path.dirname(train_script))
        if proc.returncode == 0:
            vision_engine._load_yolo()
            labeling_train_state["status"] = "SUCCESS"
            labeling_train_state["message"] = "Huấn luyện hoàn tất! Mô hình mới đã được cập nhật trực tiếp vào camera."
        else:
            labeling_train_state["status"] = "ERROR"
            labeling_train_state["message"] = f"Lỗi khi huấn luyện: {proc.stderr[:300]}"
    except Exception as e:
        labeling_train_state["status"] = "ERROR"
        labeling_train_state["message"] = f"Lỗi: {str(e)}"
    labeling_train_state["updated_at"] = datetime.now().strftime("%H:%M:%S")

@app.post("/api/labeling/retrain")
def trigger_retrain():
    if labeling_train_state["status"] == "TRAINING":
        return {"status": "running", "message": "Tiến trình huấn luyện đang chạy, vui lòng đợi..."}
    t = threading.Thread(target=run_retrain_worker, daemon=True)
    t.start()
    return {"status": "started", "message": "Đã bắt đầu tiến trình huấn luyện mô hình YOLOv8 trên nền!"}

@app.get("/api/labeling/retrain_status")
def get_retrain_status():
    return labeling_train_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)