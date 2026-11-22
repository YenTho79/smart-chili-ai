# ======================================================================================
# ĐỀ TÀI: HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
# FILE: fuzzy_weather.py
# VAI TRÒ: THUẬT TOÁN LOGIC MỜ (FUZZY LOGIC) & TÍCH HỢP DỰ BÁO THỜI TIẾT OPENWEATHER API
# THUẬT TOÁN: HỆ SUY DIỄN MỜ MAMDANI (MAMDANI FUZZY INFERENCE SYSTEM)
# - BIẾN ĐẦU VÀO (ANTECEDENTS):
#     1. Độ ẩm đất (soil): very_dry (rất khô), dry (khô), moist (đủ ẩm), wet (ướt)
#     2. Nhiệt độ khí quyển (temp): cool (mát), normal (bình thường), hot (nóng)
#     3. Khả năng mưa (rain): low (thấp), medium (trung bình), high (cao)
# - BIẾN ĐẦU RA (CONSEQUENT):
#     Nhu cầu tưới (water): none (không tưới), delay (hoãn tưới), low (tưới ít), high (tưới mạnh)
# ======================================================================================

import time
import requests
import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False

class WeatherAndFuzzyEngine:
    def __init__(self, api_key="YOUR_API_KEY", lat=10.9465, lon=106.8340):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.last_fetch = 0
        self.ttl = 600  # Cập nhật mỗi 10 phút
        self.rain_prob = 0.0
        self.weather_desc = "Chưa cập nhật"
        self.api_valid = False
        self._build_fuzzy()

    def _build_fuzzy(self):
        """XÂY DỰNG KHÔNG GIAN BIẾN MỜ VÀ TẬP LUẬT MAMDANI"""
        if not SKFUZZY_AVAILABLE:
            self.sim = None
            return

        try:
            # 1. Khởi tạo các biến ngôn ngữ
            soil = ctrl.Antecedent(np.arange(0, 101, 1), 'soil')
            temp = ctrl.Antecedent(np.arange(15, 46, 1), 'temp')
            rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')
            water = ctrl.Consequent(np.arange(0, 101, 1), 'water')

            # 2. Định nghĩa hàm liên thuộc hình thang (trapmf) và hình tam giác (trimf)
            soil['very_dry'] = fuzz.trapmf(soil.universe, [0, 0, 15, 25])
            soil['dry']      = fuzz.trimf(soil.universe, [20, 35, 50])
            soil['moist']    = fuzz.trimf(soil.universe, [45, 60, 75])
            soil['wet']      = fuzz.trapmf(soil.universe, [70, 85, 100, 100])

            temp['cool']   = fuzz.trapmf(temp.universe, [15, 15, 22, 26])
            temp['normal'] = fuzz.trimf(temp.universe, [24, 28, 32])
            temp['hot']    = fuzz.trapmf(temp.universe, [30, 34, 45, 45])

            rain['low']    = fuzz.trapmf(rain.universe, [0, 0, 20, 40])
            rain['medium'] = fuzz.trimf(rain.universe, [30, 55, 75])
            rain['high']   = fuzz.trapmf(rain.universe, [65, 80, 100, 100])

            water['none']  = fuzz.trapmf(water.universe, [0, 0, 15, 30])
            water['delay'] = fuzz.trimf(water.universe, [25, 40, 55])
            water['low']   = fuzz.trimf(water.universe, [50, 65, 80])
            water['high']  = fuzz.trapmf(water.universe, [75, 85, 100, 100])

            # 3. TẬP LUẬT CHUYÊN GIA MỜ (MAMDANI RULES)
            rules = [
                ctrl.Rule(soil['very_dry'], water['high']),                           # Luật 1: Đất rất khô -> Ưu tiên tưới mạnh
                ctrl.Rule(soil['moist'] | soil['wet'], water['none']),                # Luật 2: Đất đủ ẩm hoặc ướt -> Không tưới
                ctrl.Rule(soil['dry'] & temp['hot'] & rain['low'], water['high']),    # Luật 3: Đất khô + trời nóng + ít mưa -> Tưới mạnh
                ctrl.Rule(soil['dry'] & (temp['normal'] | temp['cool']) & rain['low'], water['low']), # Luật 4: Đất khô + trời mát -> Tưới nhẹ
                ctrl.Rule(soil['dry'] & rain['medium'], water['low']),                # Luật 5: Đất khô + khả năng mưa vừa -> Tưới ít
                ctrl.Rule(soil['dry'] & rain['high'], water['delay'])                 # Luật 6 (Trọng tâm): Đất khô + SẮP CÓ MƯA -> HOÃN TƯỚI
            ]
            self.sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        except Exception:
            self.sim = None

    def get_weather(self):
        now = time.time()
        if now - self.last_fetch < self.ttl and self.api_valid:
            return self.rain_prob, self.weather_desc, True

        if self.api_key == "YOUR_API_KEY":
            # Giả lập sắp mưa 80% để test kịch bản trọng tâm nếu chưa điền API Key
            self.rain_prob = 80.0
            self.weather_desc = "Dự báo mưa rào (Mô phỏng)"
            self.api_valid = True
            self.last_fetch = now
            return self.rain_prob, self.weather_desc, True

        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={self.lat}&lon={self.lon}&appid={self.api_key}&cnt=2&units=metric&lang=vi"
            res = requests.get(url, timeout=5).json()
            self.rain_prob = round(res['list'][0].get('pop', 0.0) * 100.0, 1)
            self.weather_desc = res['list'][0]['weather'][0]['description']
            self.api_valid = True
            self.last_fetch = now
        except Exception:
            self.api_valid = False
            self.weather_desc = "Lỗi kết nối API thời tiết (SC02)"

        return self.rain_prob, self.weather_desc, self.api_valid

    def evaluate(self, soil_pct: float, temp_c: float, rain_delay_thresh: float = 60.0, soil_dry_thresh: float = 35.0, soil_wet_thresh: float = 65.0):
        """
        BƯỚC GIẢI MỜ (DEFUZZIFICATION) & SUY DIỄN QUYẾT ĐỊNH TƯỚI
        - Nạp các thông số thực tế vào các Antecedent mờ
        - Kích hoạt suy diễn luật mờ qua bộ mô phỏng ControlSystemSimulation
        - Nhận giá trị điểm nhu cầu tưới defuzzified (score: 0 - 100)
        """
        rain_prob, desc, valid = self.get_weather()
        effective_rain = rain_prob if valid else 0.0

        if not SKFUZZY_AVAILABLE or self.sim is None:
            # Fallback mô hình mờ chuẩn tính toán trực tiếp
            if soil_pct < 20:
                score = 90.0
            elif soil_pct <= soil_dry_thresh:
                score = 80.0 if effective_rain < 40 else 45.0
            elif soil_pct < soil_wet_thresh:
                score = 30.0 if effective_rain > 50 else 55.0
            else:
                score = 10.0
        else:
            try:
                self.sim.input['soil'] = np.clip(soil_pct, 0, 100)
                self.sim.input['temp'] = np.clip(temp_c, 15, 45)
                self.sim.input['rain'] = np.clip(effective_rain, 0, 100)
                self.sim.compute() # Thực thi giải mờ tính toán trọng tâm (Centroid Defuzzification)
                score = float(self.sim.output.get('water', 0.0))
            except Exception:
                score = 0.0 if soil_pct >= soil_wet_thresh else (80.0 if soil_pct < soil_dry_thresh else 40.0)

        # PHÂN CẤP QUYẾT ĐỊNH ĐIỀU KHIỂN BƠM (ACTUATOR COMMAND)
        if soil_pct >= soil_wet_thresh:
            decision = f"KHÔNG TƯỚI (Đất đủ ẩm {soil_pct}% >= {soil_wet_thresh}%)"
            pump_cmd, pump_spd = False, 0
        elif effective_rain >= rain_delay_thresh:
            decision = f"HOÃN TƯỚI (Dự báo mưa {effective_rain}% >= {rain_delay_thresh}%)"
            pump_cmd, pump_spd = False, 0
        elif score >= 70 or soil_pct <= soil_dry_thresh:
            decision = f"TƯỚI MẠNH (Đất khô {soil_pct}% <= {soil_dry_thresh}%)"
            pump_cmd, pump_spd = True, 230
        elif score >= 50:
            decision = "TƯỚI ÍT (Bổ sung ẩm nhẹ)"
            pump_cmd, pump_spd = True, 175
        elif score >= 30:
            decision = "HOÃN TƯỚI (Dự báo sắp mưa)"
            pump_cmd, pump_spd = False, 0
        else:
            decision = "KHÔNG TƯỚI (Đất đủ ẩm)"
            pump_cmd, pump_spd = False, 0

        return {
            "score": round(score, 1),
            "pump": pump_cmd,
            "pump_speed": pump_spd,
            "decision": decision,
            "rain_prob": rain_prob,
            "weather_desc": desc,
            "api_valid": valid
        }
