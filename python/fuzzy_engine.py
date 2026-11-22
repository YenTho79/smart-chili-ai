import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class SmartChiliFuzzyEngine:
    def __init__(self):
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        # ================= 4. KHAI BÁO CÁC BIẾN ĐẦU VÀO VÀ ĐẦU RA =================
        # 4 Biến đầu vào (Antecedents)
        self.soil = ctrl.Antecedent(np.arange(0, 101, 1), 'Soil')           # Độ ẩm đất (% 0-100)
        self.temp = ctrl.Antecedent(np.arange(15, 46, 1), 'Temperature')   # Nhiệt độ không khí (°C 15-45)
        self.rain = ctrl.Antecedent(np.arange(0, 101, 1), 'Rain')           # Xác suất mưa dự báo (% 0-100)
        self.wilt = ctrl.Antecedent(np.arange(0, 101, 1), 'WiltRisk')       # Nguy cơ héo rũ từ AI Vision (% 0-100)

        # 2 Biến đầu ra (Consequents)
        self.pump = ctrl.Consequent(np.arange(0, 16, 0.5), 'PumpDuration')  # Thời gian tưới (Giây: 0 - 15s)
        self.fan  = ctrl.Consequent(np.arange(0, 256, 1), 'FanPWM')         # Tốc độ quạt làm mát (PWM: 0 - 255)

        # ================= 5. THIẾT KẾ CÁC HÀM THUỘC (MEMBERSHIP FUNCTIONS) =================
        # Hàm thuộc cho Độ ẩm đất (Soil)
        self.soil['Very_Dry'] = fuzz.trapmf(self.soil.universe, [0, 0, 15, 25])
        self.soil['Dry']      = fuzz.trimf(self.soil.universe, [20, 35, 50])
        self.soil['Optimal']  = fuzz.trimf(self.soil.universe, [45, 60, 75])
        self.soil['Wet']      = fuzz.trapmf(self.soil.universe, [70, 85, 100, 100])

        # Hàm thuộc cho Nhiệt độ không khí (Temperature)
        self.temp['Cool']   = fuzz.trapmf(self.temp.universe, [15, 15, 22, 26])
        self.temp['Normal'] = fuzz.trimf(self.temp.universe, [24, 28, 33])
        self.temp['Hot']    = fuzz.trapmf(self.temp.universe, [31, 36, 45, 45])

        # Hàm thuộc cho Xác suất mưa (Rain)
        self.rain['Low']    = fuzz.trapmf(self.rain.universe, [0, 0, 20, 40])
        self.rain['Medium'] = fuzz.trimf(self.rain.universe, [30, 50, 75])
        self.rain['High']   = fuzz.trapmf(self.rain.universe, [65, 80, 100, 100])

        # Hàm thuộc cho Nguy cơ héo rũ AI (Wilt Risk)
        self.wilt['Low']    = fuzz.trapmf(self.wilt.universe, [0, 0, 25, 45])
        self.wilt['Medium'] = fuzz.trimf(self.wilt.universe, [35, 55, 75])
        self.wilt['High']   = fuzz.trapmf(self.wilt.universe, [65, 80, 100, 100])

        # Hàm thuộc cho Đầu ra Bơm nước (Pump Duration: giây)
        self.pump['Off']    = fuzz.trapmf(self.pump.universe, [0, 0, 0.5, 2.0])
        self.pump['Short']  = fuzz.trimf(self.pump.universe, [1.5, 4.0, 7.0])
        self.pump['Medium'] = fuzz.trimf(self.pump.universe, [5.5, 8.5, 12.0])
        self.pump['Long']   = fuzz.trapmf(self.pump.universe, [10.0, 12.5, 15.0, 15.0])

        # Hàm thuộc cho Đầu ra Quạt làm mát (Fan PWM)
        self.fan['Off']    = fuzz.trapmf(self.fan.universe, [0, 0, 30, 60])
        self.fan['Low']    = fuzz.trimf(self.fan.universe, [50, 110, 170])
        self.fan['Medium'] = fuzz.trimf(self.fan.universe, [140, 190, 230])
        self.fan['Max']    = fuzz.trapmf(self.fan.universe, [210, 240, 255, 255])

        # ================= 6. TẬP LUẬT MỜ (RULE BASE) =================
        rules = [
            # Nhóm 1: Đất đủ ẩm hoặc ướt -> Luôn TẮT BƠM (Ưu tiên chống ngập úng)
            ctrl.Rule(self.soil['Optimal'] | self.soil['Wet'], self.pump['Off']),
            
            # Nhóm 2: Đất rất khô -> Tưới thời gian dài
            ctrl.Rule(self.soil['Very_Dry'] & self.rain['Low'], self.pump['Long']),
            ctrl.Rule(self.soil['Very_Dry'] & self.rain['Medium'], self.pump['Medium']),
            ctrl.Rule(self.soil['Very_Dry'] & self.rain['High'], self.pump['Short']),

            # Nhóm 3: Đất khô + Dự báo mưa cao -> HOÃN TƯỚI (Tiết kiệm nước)
            ctrl.Rule(self.soil['Dry'] & self.rain['High'], self.pump['Off']),
            ctrl.Rule(self.soil['Dry'] & self.rain['Medium'], self.pump['Short']),
            ctrl.Rule(self.soil['Dry'] & self.rain['Low'] & self.temp['Hot'], self.pump['Long']),
            ctrl.Rule(self.soil['Dry'] & self.rain['Low'] & (self.temp['Normal'] | self.temp['Cool']), self.pump['Medium']),

            # Nhóm 4: Kết hợp thị giác AI (Nguy cơ héo rũ cao -> Tăng cường tưới bù ẩm)
            ctrl.Rule(self.wilt['High'] & ~self.soil['Wet'] & self.rain['Low'], self.pump['Long']),
            ctrl.Rule(self.wilt['Medium'] & self.soil['Dry'], self.pump['Medium']),

            # Nhóm 5: Điều khiển quạt làm mát thông minh (Fan PWM)
            ctrl.Rule(self.temp['Cool'], self.fan['Off']),
            ctrl.Rule(self.temp['Normal'] & self.wilt['Low'], self.fan['Off']),
            ctrl.Rule(self.temp['Normal'] & (self.wilt['Medium'] | self.wilt['High']), self.fan['Low']),
            ctrl.Rule(self.temp['Hot'] & self.temp['Normal'], self.fan['Medium']),
            ctrl.Rule(self.temp['Hot'], self.fan['Max'])
        ]

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def evaluate(self, soil_pct: float, temp_c: float, rain_pct: float = 0.0, wilt_risk_pct: float = 0.0):
        """6. Hàm evaluate(): Đưa giá trị cảm biến vào tập luật và giải mờ (Centroid Defuzzification)"""
        # Giới hạn giá trị đầu vào hợp lệ
        s_in = np.clip(soil_pct, 0.0, 100.0)
        t_in = np.clip(temp_c, 15.0, 45.0)
        r_in = np.clip(rain_pct, 0.0, 100.0)
        w_in = np.clip(wilt_risk_pct, 0.0, 100.0)

        # Nạp giá trị vào hệ mờ
        self.sim.input['Soil'] = s_in
        self.sim.input['Temperature'] = t_in
        self.sim.input['Rain'] = r_in
        self.sim.input['WiltRisk'] = w_in

        # Tính toán suy luận mờ
        self.sim.compute()

        raw_pump = float(self.sim.output.get('PumpDuration', 0.0))
        raw_fan  = float(self.sim.output.get('FanPWM', 0.0))

        # Khóa cứng an toàn Dead-man và ngưỡng bảo vệ
        if s_in >= 70.0:  # Đất đủ ẩm
            pump_duration = 0.0
            decision = f"KHÔNG TƯỚI (Đất ẩm {s_in:.1f}% >= 70%)"
        elif r_in >= 65.0 and s_in >= 30.0:  # Dự báo mưa lớn
            pump_duration = 0.0
            decision = f"HOÃN TƯỚI (Dự báo mưa cao {r_in:.1f}%)"
        elif raw_pump < 1.0:
            pump_duration = 0.0
            decision = "KHÔNG TƯỚI (Đủ điều kiện ẩm)"
        else:
            pump_duration = round(min(12.0, raw_pump), 1)  # Giới hạn tối đa 12s
            decision = f"BẬT TƯỚI {pump_duration}s (Đất {s_in:.1f}%, Nhiệt {t_in:.1f}°C)"

        fan_speed = int(np.clip(round(raw_fan), 0, 255))
        if fan_speed < 40:
            fan_speed = 0

        return {
            "pump_duration_sec": pump_duration,
            "pump_active": pump_duration > 0,
            "fan_pwm": fan_speed,
            "fan_active": fan_speed > 0,
            "decision": decision,
            "inputs": {
                "soil": s_in, "temp": t_in, "rain": r_in, "wilt_risk": w_in
            }
        }

if __name__ == "__main__":
    engine = SmartChiliFuzzyEngine()
    print("Hệ thống suy luận mờ Smart Chili Fuzzy Engine sẵn sàng!")
    test = engine.evaluate(soil_pct=22.0, temp_c=34.5, rain_pct=10.0, wilt_risk_pct=75.0)
    print("Kết quả thử nghiệm:", test)
