import cv2
import numpy as np
import os
import time
import base64
import threading
import subprocess
from datetime import datetime

_cached_cameras = []
_last_scan_time = 0

def get_available_cameras(max_tested=2, force_refresh=False):
    """Quét và lấy danh sách tên các USB Camera đang cắm vào máy tính (DSHOW siêu tốc không xung đột handle)"""
    global _cached_cameras, _last_scan_time
    now = time.time()
    if not force_refresh and _cached_cameras and (now - _last_scan_time < 30):
        return _cached_cameras

    pnp_names = []
    try:
        cmd = 'Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName'
        res = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            pnp_names = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
    except Exception:
        pass

    cams = []
    if pnp_names:
        for i, name in enumerate(pnp_names):
            cams.append({'index': i, 'name': f'{name} (Cổng {i})', 'friendly_name': name})
    else:
        for i in range(max_tested):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cams.append({'index': i, 'name': f'USB Camera {i} (Cổng {i})', 'friendly_name': f'USB Camera {i}'})
                    cap.release()
            except Exception:
                pass

    if cams:
        _cached_cameras = cams
        _last_scan_time = now
    elif not _cached_cameras:
        _cached_cameras = [{'index': 1, 'name': 'USB2.0 HD UVC WebCam (Cổng 1)', 'friendly_name': 'USB2.0 HD UVC WebCam'}]
        _last_scan_time = now

    return _cached_cameras

class PlantVisionAnalyzer:
    CLASS_INFO = {
        0: {"name": "ot_do", "label": "Ot Do", "color": (20, 20, 240), "category": "fruit"},
        1: {"name": "ot_xanh", "label": "Ot Xanh", "color": (20, 220, 30), "category": "fruit"},
        2: {"name": "ot_hong", "label": "Ot Hong", "color": (0, 110, 220), "category": "fruit"},
        3: {"name": "la_xanh", "label": "La Xanh", "color": (40, 190, 40), "category": "leaf"},
        4: {"name": "la_heo", "label": "La Heo", "color": (0, 180, 240), "category": "leaf"},
        5: {"name": "la_benh", "label": "La Benh", "color": (0, 225, 255), "category": "leaf"}
    }

    def __init__(self, storage_dir="storage", camera_index=0):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.camera_index = int(camera_index)
        self.latest_path = os.path.join(self.storage_dir, "latest.jpg")
        
        self.lock = threading.Lock()
        self.running = True
        self.current_frame = None
        self.raw_frame = None
        self.latest_jpeg_bytes = None
        self.switch_requested = False
        self.new_camera_index = self.camera_index
        self.show_ai_overlay = True
        self.yolo_conf = 0.15
        self.yellow_alert_pct = 15.0

        # Nạp mô hình AI YOLOv8 đa phân lớp (6 lớp)
        self.yolo_model = None
        self.yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chili_yolov8.pt")
        self._load_yolo()
        
        self.last_analysis = {
            "status": "ONLINE",
            "leaf_health": "Đang quét cây ớt...",
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "red_chili_count": 0,
            "green_chili_count": 0,
            "damaged_chili_count": 0,
            "total_chili_count": 0,
            "healthy_leaf_count": 0,
            "wilting_leaf_count": 0,
            "diseased_leaf_count": 0,
            "total_leaf_count": 0,
            "camera_index": self.camera_index,
            "updated_at": datetime.now().strftime("%H:%M:%S")
        }
        
        self._create_placeholder_frame("Đang kết nối camera...")
        
        # Khởi chạy luồng camera nền liên tục (Zero Latency Video Thread)
        self.worker_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.worker_thread.start()

    def _load_yolo(self):
        """Nạp mô hình YOLOv8 nhận diện 6 lớp (3 lớp trái, 3 lớp lá)"""
        if os.path.exists(self.yolo_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.yolo_path)
            except Exception:
                self.yolo_model = None

    def _create_placeholder_frame(self, text="Không có tín hiệu Camera"):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (30, 30, 35)
        cv2.putText(img, "SMART CHILI AI - CAMERA", (580, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 200, 255), 3)
        cv2.putText(img, text, (680, 580), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (180, 180, 180), 2)
        ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with self.lock:
                self.current_frame = img
                self.latest_jpeg_bytes = buf.tobytes()

    def detect_and_analyze(self, frame):
        """
        Nhận diện và phân loại toàn diện 6 LỚP theo thiết kế đề tài SMART CHILI AI:
        - Trái ớt: ot_do (chín đỏ), ot_xanh (non xanh), ot_hong (thối/hỏng)
        - Trạng thái lá: la_xanh (lá xanh khỏe), la_heo (lá héo rũ), la_benh (lá vàng/đốm bệnh)
        - Đóng khung Bounding Box và Phân đoạn đối tượng (Instance Segmentation Mask)
        """
        if frame is None or frame.size == 0:
            return None, 0, 0, 0, 0, 0, 0, 0.0, 0.0, "Mất tín hiệu camera"

        h_img, w_img = frame.shape[:2]
        total_px = h_img * w_img
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        annotated = frame.copy()
        overlay = frame.copy()

        # Danh sách kết quả theo 6 lớp
        detected_objects = {i: [] for i in range(6)}

        # Tự động nạp lại YOLO nếu file mới được cập nhật
        if self.yolo_model is None and os.path.exists(self.yolo_path):
            self._load_yolo()

        # ================= 1. DỰ ĐOÁN VỚI MÔ HÌNH HỌC SÂU YOLOV8 (6 CLASSES) =================
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(frame, conf=self.yolo_conf, imgsz=320, verbose=False)
                has_seg_masks = hasattr(results[0], 'masks') and results[0].masks is not None
                for idx_b, b in enumerate(results[0].boxes):
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    if cls_id not in self.CLASS_INFO:
                        continue

                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    w, h = x2 - x1, y2 - y1

                    if w > 10 and h > 10:
                        poly = []
                        if has_seg_masks and idx_b < len(results[0].masks.xy):
                            m_pts = results[0].masks.xy[idx_b]
                            if len(m_pts) >= 3:
                                poly = [m_pts.astype(np.int32).reshape(-1, 1, 2)]

                        if not poly:
                            roi_hsv = hsv[y1:y2, x1:x2]
                            # Tạo mặt nạ segmentation bên trong bbox
                            if cls_id == 0:  # ot_do
                                m = cv2.inRange(roi_hsv, (0, 85, 60), (12, 255, 255)) | cv2.inRange(roi_hsv, (160, 85, 60), (180, 255, 255))
                            elif cls_id == 1:  # ot_xanh (trái ớt xanh thon dài)
                                m = cv2.inRange(roi_hsv, (34, 65, 45), (55, 255, 240))
                            elif cls_id == 2:  # ot_hong (ớt hư héo: có màu đen hoặc cam cam)
                                m = cv2.inRange(roi_hsv, (10, 50, 40), (26, 230, 200)) | cv2.inRange(roi_hsv, (0, 0, 10), (180, 255, 65))
                            elif cls_id == 5:  # la_benh (lá có chấm đen hoặc đốm vàng)
                                m = cv2.inRange(roi_hsv, (18, 60, 70), (34, 255, 255)) | cv2.inRange(roi_hsv, (0, 0, 0), (180, 255, 65))
                            else:  # la_xanh hoặc la_heo
                                m = cv2.inRange(roi_hsv, (36, 40, 35), (85, 255, 255))

                            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            poly = [c + np.array([[[x1, y1]]]) for c in cnts if cv2.contourArea(c) > 20]
                            if not poly:
                                poly = [np.array([[[x1, y1+h//2]], [[x1+w//2, y1]], [[x2, y1+h//2]], [[x1+w//2, y2]]])]

                        detected_objects[cls_id].append({
                            'bbox': (x1, y1, w, h),
                            'conf': conf,
                            'poly': poly
                        })
            except Exception:
                pass

        # ================= 1.5. HỖ TRỢ PHÂN ĐOẠN THỊ GIÁC TRỰC TIẾP NẾU YOLO CHƯA CÓ KẾT QUẢ =================
        if sum(len(v) for v in detected_objects.values()) == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Tách tán cây xanh
                foliage = cv2.inRange(hsv, np.array([32, 45, 35]), np.array([86, 255, 245]))
                foliage = cv2.morphologyEx(foliage, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                num_lbl, lbls, st, _ = cv2.connectedComponentsWithStats(foliage)
                plant_fol = np.zeros((h_img, w_img), dtype=np.uint8)
                for i in range(1, num_lbl):
                    if st[i, cv2.CC_STAT_AREA] > 100:
                        plant_fol[lbls == i] = 255
                canopy_zone = cv2.dilate(plant_fol, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
                occ = np.zeros((h_img, w_img), dtype=np.uint8)

                # 1. ot_do (Ớt chín đỏ)
                m_red = ((hsv[:, :, 0] <= 14) | (hsv[:, :, 0] >= 166)) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 40)
                red_c = cv2.bitwise_and(m_red.astype(np.uint8) * 255, canopy_zone)
                cnts_r, _ = cv2.findContours(red_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in sorted(cnts_r, key=cv2.contourArea, reverse=True):
                    a = cv2.contourArea(c)
                    if 100 < a < 4000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        hull = cv2.convexHull(c)
                        sol = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                        asp = max(float(bw)/bh, float(bh)/bw)
                        if asp >= 1.5 and sol >= 0.65 and bh < 120 and bw < 120:
                            detected_objects[0].append({'bbox': (bx, by, bw, bh), 'conf': 0.92, 'poly': [c]})
                            cv2.drawContours(occ, [c], -1, 255, -1)

                # 2. ot_xanh (Ớt chỉ thiên mọc thẳng đứng)
                m_lime = cv2.inRange(hsv, np.array([30, 45, 90]), np.array([52, 255, 255]))
                m_lime = cv2.bitwise_and(m_lime, canopy_zone)
                m_lime = cv2.bitwise_and(m_lime, cv2.bitwise_not(occ))
                k_p = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
                m_lime_cln = cv2.morphologyEx(m_lime, cv2.MORPH_OPEN, k_p)
                cnts_g, _ = cv2.findContours(m_lime_cln, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in sorted(cnts_g, key=cv2.contourArea, reverse=True):
                    a = cv2.contourArea(c)
                    if 110 < a < 4000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        hull = cv2.convexHull(c)
                        sol = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                        asp = float(bh) / max(1, bw)
                        if asp >= 1.55 and sol >= 0.68 and bh < 140 and bw < 80:
                            detected_objects[1].append({'bbox': (bx, by, bw, bh), 'conf': 0.90, 'poly': [c]})
                            cv2.drawContours(occ, [c], -1, 255, -1)

                # 3. ot_hong (Ớt hỏng/thối đen)
                dark_r = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 42]))
                dark_r = cv2.bitwise_and(dark_r, canopy_zone)
                dark_r = cv2.bitwise_and(dark_r, cv2.bitwise_not(occ))
                cnts_b, _ = cv2.findContours(dark_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in sorted(cnts_b, key=cv2.contourArea, reverse=True):
                    a = cv2.contourArea(c)
                    if 90 < a < 2000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        asp = max(float(bw)/bh, float(bh)/bw)
                        hull = cv2.convexHull(c)
                        sol = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                        if asp >= 1.5 and sol >= 0.68 and bh < 80 and bw < 50:
                            detected_objects[2].append({'bbox': (bx, by, bw, bh), 'conf': 0.85, 'poly': [c]})
                            cv2.drawContours(occ, [c], -1, 255, -1)
                            if len(detected_objects[2]) >= 3:
                                break

                # 4. la_heo (Lá vàng úa)
                m_ylw = cv2.inRange(hsv, np.array([18, 50, 65]), np.array([33, 255, 255]))
                m_ylw = cv2.bitwise_and(m_ylw, canopy_zone)
                m_ylw = cv2.bitwise_and(m_ylw, cv2.bitwise_not(occ))
                cnts_y, _ = cv2.findContours(m_ylw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in sorted(cnts_y, key=cv2.contourArea, reverse=True):
                    a = cv2.contourArea(c)
                    if 250 < a < 8000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        asp = max(float(bw)/bh, float(bh)/bw)
                        hull = cv2.convexHull(c)
                        sol = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                        if 1.05 <= asp <= 3.2 and sol >= 0.45:
                            detected_objects[4].append({'bbox': (bx, by, bw, bh), 'conf': 0.88, 'poly': [c]})
                            cv2.drawContours(occ, [c], -1, 255, -1)

                # 5. Phiến lá xanh: la_benh (có chấm đen) & la_xanh
                edg = cv2.dilate(cv2.Canny(gray, 30, 90), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                l_split = cv2.bitwise_and(plant_fol, cv2.bitwise_not(edg))
                l_split = cv2.bitwise_and(l_split, cv2.bitwise_not(occ))
                cnts_l, _ = cv2.findContours(l_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                dark_fol = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 0), (180, 255, 48)), plant_fol)
                for c in cnts_l:
                    a = cv2.contourArea(c)
                    if 300 < a < 12000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        asp = max(float(bw)/bh, float(bh)/bw)
                        hull = cv2.convexHull(c)
                        sol = a / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                        if 1.05 <= asp <= 3.2 and sol >= 0.42:
                            c_m = np.zeros((h_img, w_img), dtype=np.uint8)
                            cv2.drawContours(c_m, [c], -1, 255, -1)
                            d_cnt = np.sum(cv2.bitwise_and(dark_fol, c_m) > 0)
                            cls_target = 5 if d_cnt >= 8 else 3
                            detected_objects[cls_target].append({'bbox': (bx, by, bw, bh), 'conf': 0.86, 'poly': [c]})
                            cv2.drawContours(occ, [c], -1, 255, -1)
            except Exception:
                pass

        # ================= 2. ĐÁNH GIÁ SỨC KHỎE TÁN LÁ TOÀN CẢNH =================
        all_green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([34, 255, 255]))

        green_pct = round((cv2.countNonZero(all_green_mask) / total_px) * 100, 1)
        yellow_pct = round((cv2.countNonZero(yellow_mask) / total_px) * 100, 1)

        # Số lượng từng loại
        n_red = len(detected_objects[0])
        n_green = len(detected_objects[1])
        n_rot = len(detected_objects[2])
        n_total_fruit = n_red + n_green + n_rot

        n_leaf_healthy = len(detected_objects[3])
        n_leaf_wilt = len(detected_objects[4])
        n_leaf_dis = len(detected_objects[5])
        n_total_leaf = n_leaf_healthy + n_leaf_wilt + n_leaf_dis

        # Chẩn đoán tự động thông minh
        if n_leaf_wilt > 0:
            health = "CẢNH BÁO: Phát hiện lá héo rũ (Cây thiếu nước)"
        elif n_leaf_dis > 0:
            health = "CẢNH BÁO: Nghi ngờ lá bệnh / vàng lá"
        elif n_rot > 0:
            health = "CHÚ Ý: Phát hiện quả ớt thối/hỏng cần cắt tỉa"
        elif yellow_pct > self.yellow_alert_pct:
            health = f"Cảnh báo: Tỷ lệ vàng lá cao (>{self.yellow_alert_pct}%)"
        elif green_pct < 1.0:
            health = "Chưa phát hiện tán lá rõ rệt"
        else:
            health = "Binh thuong (Cay khoe, dang ra qua tot)"

        # ================= 3. NẾU TẮT OVERLAY -> TRẢ VỀ CAMERA MỘC SẠCH 100% =================
        if not self.show_ai_overlay:
            return (frame, n_red, n_green, n_rot, n_leaf_healthy, n_leaf_wilt, n_leaf_dis,
                    green_pct, yellow_pct, "Camera truc tiep (Tat overlay)")

        # ================= 4. VẼ PHÂN VÙNG AI (KHÔNG CHỮ THEO YÊU CẦU: "trên màn hình đó kh có chữ gì hết") =================
        # Mặt nạ bán trong suốt cho từng đối tượng nhận diện
        for cls_id, objs in detected_objects.items():
            color = self.CLASS_INFO[cls_id]["color"]
            for obj in objs:
                for p in obj['poly']:
                    cv2.drawContours(overlay, [p], -1, color, -1)

        annotated = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

        # Vẽ đường viền đa giác bám sát mép quả và lá (tuyệt đối KHÔNG chèn chữ, KHÔNG chèn banner đè màn hình)
        for cls_id, objs in detected_objects.items():
            color = self.CLASS_INFO[cls_id]["color"]
            for obj in objs:
                for p in obj['poly']:
                    cv2.drawContours(annotated, [p], -1, color, 2)

        return (annotated, n_red, n_green, n_rot, n_leaf_healthy, n_leaf_wilt, n_leaf_dis,
                green_pct, yellow_pct, health)

    def _camera_worker(self):
        """Worker chạy liên tục ghi nhận khung hình thời gian thực (25-30 FPS) tự phục hồi lỗi"""
        cap = None
        last_save_time = 0
        consecutive_fails = 0

        while self.running:
            if self.switch_requested:
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                with self.lock:
                    self.camera_index = self.new_camera_index
                    self.switch_requested = False
                consecutive_fails = 0

            if cap is None or not cap.isOpened():
                try:
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        consecutive_fails = 0
                except Exception:
                    cap = None

            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    consecutive_fails = 0
                    clean_raw = frame.copy()
                    
                    # Phân tích AI và gắn nhãn theo thời gian thực
                    ann_frame, n_red, n_green, n_rot, n_leaf_h, n_leaf_w, n_leaf_d, green_pct, yellow_pct, health = self.detect_and_analyze(frame)
                    if ann_frame is None:
                        ann_frame = clean_raw

                    ret_enc, buf = cv2.imencode('.jpg', ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ret_enc:
                        with self.lock:
                            self.raw_frame = clean_raw
                            self.current_frame = ann_frame
                            self.latest_jpeg_bytes = buf.tobytes()
                            self.last_analysis = {
                                "status": "ONLINE",
                                "leaf_health": health,
                                "green_ratio": green_pct,
                                "yellow_ratio": yellow_pct,
                                "red_chili_count": n_red,
                                "green_chili_count": n_green,
                                "damaged_chili_count": n_rot,
                                "total_chili_count": n_red + n_green + n_rot,
                                "healthy_leaf_count": n_leaf_h,
                                "wilting_leaf_count": n_leaf_w,
                                "diseased_leaf_count": n_leaf_d,
                                "total_leaf_count": n_leaf_h + n_leaf_w + n_leaf_d,
                                "camera_index": self.camera_index,
                                "updated_at": datetime.now().strftime("%H:%M:%S")
                            }

                    # Định kỳ 10 giây lưu 1 ảnh snapshot mộc không bị đè khung
                    now = time.time()
                    if now - last_save_time >= 10:
                        try:
                            cv2.imwrite(self.latest_path, clean_raw)
                            last_save_time = now
                        except Exception:
                            pass
                else:
                    consecutive_fails += 1
                    if consecutive_fails > 15:
                        if cap:
                            try:
                                cap.release()
                            except Exception:
                                pass
                        cap = None
                        consecutive_fails = 0
                        time.sleep(1.0)
            else:
                time.sleep(0.5)

            time.sleep(0.015)

        if cap:
            try:
                cap.release()
            except Exception:
                pass

    def toggle_overlay(self):
        with self.lock:
            self.show_ai_overlay = not self.show_ai_overlay
            return self.show_ai_overlay

    def switch_camera(self, new_index: int):
        return self.set_camera_index(new_index)

    def set_camera_index(self, new_index: int):
        with self.lock:
            self.new_camera_index = int(new_index)
            self.switch_requested = True
        return {"status": "ok", "new_index": new_index}

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg_bytes

    def get_raw_frame(self):
        with self.lock:
            if self.raw_frame is not None:
                return self.raw_frame.copy()
            elif self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def get_jpeg_bytes(self):
        with self.lock:
            return self.latest_jpeg_bytes

    def capture_and_analyze(self):
        return self.analyze_current()

    def get_latest_analysis(self):
        with self.lock:
            return dict(self.last_analysis)

    def analyze_current(self):
        with self.lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None

        if frame is not None:
            ann_frame, n_red, n_green, n_rot, n_leaf_h, n_leaf_w, n_leaf_d, green_pct, yellow_pct, health = self.detect_and_analyze(frame)
            if ann_frame is None:
                ann_frame = frame
            try:
                cv2.imwrite(self.latest_path, ann_frame)
            except Exception:
                pass
            with self.lock:
                self.last_analysis = {
                    "status": "ONLINE",
                    "leaf_health": health,
                    "green_ratio": green_pct,
                    "yellow_ratio": yellow_pct,
                    "red_chili_count": n_red,
                    "green_chili_count": n_green,
                    "damaged_chili_count": n_rot,
                    "total_chili_count": n_red + n_green + n_rot,
                    "healthy_leaf_count": n_leaf_h,
                    "wilting_leaf_count": n_leaf_w,
                    "diseased_leaf_count": n_leaf_d,
                    "total_leaf_count": n_leaf_h + n_leaf_w + n_leaf_d,
                    "camera_index": self.camera_index,
                    "updated_at": datetime.now().strftime("%H:%M:%S")
                }
        return self.get_latest_analysis()

    def analyze_base64(self, b64_str: str):
        """Xử lý ảnh Base64 từ ESP32-CAM nếu có"""
        try:
            img_data = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                ann_frame, n_red, n_green, n_rot, n_leaf_h, n_leaf_w, n_leaf_d, green_pct, yellow_pct, health = self.detect_and_analyze(frame)
                if ann_frame is None:
                    ann_frame = frame
                with self.lock:
                    self.current_frame = frame
                    ret_enc, buf = cv2.imencode('.jpg', ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 74])
                    if ret_enc:
                        self.latest_jpeg_bytes = buf.tobytes()
                    self.last_analysis = {
                        "status": "ONLINE",
                        "leaf_health": health,
                        "green_ratio": green_pct,
                        "yellow_ratio": yellow_pct,
                        "red_chili_count": n_red,
                        "green_chili_count": n_green,
                        "damaged_chili_count": n_rot,
                        "total_chili_count": n_red + n_green + n_rot,
                        "healthy_leaf_count": n_leaf_h,
                        "wilting_leaf_count": n_leaf_w,
                        "diseased_leaf_count": n_leaf_d,
                        "total_leaf_count": n_leaf_h + n_leaf_w + n_leaf_d,
                        "camera_index": self.camera_index,
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                        "source": "ESP32-CAM"
                    }
                cv2.imwrite(self.latest_path, ann_frame)
                return self.get_latest_analysis()
        except Exception:
            pass
        return self.get_latest_analysis()

    def stop(self):
        self.running = False
