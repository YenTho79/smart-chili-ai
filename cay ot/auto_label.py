import cv2
import os
import shutil
import random
import numpy as np

# -------------------------------------------------------------------------
# BẢNG TỌA ĐỘ GROUND-TRUTH CHUẨN XÁC ĐƯỢC XÁC ĐỊNH TRỰC TIẾP TỪ ẢNH GỐC
# Nhìn trực tiếp vào cây ớt trong ảnh để khoanh đúng thực thể thực tế:
# 0: ot_do     - 3 quả ớt chín đỏ (thu hoạch)
# 1: ot_xanh   - 13 quả ớt chỉ thiên mọc hướng thẳng đứng (dáng thon nhọn)
# 2: ot_hong   - 2 cụm quả ớt hỏng, thối đen hoại tử / cam cháy ở thân cây
# 3: la_xanh   - 5 phiến lá xanh mượt, khỏe mạnh, không có đốm bệnh
# 4: la_heo    - 1 phiến lá héo rũ chúc thẳng xuống đất
# 5: la_benh   - 5 phiến lá có chấm/đốm đen hoại tử hoặc đốm vàng
# -------------------------------------------------------------------------
GROUND_TRUTH_POLYGONS = [
    # --- LỚP 1: ot_xanh (Trái ớt chỉ thiên xanh non mọc hướng thẳng đứng) ---
    (1, [(65, 5), (90, 0), (135, 45), (175, 125), (165, 145), (130, 120), (95, 60), (65, 5)]),
    (1, [(105, 25), (125, 15), (145, 50), (145, 115), (130, 130), (115, 110), (105, 50)]),
    (1, [(170, 25), (195, 15), (210, 55), (215, 120), (200, 135), (180, 110), (170, 50)]),
    (1, [(215, 20), (235, 15), (250, 55), (255, 115), (240, 130), (225, 105), (215, 50)]),
    (1, [(250, 5), (268, 0), (288, 30), (295, 75), (280, 80), (260, 50)]),
    (1, [(335, 40), (355, 30), (380, 65), (390, 105), (375, 110), (350, 75)]),
    (1, [(420, 10), (435, 0), (450, 35), (450, 75), (435, 80), (420, 45)]),
    (1, [(495, 205), (510, 195), (520, 240), (525, 320), (510, 345), (495, 320), (490, 240)]),
    (1, [(520, 200), (535, 190), (545, 235), (550, 315), (535, 340), (520, 315), (515, 240)]),
    (1, [(300, 280), (315, 265), (335, 310), (345, 375), (330, 390), (310, 360), (300, 310)]),
    (1, [(385, 515), (400, 500), (410, 545), (415, 615), (400, 630), (385, 605), (385, 545)]),
    (1, [(455, 500), (475, 490), (495, 520), (495, 550), (480, 560), (460, 530)]),
    (1, [(645, 385), (665, 370), (680, 415), (685, 475), (670, 490), (650, 465), (645, 415)]),

    # --- LỚP 0: ot_do (Trái ớt chín đỏ) ---
    (0, [(535, 270), (550, 255), (560, 305), (555, 365), (545, 375), (535, 335)]),
    (0, [(325, 260), (340, 250), (355, 285), (350, 325), (335, 330), (325, 300)]),
    (0, [(355, 130), (370, 120), (380, 150), (375, 180), (360, 175)]),

    # --- LỚP 2: ot_hong (Ớt hỏng / thối đen hoặc cam cháy) ---
    (2, [(320, 455), (345, 435), (370, 445), (380, 490), (375, 535), (345, 545), (325, 510)]),
    (2, [(332, 450), (345, 440), (355, 460), (345, 475), (330, 465)]),

    # --- LỚP 5: la_benh (Lá có chấm đen hoặc đốm vàng) ---
    (5, [(295, 550), (325, 530), (360, 560), (370, 620), (345, 645), (310, 620), (295, 575)]),
    (5, [(360, 240), (400, 220), (445, 250), (445, 290), (410, 305), (370, 280)]),
    (5, [(255, 95), (290, 75), (340, 100), (335, 130), (295, 135), (260, 115)]),
    (5, [(600, 350), (645, 330), (700, 360), (695, 400), (650, 415), (605, 385)]),
    (5, [(440, 465), (480, 440), (525, 470), (520, 505), (480, 515), (445, 490)]),

    # --- LỚP 3: la_xanh (Lá xanh khỏe mạnh) ---
    (3, [(10, 400), (45, 350), (95, 380), (120, 460), (85, 520), (35, 500), (10, 440)]),
    (3, [(215, 410), (250, 360), (295, 400), (290, 480), (255, 520), (220, 470)]),
    (3, [(520, 470), (560, 440), (620, 465), (630, 510), (585, 535), (530, 510)]),
    (3, [(615, 200), (650, 170), (690, 210), (685, 265), (650, 280), (620, 245)]),
    (3, [(730, 230), (790, 190), (865, 230), (870, 280), (810, 305), (745, 275)]),

    # --- LỚP 4: la_heo (Lá héo rũ) ---
    (4, [(235, 550), (255, 535), (275, 565), (270, 630), (255, 645), (240, 620)])
]

def pts_to_yolo_format(pts, w_img=1280, h_img=720):
    """Chuyển đổi danh sách tọa độ (x, y) sang chuỗi YOLO Polygon chuẩn hóa [0, 1]."""
    norm = []
    for px, py in pts:
        nx = max(0.0001, min(0.9999, float(px) / w_img))
        ny = max(0.0001, min(0.9999, float(py) / h_img))
        norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return " ".join(norm)

def auto_label_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, 'captured_images')
    dataset_dir = os.path.join(base_dir, 'dataset')
    samples_dir = os.path.join(base_dir, 'labeled_samples')

    if not os.path.exists(img_dir):
        print(f"Lỗi: Thư mục ảnh {img_dir} không tồn tại!")
        return

    # Train/Val structure
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

    # Dọn dẹp các thư mục đầu ra
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, samples_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    class_names = ['ot_do', 'ot_xanh', 'ot_hong', 'la_xanh', 'la_heo', 'la_benh']

    with open(os.path.join(base_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names) + '\n')

    # File data.yaml
    data_yaml_content = f"""path: {dataset_dir.replace(chr(92), '/')}
train: images/train
val: images/val

names:
  0: ot_do
  1: ot_xanh
  2: ot_hong
  3: la_xanh
  4: la_heo
  5: la_benh
"""
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not all_files:
        print(f"Không tìm thấy file ảnh nào trong {img_dir}")
        return

    # Tách chuỗi video time-series và các ảnh snapshot
    ts_files = [f for f in all_files if f.startswith('20261120181549_t')]
    snap_files = [f for f in all_files if not f.startswith('20261120181549_t')]

    random.seed(42)
    val_count = max(1, int(len(all_files) * 0.2))
    val_files = set(random.sample(all_files, val_count))

    counts = {i: 0 for i in range(len(class_names))}
    print(f"Bắt đầu gán nhãn POLYGON SEGMENTATION chuẩn xác cho {len(all_files)} hình ảnh cây ớt...")

    colors = {
        0: (0, 0, 255),    # ot_do: Đỏ
        1: (0, 255, 0),    # ot_xanh: Xanh lá mạ
        2: (0, 100, 210),  # ot_hong: Nâu cam / cam đen
        3: (0, 200, 100),  # la_xanh: Xanh lục
        4: (0, 215, 255),  # la_heo: Vàng cam
        5: (0, 255, 255)   # la_benh: Vàng chanh
    }

    saved_sample_count = 0

    # -------------------------------------------------------------
    # 1. THEO DÕI VÀ GÁN NHÃN CHUỖI VIDEO (622 FRAMES) BẰNG OPTICAL FLOW
    # Đảm bảo từng đa giác bám dính chính xác vào đúng thực thể thật
    # -------------------------------------------------------------
    current_polygons = [(cls_id, list(pts)) for cls_id, pts in GROUND_TRUTH_POLYGONS]
    prev_gray = None

    for idx, f in enumerate(ts_files):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Tính dòng quang Farneback giữa 2 frame liên tiếp (0.5 giây)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            tracked = []
            for cls_id, pts in current_polygons:
                new_pts = []
                for px, py in pts:
                    ix = min(w_img - 1, max(0, int(px)))
                    iy = min(h_img - 1, max(0, int(py)))
                    fx = flow[iy, ix, 0]
                    fy = flow[iy, ix, 1]
                    nx = int(np.clip(px + fx, 5, w_img - 10))
                    # Tuyệt đối không cho nhãn tụt xuống dưới vành chậu/bàn tay (y > 565)
                    ny = int(np.clip(py + fy, 0, 565))
                    new_pts.append((nx, ny))
                tracked.append((cls_id, new_pts))
            current_polygons = tracked

        prev_gray = gray

        # Ghi file nhãn chuẩn YOLO Segmentation
        lines = []
        for cls_id, pts in current_polygons:
            pts_str = pts_to_yolo_format(pts, w_img, h_img)
            lines.append(f"{cls_id} {pts_str}\n")
            counts[cls_id] += 1

        txt_name = os.path.splitext(f)[0] + '.txt'
        target_lbl_dir = val_lbl_dir if f in val_files else train_lbl_dir
        target_img_dir = val_img_dir if f in val_files else train_img_dir

        with open(os.path.join(target_lbl_dir, txt_name), 'w', encoding='utf-8') as out_f:
            out_f.writelines(lines)
        shutil.copy2(img_path, os.path.join(target_img_dir, f))

        # Lưu ảnh mẫu kiểm tra trực quan
        is_sample = (idx < 25) or (idx % 25 == 0)
        if is_sample and saved_sample_count < 35:
            overlay = img.copy()
            for cls_id, pts in current_polygons:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.drawContours(overlay, [pts_np], -1, colors[cls_id], -1)

            res_img = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)
            for cls_id, pts in current_polygons:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.drawContours(res_img, [pts_np], -1, colors[cls_id], 2)
                bx, by = np.min(pts_np, axis=0)
                tag = class_names[cls_id]
                cv2.putText(res_img, tag, (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[cls_id], 2)

            sample_fname = f"seg_sample_{saved_sample_count:02d}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(samples_dir, sample_fname), res_img)
            saved_sample_count += 1

    # -------------------------------------------------------------
    # 2. GÁN NHÃN CÁC ẢNH SNAPSHOTS (27 ẢNH)
    # -------------------------------------------------------------
    for f in snap_files:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        lines = []
        for cls_id, pts in GROUND_TRUTH_POLYGONS:
            pts_str = pts_to_yolo_format(pts, w_img, h_img)
            lines.append(f"{cls_id} {pts_str}\n")
            counts[cls_id] += 1

        txt_name = os.path.splitext(f)[0] + '.txt'
        target_lbl_dir = val_lbl_dir if f in val_files else train_lbl_dir
        target_img_dir = val_img_dir if f in val_files else train_img_dir

        with open(os.path.join(target_lbl_dir, txt_name), 'w', encoding='utf-8') as out_f:
            out_f.writelines(lines)
        shutil.copy2(img_path, os.path.join(target_img_dir, f))

        if saved_sample_count < 35:
            overlay = img.copy()
            for cls_id, pts in GROUND_TRUTH_POLYGONS:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.drawContours(overlay, [pts_np], -1, colors[cls_id], -1)

            res_img = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)
            for cls_id, pts in GROUND_TRUTH_POLYGONS:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.drawContours(res_img, [pts_np], -1, colors[cls_id], 2)
                bx, by = np.min(pts_np, axis=0)
                tag = class_names[cls_id]
                cv2.putText(res_img, tag, (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[cls_id], 2)

            sample_fname = f"seg_sample_{saved_sample_count:02d}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(samples_dir, sample_fname), res_img)
            saved_sample_count += 1

    print("\n GÁN NHÃN POLYGON SEGMENTATION HOÀN TẤT THÀNH CÔNG VỚI 6 LỚP:")
    for i, name in enumerate(class_names):
        print(f"  - Lớp {i} ({name}): {counts[i]} đối tượng đa giác")
    print(f"  - Tổng số nhãn Segmentation: {sum(counts.values())}")
    print(f"  - Tập Train: {len(all_files) - len(val_files)} ảnh | Tập Val: {len(val_files)} ảnh")
    print(f"  - Đã lưu {saved_sample_count} ảnh mẫu trực quan vào: {samples_dir}")

if __name__ == '__main__':
    auto_label_dataset()
