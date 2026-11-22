import cv2
import os
import shutil
import random
import numpy as np

def auto_label_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, 'captured_images')
    labels_dir = os.path.join(base_dir, 'labels')
    dataset_dir = os.path.join(base_dir, 'dataset')
    samples_dir = os.path.join(base_dir, 'labeled_samples')

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Train/Val structure
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

    # Xóa sạch dữ liệu cũ để tránh nhầm lẫn nhãn
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 6 Lớp chuẩn theo thiết kế đề tài SMART CHILI AI:
    # 0: ot_do     - Trái ớt chín đỏ
    # 1: ot_xanh   - Trái ớt non xanh
    # 2: ot_hong   - Trái ớt hỏng / thối / hư
    # 3: la_xanh   - Lá xanh khỏe mạnh
    # 4: la_heo    - Lá héo rũ (thiếu nước)
    # 5: la_benh   - Lá bệnh / vàng lá / đốm đốm
    class_names = ['ot_do', 'ot_xanh', 'ot_hong', 'la_xanh', 'la_heo', 'la_benh']

    with open(os.path.join(base_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names) + '\n')

    # Ghi file cấu hình data.yaml cho YOLOv8
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

    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    random.seed(42)
    val_files = set(random.sample(files, int(len(files) * 0.2)))

    counts = {i: 0 for i in range(len(class_names))}

    print(f"Đang tiến hành gán nhãn 6 LỚP cho {len(files)} hình ảnh cây ớt...")

    for idx, f in enumerate(files):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        boxes_this_img = []

        # -------------------------------------------------------------
        # 1. CLASS 0: ot_do (Ớt chín đỏ)
        # -------------------------------------------------------------
        m1 = cv2.inRange(hsv, np.array([0, 115, 80]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([165, 115, 80]), np.array([180, 255, 255]))
        red_mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_boxes = []
        for c in cnts_red:
            area = cv2.contourArea(c)
            if 250 < area < 40000 and len(red_boxes) < 4:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if 1.05 <= aspect <= 7.0 and w > 14 and h > 14:
                    if 15 < x < (w_img-15) and 15 < y < (h_img-15):
                        red_boxes.append((x, y, w, h))
                        boxes_this_img.append((0, x, y, w, h))
                        counts[0] += 1

        # -------------------------------------------------------------
        # 2. CLASS 1: ot_xanh (Ớt non xanh - tiêu chuẩn ống dài, đặc)
        # -------------------------------------------------------------
        green_mask = cv2.inRange(hsv, np.array([36, 85, 45]), np.array([82, 255, 230]))
        k_green = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k_green)
        cnts_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_boxes = []
        for c in cnts_green:
            area = cv2.contourArea(c)
            if 400 < area < 30000 and len(green_boxes) < 3:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                hull = cv2.convexHull(c)
                solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                if aspect >= 2.2 and solidity >= 0.72:
                    overlap_red = any(abs(x - rx) < 30 and abs(y - ry) < 30 for rx, ry, rw, rh in red_boxes)
                    if not overlap_red and 20 < x < (w_img-20) and 20 < y < (h_img-20):
                        green_boxes.append((x, y, w, h))
                        boxes_this_img.append((1, x, y, w, h))
                        counts[1] += 1

        # -------------------------------------------------------------
        # 3. CLASS 2: ot_hong (Ớt hỏng / thối / đốm hư đen nâu)
        # -------------------------------------------------------------
        rot_mask = cv2.inRange(hsv, np.array([6, 60, 25]), np.array([24, 210, 110]))
        k_rot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rot_mask = cv2.morphologyEx(rot_mask, cv2.MORPH_OPEN, k_rot)
        cnts_rot, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rot_count = 0
        for c in cnts_rot:
            area = cv2.contourArea(c)
            if 350 < area < 15000 and rot_count < 2:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.6 and 20 < x < (w_img-20) and 20 < y < (h_img-20):
                    rot_count += 1
                    boxes_this_img.append((2, x, y, w, h))
                    counts[2] += 1

        # -------------------------------------------------------------
        # 4. CLASS 5: la_benh (Lá bệnh / vàng lá / đốm đốm)
        # -------------------------------------------------------------
        yel_mask = cv2.inRange(hsv, np.array([19, 75, 85]), np.array([32, 255, 240]))
        k_yel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yel_mask = cv2.morphologyEx(yel_mask, cv2.MORPH_OPEN, k_yel)
        cnts_yel, _ = cv2.findContours(yel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        benh_count = 0
        for c in cnts_yel:
            area = cv2.contourArea(c)
            if 500 < area < 25000 and benh_count < 2:
                x, y, w, h = cv2.boundingRect(c)
                if 25 < x < (w_img-25) and 25 < y < (h_img-25):
                    benh_count += 1
                    boxes_this_img.append((5, x, y, w, h))
                    counts[5] += 1

        # -------------------------------------------------------------
        # 5. CLASS 3 & 4: la_xanh (Lá xanh) & la_heo (Lá héo rũ)
        # -------------------------------------------------------------
        leaf_mask = cv2.inRange(hsv, np.array([35, 45, 40]), np.array([82, 255, 255]))
        k_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k_leaf)
        cnts_leaf, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_count = 0
        for c in cnts_leaf:
            area = cv2.contourArea(c)
            if 1500 < area < 40000 and leaf_count < 3:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                hull = cv2.convexHull(c)
                solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                if 1.15 <= aspect < 2.2 and 0.55 <= solidity <= 0.88:
                    if 20 < x < (w_img-20) and 20 < y < (h_img-20):
                        leaf_count += 1
                        if h > w * 1.55:
                            # Lá rũ thẳng xuống (héo rũ)
                            boxes_this_img.append((4, x, y, w, h))
                            counts[4] += 1
                        else:
                            # Lá xanh bình thường
                            boxes_this_img.append((3, x, y, w, h))
                            counts[3] += 1

        # Ghi file nhãn định dạng YOLO (Normalized coordinates)
        lines = []
        for cls_id, x, y, w, h in boxes_this_img:
            x_c = (x + w / 2.0) / w_img
            y_c = (y + h / 2.0) / h_img
            w_norm = w / float(w_img)
            h_norm = h / float(h_img)
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Lưu nhãn vào dataset train/val
        txt_name = f.replace('.jpg', '.txt')
        target_dir = val_lbl_dir if f in val_files else train_lbl_dir
        target_img_dir = val_img_dir if f in val_files else train_img_dir

        with open(os.path.join(target_dir, txt_name), 'w') as out_f:
            out_f.writelines(lines)
        shutil.copy2(img_path, os.path.join(target_img_dir, f))

        # Lưu mẫu vẽ để kiểm tra trực quan
        if idx % 100 == 0:
            sample_img = img.copy()
            colors = {
                0: (0, 0, 255),    # ot_do: Đỏ
                1: (0, 255, 0),    # ot_xanh: Xanh lá
                2: (0, 100, 200),  # ot_hong: Nâu cam
                3: (0, 200, 100),  # la_xanh: Xanh ngọc
                4: (0, 215, 255),  # la_heo: Vàng cam
                5: (0, 255, 255)   # la_benh: Vàng chanh
            }
            for cls_id, x, y, w, h in boxes_this_img:
                c = colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(sample_img, (x, y), (x+w, y+h), c, 2)
                cv2.putText(sample_img, class_names[cls_id], (x, max(20, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            cv2.imwrite(os.path.join(samples_dir, f"sample_{idx:03d}.jpg"), sample_img)

    print("\n GÁN NHÃN HOÀN TẤT THÀNH CÔNG VỚI 6 LỚP:")
    for i, name in enumerate(class_names):
        print(f"  - Lớp {i} ({name}): {counts[i]} đối tượng")
    print(f"  - Tổng số nhãn: {sum(counts.values())}")
    print(f"  - Tập Train: {len(files) - len(val_files)} ảnh | Tập Val: {len(val_files)} ảnh")

if __name__ == '__main__':
    auto_label_dataset()
