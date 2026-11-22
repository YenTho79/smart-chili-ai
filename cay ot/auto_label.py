import cv2
import os
import shutil
import random
import numpy as np

def contour_to_polygon(cnt, w_img, h_img, max_pts=16, min_pts=4):
    """
    Chuyển đổi contour OpenCV thành đa giác (polygon) chuẩn YOLO Segmentation:
    Trả về danh sách tọa độ chuẩn hóa [x1, y1, x2, y2, ..., xn, yn] trong đoạn [0, 1]
    và mảng điểm nguyên gốc (pixel) để vẽ trực quan.
    """
    peri = cv2.arcLength(cnt, True)
    if peri < 12:
        return None, None
    approx = None
    for eps in [0.008, 0.012, 0.016, 0.022, 0.03]:
        a = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(a) <= max_pts:
            approx = a
            break
    if approx is None:
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
    if len(approx) < min_pts:
        approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
    if len(approx) < 3:
        return None, None

    pts = approx.reshape(-1, 2)
    norm = []
    for p in pts:
        nx = max(0.0001, min(0.9999, float(p[0]) / w_img))
        ny = max(0.0001, min(0.9999, float(p[1]) / h_img))
        norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return norm, pts

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

    # Dọn dẹp các thư mục đầu ra để tái tạo nhãn chuẩn mới nhất
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, samples_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 6 Lớp chuẩn theo thiết kế đề tài SMART CHILI AI:
    # 0: ot_do     - Trái ớt chín đỏ (thu hoạch)
    # 1: ot_xanh   - Trái ớt non xanh (hình quả thon dài, bóng mịn)
    # 2: ot_hong   - Trái ớt hỏng / thối / hư (có màu đen hoặc cam cam)
    # 3: la_xanh   - Lá xanh khỏe mạnh (hình phiến lá)
    # 4: la_heo    - Lá héo rũ (rũ thẳng xuống, thiếu nước)
    # 5: la_benh   - Lá bệnh (lá có chấm đen hoặc đốm vàng)
    class_names = ['ot_do', 'ot_xanh', 'ot_hong', 'la_xanh', 'la_heo', 'la_benh']

    with open(os.path.join(base_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names) + '\n')

    # File cấu hình data.yaml cho YOLOv8 Segmentation
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

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        print(f"Không tìm thấy file ảnh nào trong {img_dir}")
        return

    random.seed(42)
    val_count = max(1, int(len(files) * 0.2))
    val_files = set(random.sample(files, val_count))

    counts = {i: 0 for i in range(len(class_names))}
    print(f"Đang tiến hành gán nhãn POLYGON SEGMENTATION cho {len(files)} hình ảnh cây ớt...")

    colors = {
        0: (0, 0, 255),    # ot_do: Đỏ
        1: (0, 255, 0),    # ot_xanh: Xanh lá cây
        2: (0, 100, 210),  # ot_hong: Nâu cam / cam đen
        3: (0, 200, 100),  # la_xanh: Xanh ngọc
        4: (0, 215, 255),  # la_heo: Vàng cam
        5: (0, 255, 255)   # la_benh: Vàng chanh
    }

    saved_sample_count = 0

    for idx, f in enumerate(files):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        polys_this_img = []  # danh sách (cls_id, norm_pts, pts_pixel)

        # -------------------------------------------------------------
        # 1. CLASS 0: ot_do (Ớt chín đỏ)
        # -------------------------------------------------------------
        m1 = cv2.inRange(hsv, np.array([0, 110, 75]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([165, 110, 75]), np.array([180, 255, 255]))
        red_mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_boxes = []
        red_count = 0
        for c in cnts_red:
            area = cv2.contourArea(c)
            if 250 < area < 45000 and red_count < 4:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.15 and w > 12 and h > 12:
                    if 10 < x < (w_img - 10) and 10 < y < (h_img - 10):
                        norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                        if norm_pts:
                            polys_this_img.append((0, norm_pts, pts_pix))
                            red_boxes.append((x, y, w, h))
                            counts[0] += 1
                            red_count += 1

        # -------------------------------------------------------------
        # 2. CLASS 2: ot_hong (Ớt hư héo có màu đen hoặc cam cam)
        # -------------------------------------------------------------
        # Ớt cam cam héo / cháy nắng
        m_orange = cv2.inRange(hsv, np.array([10, 55, 45]), np.array([26, 230, 200]))
        # Ớt thối đen / hoại tử đen
        m_black = cv2.inRange(hsv, np.array([0, 0, 15]), np.array([180, 255, 65]))
        m_black_fruit = m_black & cv2.inRange(hsv, np.array([5, 30, 15]), np.array([35, 255, 80]))
        rot_mask = cv2.morphologyEx(m_orange | m_black_fruit, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_rot, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rot_boxes = []
        rot_count = 0
        for c in cnts_rot:
            area = cv2.contourArea(c)
            if 400 < area < 22000 and rot_count < 3:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.35 and 15 < x < (w_img - 15) and 15 < y < (h_img - 15):
                    # Không đè lên ớt đỏ chín hoàn toàn
                    overlap_red = any(abs(x - rx) < 25 and abs(y - ry) < 25 for rx, ry, rw, rh in red_boxes)
                    if not overlap_red:
                        norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                        if norm_pts:
                            polys_this_img.append((2, norm_pts, pts_pix))
                            rot_boxes.append((x, y, w, h))
                            counts[2] += 1
                            rot_count += 1

        # -------------------------------------------------------------
        # 3. CLASS 1: ot_xanh (Ớt non xanh - đặc trưng quả ớt: dài, thon, bề mặt mịn)
        # Phân biệt kỹ với lá xanh (lá rộng hơn, ít đặc hơn)
        # -------------------------------------------------------------
        green_chili_mask = cv2.inRange(hsv, np.array([34, 75, 50]), np.array([54, 255, 240]))
        green_chili_mask = cv2.morphologyEx(green_chili_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_gc, _ = cv2.findContours(green_chili_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gc_boxes = []
        gc_count = 0
        for c in cnts_gc:
            area = cv2.contourArea(c)
            if 400 < area < 32000 and gc_count < 4:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                # Tiêu chuẩn trái ớt xanh: dáng thon dài (aspect >= 1.75) và viền phẳng mịn liền khối (solidity >= 0.70)
                if aspect >= 1.75 and solidity >= 0.70:
                    overlap_existing = any(abs(x - bx) < 30 and abs(y - by) < 30 for bx, by, bw, bh in (red_boxes + rot_boxes))
                    if not overlap_existing and 15 < x < (w_img - 15) and 15 < y < (h_img - 15):
                        norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                        if norm_pts:
                            polys_this_img.append((1, norm_pts, pts_pix))
                            gc_boxes.append((x, y, w, h))
                            counts[1] += 1
                            gc_count += 1

        # -------------------------------------------------------------
        # 4. CLASS 5: la_benh (Lá có chấm đen hoặc đốm vàng)
        # -------------------------------------------------------------
        foliage = cv2.inRange(hsv, np.array([32, 35, 30]), np.array([85, 255, 255]))
        dark_spots = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 62])) & foliage
        k_cluster = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        clustered_dark = cv2.morphologyEx(dark_spots, cv2.MORPH_CLOSE, k_cluster)
        clustered_dark = cv2.dilate(clustered_dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        yel_spots = cv2.inRange(hsv, np.array([18, 65, 75]), np.array([34, 255, 255]))
        dis_mask = cv2.morphologyEx(clustered_dark | yel_spots, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_dis, _ = cv2.findContours(dis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dis_boxes = []
        dis_count = 0
        significant_dis = sorted([c for c in cnts_dis if 500 < cv2.contourArea(c) < 28000], key=cv2.contourArea, reverse=True)
        for c in significant_dis:
            if dis_count >= 4:
                break
            x, y, w, h = cv2.boundingRect(c)
            overlap_fruit = any(abs(x - bx) < 30 and abs(y - by) < 30 for bx, by, bw, bh in (red_boxes + rot_boxes + gc_boxes))
            if not overlap_fruit and 20 < x < (w_img - 20) and 20 < y < (h_img - 20):
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys_this_img.append((5, norm_pts, pts_pix))
                    dis_boxes.append((x, y, w, h))
                    counts[5] += 1
                    dis_count += 1

        # -------------------------------------------------------------
        # 5. CLASS 3 & 4: la_xanh (Lá xanh khỏe) & la_heo (Lá héo rũ)
        # -------------------------------------------------------------
        leaf_mask = cv2.inRange(hsv, np.array([38, 40, 40]), np.array([82, 240, 245]))
        # Loại trừ vùng lá bệnh và ớt
        leaf_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(dis_mask))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cnts_leaf, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_count = 0
        significant_leaves = sorted([c for c in cnts_leaf if 800 < cv2.contourArea(c) < 35000], key=cv2.contourArea, reverse=True)
        for c in significant_leaves:
            if leaf_count >= 4:
                break
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(float(w)/h, float(h)/w)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(cv2.contourArea(c)) / hull_area if hull_area > 0 else 0
            # Phiến lá có dạng bầu dục hoặc thoi (aspect 1.05 đến 2.2, độ đặc 0.52 đến 0.88)
            if 1.05 <= aspect < 2.2 and 0.52 <= solidity <= 0.88:
                overlap_fruit = any(abs(x - bx) < 35 and abs(y - by) < 35 for bx, by, bw, bh in (red_boxes + rot_boxes + gc_boxes))
                if not overlap_fruit and 20 < x < (w_img - 20) and 20 < y < (h_img - 20):
                    norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                    if norm_pts:
                        # Kiểm tra độ héo rũ: lá rũ thẳng xuống (h > 1.6*w) và màu xanh sẫm/dull
                        roi_hsv = hsv[y:y+h, x:x+w]
                        mean_v = np.mean(roi_hsv[:, :, 2])
                        if h > w * 1.6 and mean_v < 115:
                            polys_this_img.append((4, norm_pts, pts_pix))
                            counts[4] += 1
                        else:
                            polys_this_img.append((3, norm_pts, pts_pix))
                            counts[3] += 1
                        leaf_count += 1

        # -------------------------------------------------------------
        # Ghi file nhãn chuẩn YOLO Segmentation
        # -------------------------------------------------------------
        lines = []
        for cls_id, norm_pts, _ in polys_this_img:
            pts_str = " ".join(norm_pts)
            lines.append(f"{cls_id} {pts_str}\n")

        txt_name = os.path.splitext(f)[0] + '.txt'
        target_dir = val_lbl_dir if f in val_files else train_lbl_dir
        target_img_dir = val_img_dir if f in val_files else train_img_dir

        with open(os.path.join(target_dir, txt_name), 'w', encoding='utf-8') as out_f:
            out_f.writelines(lines)
        shutil.copy2(img_path, os.path.join(target_img_dir, f))

        # -------------------------------------------------------------
        # Vẽ mẫu Segmentation trực quan để kiểm tra
        # -------------------------------------------------------------
        is_sample = (idx < 25) or (idx % 25 == 0) or ('snap' in f)
        if is_sample and saved_sample_count < 35:
            sample_img = img.copy()
            overlay = sample_img.copy()

            # 1. Vẽ mặt nạ đa giác bán trong suốt
            for cls_id, _, pts_pix in polys_this_img:
                c_color = colors.get(cls_id, (255, 255, 255))
                cv2.drawContours(overlay, [pts_pix], -1, c_color, -1)

            sample_img = cv2.addWeighted(overlay, 0.40, sample_img, 0.60, 0)

            # 2. Vẽ viền đa giác sắc nét và nhãn lớp
            for cls_id, _, pts_pix in polys_this_img:
                c_color = colors.get(cls_id, (255, 255, 255))
                cv2.drawContours(sample_img, [pts_pix], -1, c_color, 2)
                bx, by = np.min(pts_pix, axis=0)
                tag = class_names[cls_id]
                cv2.putText(sample_img, tag, (bx, max(20, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_color, 2)

            sample_fname = f"seg_sample_{saved_sample_count:02d}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(samples_dir, sample_fname), sample_img)
            saved_sample_count += 1

    print("\n GÁN NHÃN POLYGON SEGMENTATION HOÀN TẤT THÀNH CÔNG VỚI 6 LỚP:")
    for i, name in enumerate(class_names):
        print(f"  - Lớp {i} ({name}): {counts[i]} đối tượng đa giác")
    print(f"  - Tổng số nhãn Segmentation: {sum(counts.values())}")
    print(f"  - Tập Train: {len(files) - len(val_files)} ảnh | Tập Val: {len(val_files)} ảnh")
    print(f"  - Đã lưu {saved_sample_count} ảnh mẫu trực quan vào: {samples_dir}")

if __name__ == '__main__':
    auto_label_dataset()
