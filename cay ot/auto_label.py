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
    for eps in [0.010, 0.015, 0.020, 0.028]:
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
    # 1: ot_xanh   - Trái ớt chỉ thiên xanh non mọc hướng lên (dáng thon nhọn, vỏ bóng)
    # 2: ot_hong   - Trái ớt hỏng / thối / héo teo (có màu đen hoặc cam cháy)
    # 3: la_xanh   - Phiến lá xanh khỏe mạnh (không đốm bệnh)
    # 4: la_heo    - Phiến lá héo rũ (chúi thẳng xuống)
    # 5: la_benh   - Phiến lá bệnh (lá có chấm/đốm đen hoại tử hoặc đốm vàng)
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
        1: (0, 255, 0),    # ot_xanh: Xanh lá mạ
        2: (0, 100, 210),  # ot_hong: Nâu cam / cam đen
        3: (0, 200, 100),  # la_xanh: Xanh lục
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        polys_this_img = []  # danh sách (cls_id, norm_pts, pts_pixel)

        # -------------------------------------------------------------
        # VÙNG LOẠI TRỪ NHIỄU KHÔNG GIAN (SPATIAL EXCLUSION MASK)
        # Loại bỏ tuyệt đối:
        # - Watermark iVCam góc trên cùng bên trái
        # - Đáy chậu, đất trồng, dây nilon buộc cây, tay cầm ở đáy
        # - Bìa carton và mặt bàn bên trái
        # - Bo mạch ESP32, dây cáp, đèn LED đỏ bên phải
        # -------------------------------------------------------------
        mask_valid = np.ones((h_img, w_img), dtype=np.uint8) * 255
        mask_valid[0:150, 0:290] = 0   # Watermark iVCam
        mask_valid[565:, :] = 0        # Đáy chậu, đất, dây buộc, bàn tay
        mask_valid[400:, 0:230] = 0    # Thùng carton bên trái
        mask_valid[300:, 910:] = 0    # Bo mạch vi điều khiển bên phải

        # Mặt nạ toàn bộ tán lá / thực vật
        foliage = cv2.inRange(hsv, np.array([28, 30, 25]), np.array([88, 255, 240]))
        foliage = cv2.bitwise_and(foliage, mask_valid)

        # Mặt nạ đốm đen hoại tử (dùng để xác định la_benh và ot_hong)
        dark_spots = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 48]))
        dark_spots = cv2.bitwise_and(dark_spots, foliage)

        occupied = np.zeros((h_img, w_img), dtype=np.uint8)

        # -------------------------------------------------------------
        # 1. CLASS 0: ot_do (Trái ớt chín đỏ)
        # -------------------------------------------------------------
        m1 = cv2.inRange(hsv, np.array([0, 95, 60]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([168, 95, 60]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_and(m1 | m2, mask_valid)
        cnts_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_count = 0
        for c in sorted(cnts_red, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 70 < area < 10000 and red_count < 3:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.25 and 200 < x < 880:
                    norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                    if norm_pts:
                        polys_this_img.append((0, norm_pts, pts_pix))
                        cv2.drawContours(occupied, [c], -1, 255, -1)
                        counts[0] += 1
                        red_count += 1

        # -------------------------------------------------------------
        # 2. CLASS 1: ot_xanh (Trái ớt chỉ thiên xanh non mọc hướng lên)
        # -------------------------------------------------------------
        gc_mask = cv2.inRange(hsv, np.array([33, 65, 45]), np.array([60, 235, 220]))
        gc_mask = cv2.bitwise_and(gc_mask, mask_valid)
        gc_mask = cv2.bitwise_and(gc_mask, cv2.bitwise_not(occupied))
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        gc_clean = cv2.morphologyEx(gc_mask, cv2.MORPH_OPEN, k_vert)
        dist = cv2.distanceTransform(gc_clean, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, 0.28 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        cnts_gc, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gc_candidates = []
        for c in cnts_gc:
            area = cv2.contourArea(c)
            if 120 < area < 10000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = float(h) / max(1, w)
                hull = cv2.convexHull(c)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                if (aspect >= 1.35 or (aspect >= 1.15 and solidity >= 0.72)) and y + h < 560:
                    gc_candidates.append(c)

        gc_count = 0
        for c in sorted(gc_candidates, key=cv2.contourArea, reverse=True)[:6]:
            norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
            if norm_pts:
                polys_this_img.append((1, norm_pts, pts_pix))
                cv2.drawContours(occupied, [c], -1, 255, -1)
                counts[1] += 1
                gc_count += 1

        # -------------------------------------------------------------
        # 3. CLASS 2: ot_hong (Ớt hỏng / thối đen hoặc cam cháy)
        # -------------------------------------------------------------
        m_orange = cv2.inRange(hsv, np.array([10, 65, 45]), np.array([25, 220, 190]))
        m_orange = cv2.bitwise_and(m_orange, mask_valid)
        black_fruit = cv2.bitwise_and(dark_spots, cv2.bitwise_not(occupied))
        black_fruit = cv2.morphologyEx(black_fruit, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts_bf, _ = cv2.findContours(black_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rot_count = 0
        for c in sorted(cnts_bf, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 250 < area < 4000 and rot_count < 2:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.3 and 300 < x < 750 and 400 < y < 560:
                    norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                    if norm_pts:
                        polys_this_img.append((2, norm_pts, pts_pix))
                        cv2.drawContours(occupied, [c], -1, 255, -1)
                        counts[2] += 1
                        rot_count += 1
                        break

        cnts_orange, _ = cv2.findContours(m_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts_orange, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 200 < area < 5000 and rot_count < 2:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                if aspect >= 1.3 and 250 < x < 800 and 150 < y < 560:
                    overlap = np.sum(cv2.bitwise_and(occupied, cv2.drawContours(np.zeros_like(occupied), [c], -1, 255, -1)) > 0)
                    if overlap < 50:
                        norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                        if norm_pts:
                            polys_this_img.append((2, norm_pts, pts_pix))
                            cv2.drawContours(occupied, [c], -1, 255, -1)
                            counts[2] += 1
                            rot_count += 1
                            break

        # -------------------------------------------------------------
        # 4. CLASS 3, 4, 5: PHIẾN LÁ (Lá bệnh, Lá xanh khỏe, Lá héo rũ)
        # Tách phiến lá cá thể bằng đường viền Canny
        # -------------------------------------------------------------
        edges = cv2.Canny(gray, 30, 90)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        leaves_split = cv2.bitwise_and(foliage, cv2.bitwise_not(edges_dilated))
        leaves_split = cv2.bitwise_and(leaves_split, cv2.bitwise_not(occupied))
        leaves_split = cv2.morphologyEx(leaves_split, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cnts_leaf, _ = cv2.findContours(leaves_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_candidates = []
        for c in cnts_leaf:
            area = cv2.contourArea(c)
            if 350 < area < 10000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(float(w)/h, float(h)/w)
                hull = cv2.convexHull(c)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                if 1.05 <= aspect <= 2.8 and solidity >= 0.52:
                    c_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    dark_in_leaf = cv2.bitwise_and(dark_spots, c_mask)
                    dark_count = np.sum(dark_in_leaf > 0)
                    leaf_candidates.append((area, c, (x, y, w, h), dark_count))

        leaf_candidates.sort(key=lambda x: x[0], reverse=True)
        dis_count, healthy_count, wilting_count = 0, 0, 0

        for area, c, (x, y, w, h), dark_count in leaf_candidates:
            norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
            if not norm_pts:
                continue

            # Lá có chấm đen / đốm hoại tử là LÁ BỆNH (la_benh)
            if dark_count >= 12 and dis_count < 5:
                polys_this_img.append((5, norm_pts, pts_pix))
                counts[5] += 1
                dis_count += 1
            # Lá rũ thẳng xuống hướng đất là LÁ HÉO (la_heo)
            elif h > w * 1.6 and y > 320 and wilting_count < 3:
                polys_this_img.append((4, norm_pts, pts_pix))
                counts[4] += 1
                wilting_count += 1
            # Lá xanh mượt không đốm bệnh là LÁ XANH (la_xanh)
            elif dark_count < 5 and healthy_count < 5:
                polys_this_img.append((3, norm_pts, pts_pix))
                counts[3] += 1
                healthy_count += 1

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
