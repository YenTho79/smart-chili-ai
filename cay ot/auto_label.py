import cv2
import os
import shutil
import random
import numpy as np

# -------------------------------------------------------------------------
# BẢNG TÊN VÀ MÃ LỚP ĐỐI TƯỢNG (6 LỚP)
# 0: ot_do     - Ớt chín đỏ (thu hoạch)
# 1: ot_xanh   - Ớt chỉ thiên xanh non mọc hướng thẳng đứng (dáng thon nhọn)
# 2: ot_hong   - Quả ớt hỏng / thối đen hoại tử / cam cháy ở thân cây
# 3: la_xanh   - Phiến lá xanh khỏe mạnh, không có đốm bệnh
# 4: la_heo    - Phiến lá héo rũ chúc xuống
# 5: la_benh   - Phiến lá có đốm đen hoại tử hoặc đốm vàng bệnh
# -------------------------------------------------------------------------
CLASS_NAMES = ['ot_do', 'ot_xanh', 'ot_hong', 'la_xanh', 'la_heo', 'la_benh']

CLASS_COLORS = {
    0: (0, 0, 255),    # ot_do: Đỏ
    1: (0, 255, 0),    # ot_xanh: Xanh lá mạ
    2: (0, 120, 255),  # ot_hong: Cam cháy
    3: (0, 200, 100),  # la_xanh: Xanh lục
    4: (0, 215, 255),  # la_heo: Vàng cam
    5: (0, 255, 255)   # la_benh: Vàng chanh
}

def contour_to_polygon(cnt, w_img, h_img, max_pts=16, min_pts=4):
    """Chuyển đổi đường bao OpenCV thành chuỗi đa giác phân đoạn YOLO Polygon chuẩn hóa [0, 1]."""
    peri = cv2.arcLength(cnt, True)
    if peri < 15:
        return None, None
    for eps in [0.012, 0.018, 0.025, 0.035]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) <= max_pts:
            break
    if len(approx) < min_pts:
        approx = cv2.approxPolyDP(cnt, 0.008 * peri, True)
    if len(approx) < 3:
        return None, None
    pts = approx.reshape(-1, 2)
    norm = []
    for p in pts:
        nx = max(0.0001, min(0.9999, float(p[0]) / w_img))
        ny = max(0.0001, min(0.9999, float(p[1]) / h_img))
        norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return norm, pts

def segment_image(img, is_timeseries=True):
    """
    Phân đoạn thực thể chính xác cho từng bức ảnh dựa trên hình thái học và màu sắc sinh học của cây ớt:
    - Loại bỏ tuyệt đối 100% nền: Chữ iVCam, vành chậu nhựa, dây ni-lông xanh, bàn tay, đất, bàn gỗ, vòi kim loại.
    - Nhận diện đúng từng phiến lá bệnh (chấm đen), lá xanh khỏe, lá héo, ớt đỏ, ớt xanh, ớt hỏng.
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Mặt nạ tán lá tự nhiên cây ớt (H trong [34, 85] loại trừ hoàn toàn vòi kim loại & gỗ)
    foliage_raw = cv2.inRange(hsv, np.array([34, 35, 25]), np.array([85, 255, 240]))

    if is_timeseries:
        # Loại bỏ tuyệt đối vùng watermark iVCam ở góc trên bên trái
        foliage_raw[0:125, 0:215] = 0
        # Loại bỏ tuyệt đối chậu nhựa, đất, dây buộc, bàn tay, bàn gỗ (y >= 480)
        foliage_raw[480:, :] = 0
        # Loại bỏ hộp carton bên trái
        foliage_raw[340:, :190] = 0
        # Loại bỏ bo mạch & đèn LED bên phải
        foliage_raw[:, 865:] = 0
    else:
        # Ảnh snapshot: loại bỏ nền trắng chói sáng và thanh vòi xịt
        bg_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        foliage_raw = cv2.bitwise_and(foliage_raw, cv2.bitwise_not(bg_white))
        foliage_raw[0:280, 750:] = 0

    # Loại bỏ dải dây ni-lông xanh
    ribbon = cv2.inRange(hsv, np.array([58, 120, 90]), np.array([95, 255, 255]))
    foliage_raw = cv2.bitwise_and(foliage_raw, cv2.bitwise_not(ribbon))

    foliage_clean = cv2.morphologyEx(foliage_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foliage_clean)
    if num_labels <= 1:
        return []

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    plant_core = (labels == largest_idx).astype(np.uint8) * 255
    core_dilated = cv2.dilate(plant_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    plant_foliage = np.zeros((h_img, w_img), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 80:
            c_mask = (labels == i).astype(np.uint8) * 255
            if cv2.countNonZero(cv2.bitwise_and(c_mask, core_dilated)) > 0:
                plant_foliage[labels == i] = 255

    fruit_zone = cv2.dilate(plant_foliage, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))
    fruit_zone = cv2.bitwise_and(fruit_zone, cv2.bitwise_not(ribbon))
    if is_timeseries:
        fruit_zone[480:, :] = 0
    else:
        fruit_zone[0:280, 750:] = 0

    occupied = np.zeros((h_img, w_img), dtype=np.uint8)
    polys = []

    # ---------------------------------------------------------
    # A. ot_do (Lớp 0): Quả ớt chín đỏ
    # ---------------------------------------------------------
    m1 = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 110, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_and(m1 | m2, fruit_zone)
    cnts_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts_red, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if 80 < area < 4000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bw)/bh, float(bh)/bw)
            if aspect >= 1.25 and bh < 110 and bw < 110:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((0, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    if len([p for p in polys if p[0] == 0]) >= 4:
                        break

    # ---------------------------------------------------------
    # B. ot_hong (Lớp 2): Quả ớt hỏng (thối đen hoặc cam cháy)
    # ---------------------------------------------------------
    # 1) Quả hỏng cháy cam
    m_orange = cv2.inRange(hsv, np.array([10, 75, 30]), np.array([24, 255, 145]))
    m_orange = cv2.bitwise_and(m_orange, fruit_zone)
    m_orange = cv2.bitwise_and(m_orange, cv2.bitwise_not(occupied))
    cnts_org, _ = cv2.findContours(m_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts_org, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if 70 < area < 2000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bw)/bh, float(bh)/bw)
            if aspect >= 1.25 and bh < 85 and bw < 65:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((2, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    break

    # 2) Quả hỏng thối đen hoại tử
    dark_roi = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 42]))
    dark_roi = cv2.bitwise_and(dark_roi, fruit_zone)
    dark_roi = cv2.bitwise_and(dark_roi, cv2.bitwise_not(occupied))
    dark_clean = cv2.morphologyEx(dark_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts_blk, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts_blk, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if 90 < area < 2200:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bw)/bh, float(bh)/bw)
            if aspect >= 1.25 and bh < 85 and bw < 65:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((2, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    break

    # ---------------------------------------------------------
    # C. ot_xanh (Lớp 1): Ớt chỉ thiên xanh non mọc hướng thẳng đứng
    # ---------------------------------------------------------
    m_lime = cv2.inRange(hsv, np.array([30, 42, 95]), np.array([53, 255, 255]))
    m_lime = cv2.bitwise_and(m_lime, fruit_zone)
    m_lime = cv2.bitwise_and(m_lime, cv2.bitwise_not(occupied))
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    m_lime_clean = cv2.morphologyEx(m_lime, cv2.MORPH_OPEN, k_vert)
    cnts_gc, _ = cv2.findContours(m_lime_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ot_x_c = 0
    for c in sorted(cnts_gc, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if 110 < area < 5000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bh)/bw, float(bw)/bh)
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if (aspect >= 1.35 or (aspect >= 1.2 and solidity >= 0.72)) and bh < 130 and bw < 130:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((1, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    ot_x_c += 1
                    if ot_x_c >= 12:
                        break

    # ---------------------------------------------------------
    # D. Phiến lá: la_benh (5), la_xanh (3), la_heo (4)
    # ---------------------------------------------------------
    edges = cv2.Canny(gray, 30, 90)
    edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    leaves_split = cv2.bitwise_and(plant_foliage, cv2.bitwise_not(edges_dilated))
    leaves_split = cv2.bitwise_and(leaves_split, cv2.bitwise_not(occupied))
    leaves_split = cv2.morphologyEx(leaves_split, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts_leaf, _ = cv2.findContours(leaves_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dark_foliage = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 48]))
    dark_foliage = cv2.bitwise_and(dark_foliage, plant_foliage)

    leaf_candidates = []
    for c in cnts_leaf:
        area = cv2.contourArea(c)
        if 400 < area < 12000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bw)/bh, float(bh)/bw)
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if 1.05 <= aspect <= 2.8 and solidity >= 0.48:
                c_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                dark_in_leaf = cv2.bitwise_and(dark_foliage, c_mask)
                dark_count = np.sum(dark_in_leaf > 0)
                leaf_candidates.append((area, c, (bx, by, bw, bh), dark_count))

    leaf_candidates.sort(key=lambda x: x[0], reverse=True)
    dis_count, healthy_count, wilting_count = 0, 0, 0

    for area, c, (bx, by, bw, bh), dark_count in leaf_candidates:
        norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
        if not norm_pts:
            continue
        # Lá có chấm/đốm đen là la_benh
        if dark_count >= 15 and dis_count < 6:
            polys.append((5, norm_pts, pts_pix))
            cv2.drawContours(occupied, [c], -1, 255, -1)
            dis_count += 1
        elif bh > bw * 1.6 and wilting_count < 2:
            polys.append((4, norm_pts, pts_pix))
            cv2.drawContours(occupied, [c], -1, 255, -1)
            wilting_count += 1
        elif dark_count < 6 and healthy_count < 6:
            polys.append((3, norm_pts, pts_pix))
            cv2.drawContours(occupied, [c], -1, 255, -1)
            healthy_count += 1

    return polys

def auto_label_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, 'captured_images')
    dataset_dir = os.path.join(base_dir, 'dataset')
    samples_dir = os.path.join(base_dir, 'labeled_samples')

    if not os.path.exists(img_dir):
        print(f"Lỗi: Thư mục ảnh {img_dir} không tồn tại!")
        return

    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, samples_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(base_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(CLASS_NAMES) + '\n')

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

    random.seed(42)
    val_count = max(1, int(len(all_files) * 0.2))
    val_files = set(random.sample(all_files, val_count))

    counts = {i: 0 for i in range(len(CLASS_NAMES))}
    print(f"Bắt đầu gán nhãn POLYGON SEGMENTATION chính xác từng thực thể cho {len(all_files)} hình ảnh cây ớt...")

    saved_sample_count = 0

    for idx, f in enumerate(all_files):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        is_ts = f.startswith('20261120181549_t')
        polys = segment_image(img, is_timeseries=is_ts)

        lines = []
        for cls_id, norm_pts, _ in polys:
            pts_str = " ".join(norm_pts)
            lines.append(f"{cls_id} {pts_str}\n")
            counts[cls_id] += 1

        txt_name = os.path.splitext(f)[0] + '.txt'
        target_lbl_dir = val_lbl_dir if f in val_files else train_lbl_dir
        target_img_dir = val_img_dir if f in val_files else train_img_dir

        with open(os.path.join(target_lbl_dir, txt_name), 'w', encoding='utf-8') as out_f:
            out_f.writelines(lines)
        shutil.copy2(img_path, os.path.join(target_img_dir, f))

        # Lưu ảnh mẫu trực quan
        is_sample = (idx < 25) or (idx % 20 == 0)
        if is_sample and saved_sample_count < 35:
            overlay = img.copy()
            for cls_id, _, pts_pix in polys:
                cv2.drawContours(overlay, [pts_pix], -1, CLASS_COLORS[cls_id], -1)

            res_img = cv2.addWeighted(overlay, 0.42, img, 0.58, 0)
            for cls_id, _, pts_pix in polys:
                cv2.drawContours(res_img, [pts_pix], -1, CLASS_COLORS[cls_id], 2)
                bx, by = np.min(pts_pix, axis=0)
                tag = CLASS_NAMES[cls_id]
                cv2.putText(res_img, tag, (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS[cls_id], 2)

            sample_fname = f"seg_sample_{saved_sample_count:02d}_{os.path.splitext(f)[0]}.jpg"
            cv2.imwrite(os.path.join(samples_dir, sample_fname), res_img)
            saved_sample_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_files):
            print(f" -> Đã xử lý {idx + 1}/{len(all_files)} ảnh...")

    print("\n GÁN NHÃN POLYGON SEGMENTATION HOÀN TẤT THÀNH CÔNG VỚI 6 LỚP:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  - Lớp {i} ({name}): {counts[i]} đối tượng đa giác")
    print(f"  - Tổng số nhãn Segmentation: {sum(counts.values())}")
    print(f"  - Tập Train: {len(all_files) - len(val_files)} ảnh | Tập Val: {len(val_files)} ảnh")
    print(f"  - Đã lưu {saved_sample_count} ảnh mẫu trực quan vào: {samples_dir}")

if __name__ == '__main__':
    auto_label_dataset()
