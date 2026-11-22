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
    Phân đoạn thực thể toàn diện cho từng bức ảnh:
    - Nhận diện đầy đủ toàn bộ trái ớt và phiến lá trên cây.
    - Loại bỏ 100% nền: Chữ iVCam, vành chậu, dây ni-lông xanh, bàn tay, đất, bàn gỗ, vòi kim loại.
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Mặt nạ loại trừ nền (Background Mask)
    bg_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    if is_timeseries:
        # Loại bỏ 100% watermark iVCam ở góc trên bên trái
        bg_mask[0:170, 0:280] = 255
        # Loại bỏ bàn tay / ngón tay cầm chậu ở góc dưới bên trái
        bg_mask[480:, :280] = 255
        # Loại bỏ bàn gỗ / nền thiết bị ở góc dưới bên phải
        bg_mask[520:, 1050:] = 255
        # Loại bỏ đáy chậu và bàn ở phía dưới
        bg_mask[580:, :] = 255

        # Loại bỏ dây ni-lông xanh buộc quanh miệng chậu (vùng y từ 480 đến 580)
        ribbon_roi = (hsv[480:580, :, 0] >= 38) & (hsv[480:580, :, 0] <= 62) & \
                     (hsv[480:580, :, 1] >= 45) & (hsv[480:580, :, 1] <= 120) & \
                     (hsv[480:580, :, 2] >= 45) & (hsv[480:580, :, 2] <= 130)
        ribbon = np.zeros((h_img, w_img), dtype=np.uint8)
        ribbon[480:580, :][ribbon_roi] = 255
        ribbon = cv2.dilate(ribbon, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        bg_mask[ribbon > 0] = 255
    else:
        # Ảnh snapshot: loại bỏ nền trắng
        bg_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        bg_mask = cv2.bitwise_or(bg_mask, bg_white)
        bg_mask[0:280, 750:] = 255

    # 2. Tách tán cây xanh (Plant Foliage)
    foliage = cv2.inRange(hsv, np.array([32, 45, 35]), np.array([86, 255, 245]))
    foliage = cv2.bitwise_and(foliage, cv2.bitwise_not(bg_mask))
    foliage = cv2.morphologyEx(foliage, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foliage)
    if num_labels <= 1:
        return []

    plant_foliage = np.zeros((h_img, w_img), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            plant_foliage[labels == i] = 255

    # Vùng cho phép quả (gần tán lá, không lọt ra nền)
    fruit_zone = cv2.dilate(plant_foliage, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
    fruit_zone = cv2.bitwise_and(fruit_zone, cv2.bitwise_not(bg_mask))

    occupied = np.zeros((h_img, w_img), dtype=np.uint8)
    polys = []

    # ---------------------------------------------------------
    # 1. ot_do (Lớp 0: Quả ớt chín đỏ)
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

    # ---------------------------------------------------------
    # 2. ot_hong (Lớp 2: Quả ớt hỏng, thối đen hoặc cam cháy)
    # ---------------------------------------------------------
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
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if aspect >= 1.3 and solidity >= 0.65 and bh < 85 and bw < 65:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((2, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    if len([p for p in polys if p[0] == 2]) >= 4:
                        break

    # ---------------------------------------------------------
    # 3. ot_xanh (Lớp 1: Toàn bộ ớt chỉ thiên xanh non mọc hướng lên)
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
        if 100 < area < 5000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bh)/bw, float(bw)/bh)
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if (aspect >= 1.30 or (aspect >= 1.15 and solidity >= 0.70)) and bh < 130 and bw < 130:
                norm_pts, pts_pix = contour_to_polygon(c, w_img, h_img)
                if norm_pts:
                    polys.append((1, norm_pts, pts_pix))
                    cv2.drawContours(occupied, [c], -1, 255, -1)
                    ot_x_c += 1
                    if ot_x_c >= 25:
                        break

    # ---------------------------------------------------------
    # 4. Phiến lá: la_benh (5), la_xanh (3), la_heo (4)
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
        if 350 < area < 12000:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(float(bw)/bh, float(bh)/bw)
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if 1.05 <= aspect <= 3.0 and solidity >= 0.45:
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
        # Lá có chấm đen là lá bệnh
        if dark_count >= 14 and dis_count < 15:
            polys.append((5, norm_pts, pts_pix))
            cv2.drawContours(occupied, [c], -1, 255, -1)
            dis_count += 1
        elif bh > bw * 1.55 and wilting_count < 6:
            polys.append((4, norm_pts, pts_pix))
            cv2.drawContours(occupied, [c], -1, 255, -1)
            wilting_count += 1
        elif dark_count < 6 and healthy_count < 22:
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
    print(f"Bắt đầu gán nhãn POLYGON SEGMENTATION TOÀN DIỆN cho TOÀN BỘ {len(all_files)} hình ảnh cây ớt...")

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

        # Xuất ảnh mẫu trực quan cho TOÀN BỘ 649 ẢNH để người dùng kiểm tra bất kỳ ảnh nào
        overlay = img.copy()
        for cls_id, _, pts_pix in polys:
            cv2.drawContours(overlay, [pts_pix], -1, CLASS_COLORS[cls_id], -1)

        res_img = cv2.addWeighted(overlay, 0.42, img, 0.58, 0)
        for cls_id, _, pts_pix in polys:
            cv2.drawContours(res_img, [pts_pix], -1, CLASS_COLORS[cls_id], 2)
            bx, by = np.min(pts_pix, axis=0)
            tag = CLASS_NAMES[cls_id]
            cv2.putText(res_img, tag, (bx, max(18, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS[cls_id], 2)

        sample_fname = f"seg_{idx+1:03d}_{os.path.splitext(f)[0]}.jpg"
        cv2.imwrite(os.path.join(samples_dir, sample_fname), res_img)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_files):
            print(f" -> Đã xử lý & vẽ nhãn {idx + 1}/{len(all_files)} ảnh...")

    print(f"\n GÁN NHÃN POLYGON SEGMENTATION HOÀN TẤT CHO TOÀN BỘ {len(all_files)} ẢNH:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  - Lớp {i} ({name}): {counts[i]} đối tượng đa giác")
    print(f"  - Tổng số nhãn Segmentation: {sum(counts.values())}")
    print(f"  - Tập Train: {len(all_files) - len(val_files)} ảnh | Tập Val: {len(val_files)} ảnh")
    print(f"  - Đã xuất TOÀN BỘ {len(all_files)}/{len(all_files)} ảnh trực quan vào: {samples_dir}")

if __name__ == '__main__':
    auto_label_dataset()
