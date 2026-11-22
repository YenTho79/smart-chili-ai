# ======================================================================================
# ĐỀ TÀI: HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
# FILE: train_yolo.py
# VAI TRÒ: HUẤN LUYỆN MÔ HÌNH HỌC SÂU (DEEP LEARNING) YOLOV8 NHẬN DIỆN 6 LỚP ĐỐI TƯỢNG
# - 3 LỚP TRÁI: 0: ot_do (Ớt chín đỏ), 1: ot_xanh (Ớt non xanh), 2: ot_hong (Ớt thối/hỏng)
# - 3 LỚP LÁ  : 3: la_xanh (Lá xanh khỏe), 4: la_heo (Lá héo rũ), 5: la_benh (Lá bệnh/vàng)
# ĐẦU RA: File trọng số mô hình chili_yolov8.pt nạp trực tiếp cho Camera trực tiếp
# ======================================================================================

import os
import shutil
import glob
import cv2
from ultralytics import YOLO

def train():
    # --- BƯỚC 1: XÁC ĐỊNH ĐƯỜNG DẪN TẬP DỮ LIỆU DATASET ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Đảm bảo đường dẫn trong data.yaml luôn chuẩn trên mọi máy
    norm_dataset_dir = dataset_dir.replace("\\", "/")
    data_yaml_content = f"""path: {norm_dataset_dir}
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
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    val_img_dir = os.path.join(dataset_dir, "images", "val")
    val_lbl_dir = os.path.join(dataset_dir, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Xóa file cache cũ
    for c in glob.glob(os.path.join(dataset_dir, "labels", "*.cache")):
        try: os.remove(c)
        except Exception: pass

    # 1. Tự động dọn dẹp các ảnh không có nhãn để không biến thành negative sample
    for split, img_d, lbl_d in [('train', train_img_dir, train_lbl_dir), ('val', val_img_dir, val_lbl_dir)]:
        if os.path.exists(img_d):
            for f in os.listdir(img_d):
                base_name = os.path.splitext(f)[0]
                lbl_f = os.path.join(lbl_d, f"{base_name}.txt")
                if not os.path.exists(lbl_f) or os.path.getsize(lbl_f) == 0:
                    try: os.remove(os.path.join(img_d, f))
                    except Exception: pass

    # 2. Đảm bảo tập Val không bị trống
    train_lbl_files = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt') and os.path.getsize(os.path.join(train_lbl_dir, f)) > 0]
    val_lbl_files = [f for f in os.listdir(val_lbl_dir) if f.endswith('.txt') and os.path.getsize(os.path.join(val_lbl_dir, f)) > 0]

    if not val_lbl_files and train_lbl_files:
        for f in train_lbl_files:
            shutil.copy2(os.path.join(train_lbl_dir, f), os.path.join(val_lbl_dir, f))
            img_name = f.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(train_img_dir, img_name)):
                shutil.copy2(os.path.join(train_img_dir, img_name), os.path.join(val_img_dir, img_name))

    print(f"Bắt đầu huấn luyện mô hình YOLOv8 trên {len(train_lbl_files)} ảnh có nhãn...")
    
    # --- BƯỚC 2: KHỞI TẠO MẠNG NƠ-RON DEEP LEARNING YOLOV8 NANO ---
    weights_path = os.path.join(base_dir, "..", "yolov8n-seg.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(base_dir, "..", "yolov8n.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8n-seg.pt"

    model = YOLO(weights_path)
    
    # --- BƯỚC 3: TIẾN HÀNH HUẤN LUYỆN (TRAINING PIPELINE) ---
    # epochs=25: Huấn luyện 25 lượt qua toàn bộ tập ảnh
    # imgsz=320: Kích thước ảnh chuẩn hóa để suy luận realtime 30 FPS
    # batch=4: Kích thước lô dữ liệu tối ưu bộ nhớ GPU/CPU
    results = model.train(
        data=yaml_path,
        epochs=25,
        imgsz=320,
        batch=4,
        workers=2,
        name="chili_detector",
        project=os.path.join(base_dir, "runs"),
        exist_ok=True
    )
    
    # --- BƯỚC 4: XUẤT VÀ SAO CHÉP TRỌNG SỐ TỐT NHẤT (chili_yolov8.pt) SANG PYTHON SERVER ---
    best_pt = os.path.join(base_dir, "runs", "chili_detector", "weights", "best.pt")
    python_dir = os.path.abspath(os.path.join(base_dir, "..", "python"))
    
    if os.path.exists(best_pt):
        target_pt = os.path.join(python_dir, "chili_yolov8.pt")
        shutil.copy2(best_pt, target_pt)
        shutil.copy2(best_pt, os.path.join(base_dir, "chili_yolov8.pt"))
        print(f"✅ Đã lưu mô hình tốt nhất vào: {target_pt}")
        print("Mô hình đã sẵn sàng hoạt động trên Camera trực tiếp!")
    else:
        print("Không tìm thấy best.pt sau khi train.")

if __name__ == "__main__":
    train()
