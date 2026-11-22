# ======================================================================================
# ĐỀ TÀI: HỆ THỐNG VƯỜN ỚT THÔNG MINH - SMART CHILI AI
# FILE: train_model.py
# VAI TRÒ: HUẤN LUYỆN MACHINE LEARNING (CÂY QUYẾT ĐỊNH & RANDOM FOREST)
# - ĐÁP ỨNG CHUẨN RUBRIC BÀI LAB TRƯỜNG: Dự đoán hành vi tưới nước từ 5 thông số
# - BIẾN ĐẦU VÀO (FEATURES):
#     1. soil_pct: Độ ẩm đất (%)
#     2. temp: Nhiệt độ khí quyển (°C)
#     3. hum: Độ ẩm không khí (%)
#     4. lux: Cường độ ánh sáng (Lux)
#     5. rain_prob: Xác suất mưa từ API thời tiết (%)
# - BIẾN MỤC TIÊU (ACTION / TARGET):
#     0: Không tưới | 1: Hoãn tưới (chờ mưa) | 2: Tưới nước
# - ĐẦU RA:
#     smart_garden_tree.joblib : File mô hình nạp cho server.py
#     confusion_matrix.png     : Ảnh ma trận nhầm lẫn để chèn vào báo cáo
#     decision_tree.png        : Ảnh sơ đồ cây quyết định để chèn vào báo cáo
# ======================================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

CSV_FILE = "smart_garden_demo.csv"
MODEL_FILE = "smart_garden_tree.joblib"

def generate_sample_dataset(n_samples=600):
    np.random.seed(42)
    soil = np.random.uniform(5, 95, n_samples)
    temp = np.random.uniform(20, 40, n_samples)
    hum = np.random.uniform(40, 95, n_samples)
    lux = np.random.uniform(100, 45000, n_samples)
    rain_prob = np.random.uniform(0, 100, n_samples)

    # Nhãn quyết định: 0 = Không tưới, 1 = Hoãn tưới (chờ mưa), 2 = Tưới nước
    action = []
    for s, t, r in zip(soil, temp, rain_prob):
        if s < 20:
            action.append(2)  # Khô nguy cấp -> Bắt buộc tưới
        elif s >= 55:
            action.append(0)  # Đủ ẩm -> Không tưới
        elif r >= 65:
            action.append(1)  # Đất hơi khô nhưng sắp mưa -> Hoãn tưới
        else:
            action.append(2)  # Đất khô & ít mưa -> Tưới

    df = pd.DataFrame({
        "soil_pct": np.round(soil, 1),
        "temp": np.round(temp, 1),
        "hum": np.round(hum, 1),
        "lux": np.round(lux, 0),
        "rain_prob": np.round(rain_prob, 1),
        "action": action
    })
    df.to_csv(CSV_FILE, index=False)
    print(f"[OK] Đã tạo dữ liệu mẫu tại {CSV_FILE} ({n_samples} dòng)")
    return df

def train():
    # --- BƯỚC 1: ĐỌC DỮ LIỆU ĐA BIẾN TỪ FILE CSV ---
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = generate_sample_dataset()

    features = ["soil_pct", "temp", "hum", "lux", "rain_prob"]
    X = df[features]
    y = df["action"]

    # --- BƯỚC 2: PHÂN CHIA TẬP HUẤN LUYỆN (80%) VÀ TẬP KIỂM THỬ (20%) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BƯỚC 3: HUẤN LUYỆN THUẬT TOÁN HỌC MÁY RANDOM FOREST ---
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("\n=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ===")
    print(classification_report(y_test, y_pred, target_names=["Không tưới", "Hoãn tưới", "Tưới"]))

    # --- BƯỚC 4: LƯU MÔ HÌNH VÀO FILE .joblib ĐỂ NẠP CHO SERVER ---
    joblib.dump(rf_model, MODEL_FILE)
    print(f"[OK] Đã lưu mô hình vào {MODEL_FILE}")

    # --- BƯỚC 5: XUẤT ẢNH MA TRẬN NHẦM LẪN (CONFUSION MATRIX) BÁO CÁO ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Khong tuoi", "Hoan tuoi", "Tuoi"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - SMART CHILI AI")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # --- BƯỚC 6: XUẤT ẢNH MINH HỌA SƠ ĐỒ CÂY QUYẾT ĐỊNH (DECISION TREE) BÁO CÁO ---
    dt_vis = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    plt.figure(figsize=(14, 7))
    plot_tree(dt_vis, feature_names=features, class_names=["Khong tuoi", "Hoan tuoi", "Tuoi"], filled=True, rounded=True)
    plt.title("Decision Tree Rules - SMART CHILI AI")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150)
    plt.close()
    print("[OK] Đã xuất confusion_matrix.png và decision_tree.png")

if __name__ == "__main__":
    train()