import os
import pandas as pd
import numpy as np

# Danh sách các hãng xe siêu sang / quá hiếm gặp trên thị trường VN
# Loại bỏ hoàn toàn khỏi tầng Gold để tránh đầu độc model với outlier giá trị cực đoan
EXCLUDED_BRANDS = [
    'Aion', 'Aston Martin', 'Bentley', 'Dongfeng', 'GAC', 'Haima',
    'Haval', 'Hongqi', 'Jaecoo', 'Jaguar', 'Lotus', 'Lynk & Co',
    'Maserati', 'Omoda', 'Ram', 'Skoda', 'Wuling',
    'Rolls Royce', 'Rolls-Royce', 'LandRover', 'Land Rover'
]

def generate_gold_layer():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')
    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    gold_path = os.path.join(gold_dir, 'gold_cars_train.csv')

    print("[*] Đang khởi chạy tiến trình trích xuất và cô lập tầng dữ liệu GOLD...")
    if not os.path.exists(silver_path):
        print(f"[-] Chưa tìm thấy file dữ liệu Silver tại: {silver_path}")
        return

    df = pd.read_csv(silver_path)
    initial_count = len(df)

    # ----------------------------------------------------------------------
    # BƯỚC 0: LOẠI BỎ PHÂN KHÚC SIÊU SANG / HÃNG QUÁ HIẾM GẶP
    # ----------------------------------------------------------------------
    # Chuẩn hóa case-insensitive để khớp tên brand
    excluded_lower = [b.lower().strip() for b in EXCLUDED_BRANDS]
    mask_excluded = df['brand'].astype(str).str.lower().str.strip().isin(excluded_lower)
    removed_count = mask_excluded.sum()
    df = df[~mask_excluded].reset_index(drop=True)
    print(f"[*] Đã loại bỏ {removed_count} dòng thuộc các hãng siêu sang/hiếm: {', '.join(EXCLUDED_BRANDS)}")

    # ----------------------------------------------------------------------
    # BƯỚC 1: LOẠI BỎ NHÓM XE QUÁ HIẾM (< 3 MẪU TOÀN SÀN THEO BRAND+MODEL_CORE)
    # ----------------------------------------------------------------------
    df['category_counts'] = df.groupby(['brand', 'model_core'])['id'].transform('count')
    df = df[df['category_counts'] >= 3].drop(columns=['category_counts']).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # BƯỚC 2: THANH TRỪNG TIN RÁC LƯỜI ĐIỀN SỐ KM
    # ----------------------------------------------------------------------
    # Nếu xe đời sâu (tuổi xe >= 1) mà số Km rao bán vẫn bằng 0 hoặc nhỏ hơn 1000 Km -> Tin rác, xóa!
    df = df[~((df['car_age'] >= 1) & (df['odo_numeric'] <= 1000))].reset_index(drop=True)

    # Xe đời mới tinh cùng năm nếu khuyết ODO (bằng 0) thì để NaN, imputer sẽ lấp bằng trung vị
    df['odo_numeric'] = df['odo_numeric'].replace(0, np.nan)

    # ----------------------------------------------------------------------
    # BƯỚC 3: GROUPED IQR - LỌC OUTLIER GIÁ RIÊNG BIỆT THEO TỪNG DÒNG XE
    # ----------------------------------------------------------------------
    cleaned_groups = []
    grouped = df.groupby(['brand', 'model_core'])
    for names, group in grouped:
        if len(group) >= 5:
            q1 = group['price_numeric'].quantile(0.25)
            q3 = group['price_numeric'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered_group = group[(group['price_numeric'] >= lower_bound) & (group['price_numeric'] <= upper_bound)]
            cleaned_groups.append(filtered_group)
        else:
            cleaned_groups.append(group)

    gold_df = pd.concat(cleaned_groups, ignore_index=True)

    # ----------------------------------------------------------------------
    # BƯỚC 4: ÁP TRẦN GIÁ TUYỆT ĐỐI VÀ TUỔI XE
    # ----------------------------------------------------------------------
    # Sau khi đã loại các hãng siêu sang, ngưỡng 3.5 tỷ chặn các outlier giá còn sót
    # (ví dụ phiên bản hiếm/độ của các hãng phổ thông)
    gold_df = gold_df[gold_df['price_numeric'] <= 9000]

    # Áp trần chặn tuổi xe nghiêm ngặt <= 10 năm
    gold_df = gold_df[gold_df['car_age'] <= 10].reset_index(drop=True)

    # ----------------------------------------------------------------------
    # BƯỚC 5: ĐỒNG BỘ HÓA NHÃN final_model_feature
    # ----------------------------------------------------------------------
    gold_df['is_dense_model'] = gold_df.groupby(['brand', 'model_core'])['id'].transform('count') >= 15
    gold_df['final_model_feature'] = np.where(
        gold_df['is_dense_model'],
        gold_df['brand'].astype(str) + "_" + gold_df['model_trim'].astype(str),
        gold_df['brand'].astype(str) + "_" + gold_df['model_core'].astype(str)
    )

    os.makedirs(gold_dir, exist_ok=True)
    gold_df.to_csv(gold_path, index=False, encoding='utf-8-sig')

    print(f"[*] Tổng số dòng ban đầu: {initial_count}")
    print(f"[SUCCESS] Tầng Gold cô lập thành công tại: {gold_path}. Quy mô: {len(gold_df)} dòng dữ liệu siêu sạch.")

if __name__ == "__main__":
    generate_gold_layer()