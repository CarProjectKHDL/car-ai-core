import pandas as pd
import json
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')
gold_path = os.path.join(BASE_DIR, 'data', 'gold', 'gold_cars_train.csv')
dict_path = os.path.join(BASE_DIR, 'data', 'silver', 'brand_model_dictionary.json')

print("=== 1. KIỂM TRA FILE SILVER ===")
if os.path.exists(silver_path):
    df_s = pd.read_csv(silver_path)
    print("Cột:", df_s.columns.tolist())
    print("Kiểu dữ liệu:\n", df_s.dtypes)
    print("Mẫu 2 dòng dữ liệu đầu tiên:\n", df_s[['brand', 'model_core', 'model_trim', 'price_numeric']].head(2).to_dict(orient='records'))
else:
    print("Không tìm thấy Silver")

print("\n=== 2. KIỂM TRA FILE GOLD ===")
if os.path.exists(gold_path):
    df_g = pd.read_csv(gold_path)
    print("Cột:", df_g.columns.tolist())
    print("Số lượng dòng:", len(df_g))
    print("Thống kê mô tả giá (price_numeric):\n", df_g['price_numeric'].describe())
    
    # Kiểm tra trường final_model_feature tổ hợp
    if 'final_model_feature' in df_g.columns:
        print("Mẫu nhãn final_model_feature:", df_g['final_model_feature'].head(5).tolist())
    else:
        # Nếu chưa có, giả lập logic tạo để xem nhãn sinh ra là gì
        df_g['is_dense_model'] = df_g.groupby(['brand', 'model_core'])['id'].transform('count') >= 15
        df_g['final_model_feature'] = np.where(df_g['is_dense_model'], df_g['model_trim'], df_g['model_core'])
        print("Mẫu nhãn sinh động final_model_feature:", df_g['final_model_feature'].head(5).tolist())
        print("Số lượng nhãn độc nhất:", df_g['final_model_feature'].nunique())
else:
    print("Không tìm thấy Gold")

print("\n=== 3. KIỂM TRA DICTIONARY JSON (MSRP CORES) ===")
if os.path.exists(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Các hãng xe có trong từ điển:", list(data.keys())[:5])
    # Trích xuất 1 ví dụ cụ thể của hãng toyota hoặc mercedes để xem cấu trúc mệnh giá
    first_brand = list(data.keys())[0]
    print(f"Cấu trúc mẫu của hãng [{first_brand}]:")
    print(json.dumps(data[first_brand], ensure_ascii=False, indent=2)[:500])
else:
    print("Không tìm thấy Dictionary")
