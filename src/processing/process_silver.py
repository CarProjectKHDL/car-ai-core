import os
import re
import json
import pandas as pd
import numpy as np

def load_hybrid_master_dictionary(base_dir):
    """Đọc từ điển lai phân cấp"""
    dict_path = os.path.join(base_dir, 'data', 'silver', 'brand_model_dictionary.json')
    if not os.path.exists(dict_path):
        print(f"[-] Không tìm thấy file từ điển phân cấp lai tại {dict_path}!")
        return {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_clean_numeric_price(title_raw):
    """
    Regex Engine: Bóc tách giá xe tuyệt đối
    """
    title_str = str(title_raw).strip()
    price_numeric = np.nan

    ty_match = re.search(r'(\d+\.?\d*)\s*Tỷ(?:\s*(\d+)\s*Triệu)?', title_str, re.IGNORECASE)
    trieu_match = re.search(r'(\d+)\s*Triệu', title_str, re.IGNORECASE)

    if ty_match:
        ty_part = float(ty_match.group(1))
        trieu_part = float(ty_match.group(2)) if ty_match.group(2) else 0.0
        price_numeric = ty_part * 1000 + trieu_part
    elif trieu_match:
        price_numeric = float(trieu_match.group(1))
        
    return price_numeric

def extract_hierarchical_car_info(title_raw, master_dict):
    """
    Thuật toán trích xuất đa tầng phân cấp (Hierarchical Extraction Engine):
    Vá triệt để lỗi bóc tách nhãn 'az' lệch lạc của các dòng xe Mazda.
    """
    title_lower = " ".join(str(title_raw).split()).lower()
    
    official_brand = "Unknown"
    brand_key_found = None
    model_core = "Khác"
    model_trim = "Khác"
    
    sorted_brands = sorted(master_dict.keys(), key=len, reverse=True)
    for brand_key in sorted_brands:
        if brand_key in title_lower:
            brand_key_found = brand_key
            official_brand = master_dict[brand_key]["official_brand"]
            break
            
    if official_brand == "Unknown":
        if any(x in title_lower for x in ["mercedes", "benz", "mec", "mer"]):
            official_brand = "Mercedes-Benz"
            brand_key_found = "mercedes"
        elif any(x in title_lower for x in ["vinfast", "vf"]):
            official_brand = "VinFast"
            brand_key_found = "vinfast"
        elif "bmw" in title_lower:
            official_brand = "BMW"
            brand_key_found = "bmw"
        elif any(x in title_lower for x in ["mazda", "masda"]):
            official_brand = "Mazda"
            brand_key_found = "mazda"

    if official_brand != "Unknown" and brand_key_found in master_dict:
        brand_data = master_dict[brand_key_found]
        core_models = brand_data.get("core_models", [])
        trims_data = brand_data.get("trims", {})
        
        for core in sorted(core_models, key=len, reverse=True):
            if core.lower() in title_lower:
                model_core = core
                break
                
        core_key = model_core.lower()
        if core_key in trims_data:
            available_trims = trims_data[core_key]
            for trim in sorted(available_trims, key=len, reverse=True):
                if trim.lower() in title_lower:
                    model_trim = trim
                    break
                    
        if model_core == "Khác":
            clean_text = re.sub(r'\b' + re.escape(brand_key_found) + r'\b', '', title_lower).strip()
            if brand_key_found == "mazda":
                clean_text = clean_text.replace("mazda", "").replace("masda", "").strip()
            
            clean_text = re.sub(r'^xe\s+', '', clean_text).strip()
            parts = [p for p in clean_text.split() if len(p) > 1 and not p.isdigit()]
            fallback_core = parts[0].strip().replace('-', '').replace('_', '') if parts else "Khác"
            
            if fallback_core.lower() in ['az', 'z', 'm']:
                model_core = "Khác"
            else:
                model_core = fallback_core.capitalize() if fallback_core else "Khác"
            
        if model_trim == "Khác" or len(model_trim) <= 2:
            model_trim = model_core

    return official_brand, model_core, model_trim

def extract_accident_free_status(description):
    if not isinstance(description, str) or pd.isna(description):
        return 0
    desc_lower = description.lower()
    patterns = [
        r"khôn?g\s+(đâm|va)\s+(đụng|quẹt|chạm|nhau)",
        r"khôn?g\s+(tai\s+nạn|lỗi)",
        r"khôn?g\s+(ngập\s+nước|thủy\s+kích|lội\s+nước)",
        r"keo\s+chỉ\s+(zin|nguyên\s+bản)",
        r"máy\s+(móc\s+)?(zin|nguyên\s+bản|chưa\s+hạ|chưa\s+bổ)",
        r"bao\s+(check|test)\s+hãng"
    ]
    for pattern in patterns:
        if re.search(pattern, desc_lower):
            return 1
    return 0

def clean_and_transform_silver():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bronze_path = os.path.join(BASE_DIR, 'data', 'bronze', 'bronze_cars_raw.csv')
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')

    print("[*] Đang khởi chạy quy trình chuẩn hóa dữ liệu tầng Silver...")
    if not os.path.exists(bronze_path):
        print(f"[-] Không tìm thấy file thô tại {bronze_path}")
        return

    master_dict = load_hybrid_master_dictionary(BASE_DIR)
    df = pd.read_csv(bronze_path)

    df['_ingested_at'] = pd.to_datetime(df['_ingested_at'])
    df = df.sort_values(by='_ingested_at', ascending=False)
    df = df.drop_duplicates(subset=['id'], keep='first')

    silver_data = []
    desc_col = 'description' if 'description' in df.columns else 'raw_description'

    for _, row in df.iterrows():
        try:
            if pd.isna(row['id']) or pd.isna(row['raw_title']):
                continue

            item = {
                'id': str(int(row['id'])),
                'url': row['url'],
                'crawl_date': row['_ingested_at'].strftime('%Y-%m-%d')
            }
            
            raw_title = str(row['raw_title'])
            brand, m_core, m_trim = extract_hierarchical_car_info(raw_title, master_dict)
            item['brand'] = brand
            item['model_core'] = m_core
            item['model_trim'] = m_trim
            item['model'] = m_trim 
            item['price_numeric'] = extract_clean_numeric_price(raw_title)

            item['year'] = int(row['raw_year']) if pd.notna(row['raw_year']) else np.nan
            item['car_age'] = 2026 - item['year'] if pd.notna(item['year']) else np.nan

            # Flag xe mới/lướt
            item['is_brand_new'] = 1 if pd.notna(item['car_age']) and item['car_age'] <= 0 else 0

            odo_str = str(row['raw_odo']).replace(',', '').replace('.', '').replace('Km', '').strip()
            item['odo_numeric'] = int(odo_str) if odo_str.isdigit() else 0

            if pd.notna(item['car_age']) and item['car_age'] > 0:
                item['odo_per_year'] = round(item['odo_numeric'] / item['car_age'], 2)
            else:
                item['odo_per_year'] = float(item['odo_numeric'])

            item['transmission'] = row['raw_transmission'] if pd.notna(row['raw_transmission']) else "Không rõ"
            item['fuel'] = row['raw_fuel'] if pd.notna(row['raw_fuel']) else "Xăng"
            item['origin'] = row['raw_origin'] if pd.notna(row['raw_origin']) else "Không rõ"
            item['condition'] = row['raw_condition'] if pd.notna(row['raw_condition']) else "Xe đã dùng"
            
            # BIẾN LOCATION ĐÃ BỊ LOẠI BỎ TOÀN DIỆN KHỎI HỆ THỐNG TẠI ĐÂY LÀM SẠCH SCHEMA

            desc_val = str(row.get(desc_col, ''))
            desc_lower = desc_val.lower()
            item['is_first_owner'] = 1 if any(x in desc_lower for x in ['chính chủ', '1 chủ', 'một chủ', 'mua từ mới', 'đập hộp']) else 0
            item['has_upgrades'] = 1 if any(x in desc_lower for x in ['độ', 'màn hình', 'android', 'camera 360', 'cam 360', 'sub', 'loa', 'phim cách nhiệt']) else 0
            item['is_verified_accident_free'] = extract_accident_free_status(desc_val)

            silver_data.append(item)
        except Exception as e:
            continue

    silver_df = pd.DataFrame(silver_data)
    silver_df = silver_df.dropna(subset=['price_numeric', 'year'])
    
    os.makedirs(os.path.dirname(silver_path), exist_ok=True)
    silver_df.to_csv(silver_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Tầng Silver được cập nhật sạch sẽ (Không chứa Location). Total: {len(silver_df)} dòng.")
    # Trong silver_df, sau khi build xong
    mask_odo_zero_old_car = (silver_df['car_age'] > 0) & (silver_df['odo_numeric'] == 0)
    print(f"Xe cũ (car_age>0) nhưng odo=0 (có thể lỗi crawl): {mask_odo_zero_old_car.sum()}")
if __name__ == "__main__":
    clean_and_transform_silver()