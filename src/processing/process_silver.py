import os
import re
import pandas as pd
import numpy as np

def clean_and_transform_silver():
    bronze_path = os.path.join('data', 'bronze', 'bronze_cars_raw.csv')
    silver_path = os.path.join('data', 'silver', 'silver_cars_clean.csv')
    
    if not os.path.exists(bronze_path):
        print(f"Không tìm thấy file dữ liệu tại {bronze_path}")
        return

    df = pd.read_csv(bronze_path)
    print(f"[*] Đang xử lý {len(df)} bản ghi thô từ tầng Bronze...")

    # DEDUPLICATION
    df['_ingested_at'] = pd.to_datetime(df['_ingested_at'])
    df = df.sort_values(by='_ingested_at', ascending=False)
    df = df.drop_duplicates(subset=['id'], keep='first')

    # TỪ ĐIỂN ÁNH XẠ
    BRAND_MAPPING = {
        'mercedes': 'Mercedes-Benz', 'mercedes-benz': 'Mercedes-Benz', 'mercedesbenz': 'Mercedes-Benz',
        'benz': 'Mercedes-Benz', 'merc': 'Mercedes-Benz', 'mec': 'Mercedes-Benz', 
        'mecedes': 'Mercedes-Benz', 'mercede': 'Mercedes-Benz',
        
        'vinfast': 'VinFast', 'vf': 'VinFast', 'vin': 'VinFast',
        
        'mitsubishi': 'Mitsubishi', 'mit': 'Mitsubishi', 'mitsu': 'Mitsubishi', 'mitsubisi': 'Mitsubishi',
        
        'volkswagen': 'Volkswagen', 'vw': 'Volkswagen', 'volkwagen': 'Volkswagen',
        
        'landrover': 'Land Rover', 'land': 'Land Rover', 'rover': 'Land Rover',
        
        'bmw': 'BMW', 'toyota': 'Toyota', 'kia': 'Kia', 'mazda': 'Mazda', 
        'honda': 'Honda', 'ford': 'Ford', 'nissan': 'Nissan', 'peugeot': 'Peugeot', 'mg': 'MG',
        'porsche': 'Porsche', 'audi': 'Audi', 'volvo': 'Volvo', 'subaru': 'Subaru',
        'chevrolet': 'Chevrolet', 'chevy': 'Chevrolet',
        'lexus': 'Lexus', 'lex': 'Lexus',
        'suzuki': 'Suzuki', 'su': 'Suzuki',
        'hyundai': 'Hyundai', 'huyn dai': 'Hyundai',
        
        'wuling': 'Wuling', 'byd': 'BYD', 'haval': 'Haval', 'omoda': 'Omoda', 
        'jaecoo': 'Jaecoo', 'lynkco': 'Lynk & Co', 'lynk': 'Lynk & Co'
    }

    silver_data = []

    for _, row in df.iterrows():
        try:
            if pd.isna(row['id']) or pd.isna(row['raw_title']):
                continue

            item = {
                'id': str(int(row['id'])),
                'url': row['url'],
                'crawl_date': row['_ingested_at'].strftime('%Y-%m-%d')
            }
            
            # Đưa tiêu đề về dạng chữ thường
            title_clean = " ".join(str(row['raw_title']).split())
            title_lower = title_clean.lower()
            
            # Tách các từ trong tiêu đề
            words = title_clean.split()
            if words[0].lower() == "xe" and len(words) > 1:
                first_keyword = words[1].lower()
                fallback_name = words[1].capitalize()
            else:
                first_keyword = words[0].lower()
                fallback_name = words[0].capitalize()

            # 1 & 4. TRÍCH XUẤT HÃNG XE CÓ CƠ CHẾ DỰ PHÒNG FALLBACK
            detected_brand = "Unknown"
            # Bước A: Quét toàn bộ chuỗi tiêu đề xem có chứa KEY nào trong từ điển không
            for key, standardized_name in BRAND_MAPPING.items():
                if key in title_lower:
                    detected_brand = standardized_name
                    break
            
            # Bước B: Nếu không khớp từ điển, áp dụng Fallback lấy từ nhận diện đầu tiên viết hoa
            if detected_brand == "Unknown" and first_keyword != "xe":
                detected_brand = BRAND_MAPPING.get(first_keyword, fallback_name)
            
            item['brand'] = detected_brand

            # 2. TRÍCH XUẤT DÒNG XE NÂNG CAO
            model_text = title_clean.replace("Xe ", "")
            if detected_brand != "Unknown":
                model_text = re.sub(detected_brand, '', model_text, flags=re.IGNORECASE)
                if detected_brand == "Mercedes-Benz":
                    model_text = re.sub('Benz|Mercedes|Mec', '', model_text, flags=re.IGNORECASE)
            
            model_parts = model_text.strip().split('-')[0].strip().split()
            extracted_model = " ".join(model_parts[:2]) if len(model_parts) >= 2 else (model_parts[0] if model_parts else "Unknown")
            extracted_model = re.sub(r'\d{4}', '', extracted_model).strip()
            item['model'] = extracted_model if extracted_model else "Unknown"

            # 3. TRÍCH XUẤT PRICE NUMERIC
            price_numeric = np.nan
            if '-' in title_clean:
                price_text = title_clean.split('-')[-1].strip()
                if 'Tỷ' in price_text:
                    parts = price_text.split('Tỷ')
                    ty_match = re.findall(r'\d+\.?\d*', parts[0])
                    ty = float(ty_match[0]) if ty_match else 0
                    trieu = 0
                    if len(parts) > 1 and 'Triệu' in parts[1]:
                        trieu_match = re.findall(r'\d+\.?\d*', parts[1])
                        if trieu_match: trieu = float(trieu_match[0])
                    price_numeric = ty * 1000 + trieu
                elif 'Triệu' in price_text:
                    trieu_match = re.findall(r'\d+\.?\d*', price_text)
                    if trieu_match: price_numeric = float(trieu_match[0])
            
            item['price_numeric'] = price_numeric

            # CÁC THUỘC TÍNH SỐ & ĐẶC TRƯNG PHÁ SINH
            item['year'] = int(row['raw_year']) if pd.notna(row['raw_year']) else np.nan
            item['car_age'] = 2026 - item['year'] if pd.notna(item['year']) else np.nan

            # Xử lý Odo số
            odo_str = str(row['raw_odo']).replace(',', '').replace('.', '').replace('Km', '').strip()
            item['odo_numeric'] = int(odo_str) if odo_str.isdigit() else 0

            if pd.notna(item['car_age']) and item['car_age'] > 0:
                item['odo_per_year'] = round(item['odo_numeric'] / item['car_age'], 2)
            else:
                item['odo_per_year'] = float(item['odo_numeric'])

            # BIẾN PHÂN LOẠI
            item['transmission'] = row['raw_transmission'] if pd.notna(row['raw_transmission']) else "Không rõ"
            item['fuel'] = row['raw_fuel'] if pd.notna(row['raw_fuel']) else "Xăng"
            item['origin'] = row['raw_origin'] if pd.notna(row['raw_origin']) else "Không rõ"
            item['condition'] = row['raw_condition'] if pd.notna(row['raw_condition']) else "Xe đã dùng"
            
            loc_str = str(row['raw_location'])
            item['location'] = loc_str.split(',')[-1].strip() if ',' in loc_str else loc_str

            # NLP REGEX FEATURE ENGINEERING
            desc_lower = str(row['raw_description']).lower()
            item['is_first_owner'] = 1 if any(x in desc_lower for x in ['chính chủ', '1 chủ', 'một chủ', 'mua từ mới', 'đập hộp']) else 0
            item['has_upgrades'] = 1 if any(x in desc_lower for x in ['độ', 'màn hình', 'android', 'camera 360', 'cam 360', 'sub', 'loa', 'phim cách nhiệt']) else 0

            silver_data.append(item)
        except Exception as e:
            print(f"⚠️ [Bỏ qua dòng lỗi] ID {row.get('id', 'Unknown')}: {e}")
            continue

    silver_df = pd.DataFrame(silver_data)
    silver_df = silver_df.dropna(subset=['price_numeric', 'year'])
    
    silver_df.to_csv(silver_path, index=False, encoding='utf-8-sig')
    print(f"[GUARDRAIL UPGRADED] Tầng Silver cập nhật thành công tại '{silver_path}' với {len(silver_df)} dòng dữ liệu")

if __name__ == "__main__":
    clean_and_transform_silver()