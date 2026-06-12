import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne

def push_silver_to_mongodb():
    mongo_uri = os.environ.get("MONGO_URI")
    
    if not mongo_uri:
        print("[LỖI] Không tìm thấy cấu hình MONGO_URI trong biến môi trường hệ thống!")
        sys.exit(1)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')

    if not os.path.exists(silver_path):
        print(f"[LỖI] Không tìm thấy file dữ liệu Silver tại {silver_path}")
        return

    df = pd.read_csv(silver_path)
    df = df.fillna({
    'brand': 'Unknown',
    'model': 'Khác',
    'price_numeric': 0,
    'year': 0,
    'car_age': 0,
    'odo_numeric': 0,
    'transmission': 'Không rõ',
    'fuel': 'Xăng'
    })
    df['location'] = df['location'].replace(['nan', 'NaN', None, np.nan], 'Không rõ')
    records = df.to_dict(orient='records')
    total_records = len(records)
    print(f"[*] Đang chuẩn bị đồng bộ {total_records} bản ghi từ Tầng Silver lên MongoDB Cloud Atlas...")

    client = MongoClient(mongo_uri)
    db = client['car_market_db']         
    collection = db['silver_cars']       

    # Chia nhỏ dữ liệu thành các lô để tránh quá tải
    BATCH_SIZE = 1000
    total_upserted = 0
    total_modified = 0

    print(f"Luồng truyền tải chia nhỏ (Batch size = {BATCH_SIZE})...")

    # Vòng lặp tịnh tiến cắt nhỏ danh sách xe
    for i in range(0, total_records, BATCH_SIZE):
        batch_records = records[i:i + BATCH_SIZE]
        operations = []
        
        for record in batch_records:
            filter_query = {'id': str(record['id'])}
            update_query = {'$set': record}
            operations.append(UpdateOne(filter_query, update_query, upsert=True))
        
        if operations:
            try:
                # Gửi lô hiện tại lên đám mây
                result = collection.bulk_write(operations, ordered=False)
                total_upserted += result.upserted_count
                total_modified += result.modified_count
                
                # Tính toán phần trăm tiến độ trực quan
                progress = min(((i + BATCH_SIZE) / total_records) * 100, 100)
                print(f"   [+] Đã đồng bộ thành công cụm dòng {i} -> {min(i + BATCH_SIZE, total_records)} ({progress:.2f}%)")
            except Exception as e:
                print(f"   Lỗi xảy ra tại lô dữ liệu {i}: {e}")
                continue
    
    print("\n================================================================")
    print("[DATABASE PIPELINE] ĐỒNG BỘ THÀNH CÔNG")
    print(f"   - Tổng số xe mới tinh được chèn vào kho Cloud: {total_upserted}")
    print(f"   - Tổng số xe cũ được cập nhật trạng thái: {total_modified}")
    print("================================================================")
    
    client.close()

if __name__ == "__main__":
    push_silver_to_mongodb()