import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne

def push_silver_to_mongodb():
    mongo_uri = os.environ.get("MONGO_URI")
    
    if not mongo_uri:
        print("[!] CẢNH BÁO: Chưa tìm thấy biến môi trường MONGO_URI.")
        print("    -> Tự động Fallback về cơ sở dữ liệu Localhost MongoDB...")
        mongo_uri = "mongodb://localhost:27017/"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')

    print(f"[*] Đang nạp dữ liệu từ tầng Silver sạch tại: {silver_path}")
    if not os.path.exists(silver_path):
        print(f"[LỖI] Không tìm thấy file dữ liệu Silver tại mục: {silver_path}")
        return

    # 2. Đọc tệp dữ liệu Silver
    df = pd.read_csv(silver_path)
    
    # Lấp các ô khuyết nhãn số học hoặc phân loại cơ bản
    df = df.fillna({
        'brand': 'Unknown',
        'model_core': 'Khác',
        'model_trim': 'Khác',
        'price_numeric': 0,
        'year': 0,
        'car_age': 0,
        'odo_numeric': 0,
        'transmission': 'Không rõ',
        'fuel': 'Xăng'
    })

    # 3. Chuyển đổi DataFrame thành tệp cấu trúc Dictionary chuẩn JSON-like của MongoDB
    records = df.to_dict(orient='records')
    total_records = len(records)
    print(f"[*] Tìm thấy {total_records} dòng xe. Bắt đầu mở cổng Bulk Write kết nối Cloud...")

    try:
        # Khởi tạo Client và trỏ trúng DB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client['car_data_platform']
        collection = db['silver_cars_collection']
    except Exception as connection_error:
        print(f"[LỖI KẾT NỐI] Không thể kết nối tới Cluster MongoDB: {connection_error}")
        return

    # 4. Chia gói dữ liệu
    BATCH_SIZE = 2000
    total_upserted = 0
    total_modified = 0

    print("\n================================================================")
    print("[DATABASE PIPELINE] ĐANG ĐỒNG BỘ HÓA DỮ LIỆU LÊN MONGODB ATLAS...")
    print("================================================================")

    for i in range(0, total_records, BATCH_SIZE):
        batch_records = records[i:i + BATCH_SIZE]
        operations = []
        
        for record in batch_records:
            # Sử dụng ID duy nhất của bài đăng xe làm Khóa chính
            filter_query = {'id': str(record['id'])}
            update_query = {'$set': record}
            
            # Upsert = True: Nếu chưa có id này thì chèn mới, nếu đã có thì ghi đè cập nhật
            operations.append(UpdateOne(filter_query, update_query, upsert=True))
        
        if operations:
            try:
                # Thực thi Bulk Write lô dữ liệu hiện tại lên đám mây
                result = collection.bulk_write(operations, ordered=False)
                total_upserted += result.upserted_count
                total_modified += result.modified_count
                
                progress = min(((i + BATCH_SIZE) / total_records) * 100, 100)
                print(f"   [+] Đã đồng bộ thành công cụm dòng {i} -> {min(i + BATCH_SIZE, total_records)} ({progress:.2f}%)")
            except Exception as e:
                print(f"   [-] Lỗi xảy ra tại lô dữ liệu dải {i}: {e}")
                continue
    
    print("\n================================================================")
    print("[DATABASE PIPELINE] ĐỒNG BỘ THÀNH CÔNG")
    print(f"   - Tổng số xe mới tinh được chèn vào kho Cloud: {total_upserted}")
    print(f"   - Tổng số xe cũ được cập nhật trạng thái       : {total_modified}")
    print("================================================================\n")
    
    client.close()

if __name__ == "__main__":
    push_silver_to_mongodb()