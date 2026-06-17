import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne

def push_data_to_mongodb():
    mongo_uri = os.environ.get("MONGO_URI")
    
    if not mongo_uri:
        print("[!] CẢNH BÁO: Chưa tìm thấy biến môi trường MONGO_URI.")
        print("    -> Tự động Fallback về cơ sở dữ liệu Localhost MongoDB...")
        mongo_uri = "mongodb://localhost:27017/"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bronze_path = os.path.join(BASE_DIR, 'data', 'bronze', 'bronze_cars_raw.csv')
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')

    try:
        # Khởi tạo Client và trỏ trúng DB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client['car_data_platform']
        print("\n================================================================")
        print("[DATABASE PIPELINE] ĐANG KHỞI TẠO ĐỒNG BỘ LÊN MONGODB ATLAS...")
        print("================================================================")
    except Exception as connection_error:
        print(f"[LỖI KẾT NỐI] Không thể kết nối tới Cluster MongoDB: {connection_error}")
        return

    BATCH_SIZE = 2000

    # =========================================================================
    # PHẦN 1: ĐỒNG BỘ DỮ LIỆU THÔ (BRONZE) CHO RAG
    # =========================================================================
    print(f"\n[*] BƯỚC 1: Đang nạp dữ liệu từ tầng Bronze thô tại: {bronze_path}")
    if not os.path.exists(bronze_path):
        print(f"   [!] Bỏ qua tầng Bronze do không tìm thấy file.")
    else:
        try:
            df_bronze = pd.read_csv(bronze_path)
            # Lấp đầy NaN bằng chuỗi rỗng để tránh lỗi khi nạp lên MongoDB
            df_bronze = df_bronze.fillna("")
            bronze_records = df_bronze.to_dict(orient='records')
            bronze_collection = db['bronze_cars_collection']
            
            total_bronze = len(bronze_records)
            print(f"   [*] Tìm thấy {total_bronze} dòng xe thô. Bắt đầu đẩy Bulk Upsert...")
            
            b_upserted = 0
            b_modified = 0
            
            for i in range(0, total_bronze, BATCH_SIZE):
                batch_records = bronze_records[i:i + BATCH_SIZE]
                operations = []
                for record in batch_records:
                    # Kiểm tra đảm bảo record có trường id
                    record_id = str(record.get('id', ''))
                    if record_id:
                        filter_query = {'id': record_id}
                        update_query = {'$set': record}
                        operations.append(UpdateOne(filter_query, update_query, upsert=True))
                
                if operations:
                    result = bronze_collection.bulk_write(operations, ordered=False)
                    b_upserted += result.upserted_count
                    b_modified += result.modified_count
                    progress = min(((i + BATCH_SIZE) / total_bronze) * 100, 100)
                    print(f"      + Bronze: {min(i + BATCH_SIZE, total_bronze)}/{total_bronze} ({progress:.2f}%)")
                    
            print(f"   [SUCCESS] Tầng Bronze: Chèn mới {b_upserted} | Cập nhật cũ {b_modified}")
        except Exception as e:
            print(f"   [LỖI] Xảy ra lỗi trong luồng Bronze: {e}")

    # =========================================================================
    # PHẦN 2: ĐỒNG BỘ DỮ LIỆU SILVER CHO MACHINE LEARNING
    # =========================================================================
    print(f"\n[*] BƯỚC 2: Đang nạp dữ liệu từ tầng Silver sạch tại: {silver_path}")
    if not os.path.exists(silver_path):
        print(f"   [!] Bỏ qua tầng Silver do không tìm thấy file.")
    else:
        try:
            df_silver = pd.read_csv(silver_path)
            # Lấp các ô khuyết nhãn số học hoặc phân loại cơ bản
            df_silver = df_silver.fillna({
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

            silver_records = df_silver.to_dict(orient='records')
            silver_collection = db['silver_cars_collection']
            
            total_silver = len(silver_records)
            print(f"   [*] Tìm thấy {total_silver} dòng xe sạch. Bắt đầu đẩy Bulk Upsert...")
            
            s_upserted = 0
            s_modified = 0

            for i in range(0, total_silver, BATCH_SIZE):
                batch_records = silver_records[i:i + BATCH_SIZE]
                operations = []
                for record in batch_records:
                    record_id = str(record.get('id', ''))
                    if record_id:
                        filter_query = {'id': record_id}
                        update_query = {'$set': record}
                        operations.append(UpdateOne(filter_query, update_query, upsert=True))
                
                if operations:
                    result = silver_collection.bulk_write(operations, ordered=False)
                    s_upserted += result.upserted_count
                    s_modified += result.modified_count
                    progress = min(((i + BATCH_SIZE) / total_silver) * 100, 100)
                    print(f"      + Silver: {min(i + BATCH_SIZE, total_silver)}/{total_silver} ({progress:.2f}%)")
                    
            print(f"   [SUCCESS] Tầng Silver: Chèn mới {s_upserted} | Cập nhật cũ {s_modified}")
        except Exception as e:
            print(f"   [LỖI] Xảy ra lỗi trong luồng Silver: {e}")

    print("\n================================================================")
    print("[DATABASE PIPELINE] ĐỒNG BỘ HOÀN TẤT TỚI MONGODB ATLAS")
    print("================================================================\n")
    client.close()

if __name__ == "__main__":
    push_data_to_mongodb()