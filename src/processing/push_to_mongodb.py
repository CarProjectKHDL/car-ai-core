import os
import sys
import pandas as pd
from pymongo import MongoClient, UpdateOne

def push_silver_to_mongodb():
    mongo_uri = os.environ.get("MONGO_URI")
    
    if not mongo_uri:
        print("Không tìm thấy cấu hình MONGO_URI")
        sys.exit(1)

    # Xác định đường dẫn tuyệt đối tới file tầng Silver
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')

    if not os.path.exists(silver_path):
        print(f"Không tìm thấy file dữ liệu")
        return

    # Nạp dữ liệu bằng Pandas và chuyển đổi thành cấu trúc JSON-like
    df = pd.read_csv(silver_path)
    # Đổi NaN thành chuỗi rỗng
    df = df.fillna("")
    records = df.to_dict(orient='records')
    print(f"[*] Đang chuẩn bị đồng bộ {len(records)} bản ghi từ tầng Silver lên MongoDB")

    # Khởi tạo Client kết nối tới MongoDB
    client = MongoClient(mongo_uri)
    db = client['car_market_db']
    collection = db['silver_cars']

    # BULK WRITE & UPSERT tối ưu hóa băng thông đường truyền
    operations = []
    for record in records:
        # Sử dụng id bài đăng làm khóa chính
        filter_query = {'id': str(record['id'])}
        update_query = {'$set': record}
        
        operations.append(UpdateOne(filter_query, update_query, upsert=True))

    if operations:
        try:
            result = collection.bulk_write(operations)
            print("\n================================================================")
            print("[DATABASE PIPELINE] MISSION SUCCESSFULLY")
            print(f"   - Số lượng xe mới được chèn vào kho: {result.upserted_count}")
            print(f"   - Số lượng xe cũ được cập nhật thông tin mới: {result.modified_count}")
            print("================================================================")
        except Exception as e:
            print(f"Lỗi: {e}")
    
    client.close()

if __name__ == "__main__":
    push_silver_to_mongodb()