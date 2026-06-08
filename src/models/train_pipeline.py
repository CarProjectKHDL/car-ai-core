import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_gold_ai_pipeline():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    silver_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')
    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    os.makedirs(gold_dir, exist_ok=True)
    
    if not os.path.exists(silver_path):
        print(f"Không tìm thấy dữ liệu")
        return

    df = pd.read_csv(silver_path)
    print(f"[GOLD LAYER] Đang nạp {len(df)} dòng dữ liệu từ tầng Silver...")

    # Bộ lọc loại bỏ tin rác/tin ảo
    df = df[(df['price_numeric'] >= 20) & (df['price_numeric'] <= 40000)]
    df = df[df['year'] >= 1990]
    
    # PHÂN TÁCH BIẾN ĐẶC TRƯNG & BIẾN MỤC TIÊU
    features = ['brand', 'model', 'year', 'car_age', 'odo_numeric', 'odo_per_year', 
                'transmission', 'fuel', 'origin', 'condition', 'is_first_owner', 'has_upgrades']
    target = 'price_numeric'

    X = df[features]
    y = df[target]

    # Chia tập dữ liệu Train/Test theo tỷ lệ 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🔹 Phân tách dữ liệu: Tập Train ({X_train.shape[0]} dòng) | Tập Test ({X_test.shape[0]} dòng)")

    # BỘ TIỀN XỬ LÝ
    categorical_features = ['brand', 'model', 'transmission', 'fuel', 'origin', 'condition']
    numerical_features = ['year', 'car_age', 'odo_numeric', 'odo_per_year', 'is_first_owner', 'has_upgrades']

    # OneHotEncoder dùng handle_unknown='ignore' để phòng vệ nếu tập Test dính xechưa từng thấy ở tập Train
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # HUÂN LUYỆN VÀ ĐỐI CHIẾU HAI MÔ HÌNH
    print("\nĐang chạy mô hình 1: Random Forest Regressor...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)

    print("Đang chạy mô hình 2: XGBoost Regressor...")
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
    ])
    xgb_pipeline.fit(X_train, y_train)

    # ĐÁNH GIÁ VÀ SO SÁNH SAI SỐ TOÁN HỌC
    pipelines = {"Random Forest": rf_pipeline, "XGBoost": xgb_pipeline}
    best_r2 = -1
    best_model_name = ""
    best_pipeline = None

    print("\nKẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP:")
    print("=" * 70)
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Mô hình [{name}]:")
        print(f"   - Hệ số xác định R² Score         : {r2 * 100:.2f}%")
        print(f"   - Sai số tuyệt đối trung bình (MAE) : {mae:.2f} Triệu đồng")
        print(f"   - Sai số bình phương trung bình (RMSE): {rmse:.2f} Triệu đồng")
        print("-" * 70)

        # Lựa chọn mô hình xuất sắc hơn
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"Mô hình được chọn: {best_model_name} (R² = {best_r2*100:.2f}%)")

    model_save_path = os.path.join(gold_dir, 'best_car_pricer_model.pkl')
    joblib.dump(best_pipeline, model_save_path)
    print(f"Đã đóng gói và lưu mô hình tại: '{model_save_path}'")

    # THUẬT TOÁN PHÁT HIỆN KÈO NGON (ANOMALY RESIDUAL DETECTION)
    print("\nANOMALY RESIDUAL DETECTION...")
    
    # Tạo bản sao dữ liệu gốc
    market_df = df.copy()
    market_df['predicted_price'] = best_pipeline.predict(X)
    
    # Tính phần trăm chênh lệch: (Giá thực tế - Giá AI định giá) / Giá AI định giá
    market_df['price_diff_pct'] = (market_df['price_numeric'] - market_df['predicted_price']) / market_df['predicted_price']
    
    # TIÊU CHUẨN KÈO NGON: 
    # - Giá rẻ hơn giá trị thật do AI tính toán từ 15% trở lên
    # - Odo không quá lớn
    sweet_deals = market_df[(market_df['price_diff_pct'] <= -0.15) & (market_df['odo_per_year'] < 25000)]
    
    # Sắp xếp xe có mức giảm giá sâu nhất lên đầu
    sweet_deals = sweet_deals.sort_values(by='price_diff_pct')

    # Trích xuất
    gold_deals_path = os.path.join(gold_dir, 'gold_deals.csv')
    output_cols = ['id', 'brand', 'model', 'year', 'price_numeric', 'predicted_price', 'price_diff_pct', 'url']
    
    sweet_deals[output_cols].head(30).to_csv(gold_deals_path, index=False, encoding='utf-8-sig')
    print(f"Đã lọc ra {len(sweet_deals)} xe.")
    print(f"Danh sách 30 xe giá hời nhất đã được cập nhật tại: '{gold_deals_path}'")

if __name__ == "__main__":
    run_gold_ai_pipeline()