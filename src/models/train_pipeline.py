import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder

def load_master_msrp_dictionary(base_dir):
    dict_path = os.path.join(base_dir, 'data', 'silver', 'brand_model_dictionary.json')
    if not os.path.exists(dict_path):
        return {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_heuristic_guardrail(df_input, y_pred, master_dict):
    """
    Guardrail Layer nâng cao:
    Bảo vệ xung lực xe sang dựa trên quy đổi mệnh giá MSRP chuẩn của Việt Nam
    """
    if isinstance(y_pred, pd.Series):
        y_pred_bounded = y_pred.to_numpy().astype(float).copy()
    else:
        y_pred_bounded = np.array(y_pred, dtype=float).copy()

    adjusted_count = 0
    msrp_lookup = {}

    for brand_key, brand_data in master_dict.items():
        trims_data = brand_data.get("trims", {})
        for core_key, core_trims in trims_data.items():
            for trim_name, msrp_val in core_trims.items():
                msrp_lookup[trim_name.lower().strip()] = float(msrp_val)

    df_reset = df_input.reset_index(drop=True)

    for idx, row in df_reset.iterrows():
        model_trim_raw = str(row.get('model_trim', '')).lower().strip()
        model_core_raw = str(row.get('model_core', '')).lower().strip()

        msrp_cap = msrp_lookup.get(model_trim_raw) or msrp_lookup.get(model_core_raw)

        # Chỉ bảo vệ áp trần nếu từ điển trả về giá trị hợp lệ lớn hơn 100 Triệu VNĐ
        if msrp_cap and isinstance(msrp_cap, (int, float)) and msrp_cap > 100.0:
            max_allowed_price = msrp_cap * 1.25  # Biên độ bao dung 125% cho xe cũ chất lượng cao/độ đồ chơi

            # Chỉ can thiệp nếu giá đoán vượt trần một cách phi thực tế trong dải phổ thông
            if y_pred_bounded[idx] > max_allowed_price and y_pred_bounded[idx] < 4000.0:
                y_pred_bounded[idx] = max_allowed_price
                adjusted_count += 1

        if y_pred_bounded[idx] < 10.0:
            y_pred_bounded[idx] = 10.0

    if adjusted_count > 0:
        print(f"[Guardrail Layer] Đã tối ưu hóa áp trần Heuristic Price Cap cho {adjusted_count} dòng xe.")

    return y_pred_bounded

def execute_advanced_training_pipeline():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Ưu tiên đọc từ file gold_cars_train.csv đích thực của hệ thống
    gold_path = os.path.join(BASE_DIR, 'data', 'gold', 'gold_cars_train.csv')
    if not os.path.exists(gold_path):
        gold_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')
        print(f"[!] Chạy chế độ Fallback liên kết luồng dữ liệu: {gold_path}")

    df = pd.read_csv(gold_path)

    # ----------------------------------------------------------------------------------
    # DYNAMIC CATEGORICAL FALLBACK CHỐNG NHÃN HIẾM CỰC ĐOAN
    # ----------------------------------------------------------------------------------
    # Nếu nhóm (brand + model_trim) có >= 5 mẫu thì lấy Trim chi tiết,
    # ngược lại fallback về Core Model an toàn để TargetEncoder có đủ dữ liệu học
    df['trim_counts'] = df.groupby(['brand', 'model_trim'])['id'].transform('count')
    df['final_model_feature'] = np.where(
        df['trim_counts'] >= 5,
        df['brand'].astype(str) + "_" + df['model_trim'].astype(str),
        df['brand'].astype(str) + "_" + df['model_core'].astype(str)
    )
    df = df.drop(columns=['trim_counts'])

    print(f"[*] Nạp dữ liệu thành công: {len(df)} dòng siêu sạch từ tầng Gold.")

    X = df.drop(columns=['price_numeric'])
    y = df['price_numeric']

    # Tách tập kiểm thử độc lập tuyệt đối chống Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ĐỊNH NGHĨA PIPELINES XỬ LÝ TOÁN HỌC
    # Lấp các ô NaN (odo=0 đã chuyển NaN ở tầng Gold) bằng trung vị trước khi tính Log
    log_odo_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(lambda x: np.log1p(x.astype(float)), validate=False)),
        ('scaler', StandardScaler())
    ])

    numeric_features = ['car_age', 'odo_per_year', 'is_first_owner', 'has_upgrades', 'is_verified_accident_free', 'is_brand_new']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # SCHEMA ĐÃ LOẠI BỎ HOÀN TOÀN BIẾN LOCATION ĐỂ CHỐNG THƯA MA TRẬN
    low_cardinality_features = ['brand', 'transmission', 'fuel', 'origin', 'condition']
    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Giá trị Trung vị (Median) của tập Train làm chốt chặn an toàn cho Unseen Category
    train_median_price = float(y_train.median())
    print(f"[*] Chốt chặn an toàn toán học cho Unseen Category (Global Median): {train_median_price} Triệu VNĐ")

    high_cardinality_features = ['final_model_feature']
    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('target_enc', TargetEncoder(
            smoothing=50.0,
            handle_unknown=train_median_price,  # Ép gán thẳng bằng Trung vị Train nếu gặp nhãn lạ ở tập Test
            handle_missing=train_median_price
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('log_odo', log_odo_transformer, ['odo_numeric']),
            ('num', numeric_transformer, numeric_features),
            ('cat_low', low_card_transformer, low_cardinality_features),
            ('cat_high', high_card_transformer, high_cardinality_features)
        ],
        remainder='drop'
    )

    print("[*] Đang tiến hành xử lý kỹ nghệ đặc trưng song song...")
    X_train_encoded = preprocessor.fit_transform(X_train, y_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Kiểm tra category nào trong test không có trong train (chỉ để giám sát, không can thiệp)
    train_categories = set(X_train['final_model_feature'].unique())
    test_categories = set(X_test['final_model_feature'].unique())
    unseen = test_categories - train_categories
    print(f"[*] Categories chỉ có trong test (unseen): {len(unseen)}")
    if unseen:
        print(f"    {unseen}")

    # KHỞI TẠO BỘ BA NGUYÊN TỬ TẦNG 0
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    lgb = LGBMRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
    cat = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, random_seed=42, verbose=0)

    base_models = {'XGBoost': xgb, 'LightGBM': lgb, 'CatBoost': cat}

    # KIẾN TRÚC OUT-OF-FOLD STACKING
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros((X_train_encoded.shape[0], len(base_models)))
    fitted_models = {name: [] for name in base_models.keys()}
    model_names = list(base_models.keys())

    print("[*] Khởi chạy huấn luyện OOF Stacking với 5 Folds trên ma trận đặc trưng đậm đặc...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train)):
        X_tr, X_va = X_train_encoded[train_idx], X_train_encoded[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for idx, name in enumerate(model_names):
            from sklearn.base import clone
            instance = clone(base_models[name])
            instance.fit(X_tr, y_tr)
            oof_predictions[val_idx, idx] = instance.predict(X_va)
            fitted_models[name].append(instance)

    print("[*] Đang tối ưu hóa Meta-Learner (Random Forest Regressor) ở Tầng 1...")
    meta_learner = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    meta_learner.fit(oof_predictions, y_train)

    print("[*] Đang tiến hành dự đoán trên tập Test biệt lập...")
    meta_features_test = np.zeros((X_test_encoded.shape[0], len(base_models)))
    for idx, name in enumerate(model_names):
        fold_preds = np.column_stack([model.predict(X_test_encoded) for model in fitted_models[name]])
        meta_features_test[:, idx] = np.mean(fold_preds, axis=1)

    y_pred_raw = meta_learner.predict(meta_features_test)

    # Áp dụng lớp phòng vệ Hậu xử lý chuẩn hóa
    master_dict = load_master_msrp_dictionary(BASE_DIR)
    y_pred_final = apply_heuristic_guardrail(X_test, y_pred_raw, master_dict)

    # ĐÁNH GIÁ KẾT QUẢ CUỐI CÙNG
    r2 = r2_score(y_test, y_pred_final)
    mae = mean_absolute_error(y_test, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    ratio = rmse / mae
    df_test_eval = df.iloc[X_test.index].copy()
    df_test_eval['predicted'] = y_pred_final
    df_test_eval['abs_err'] = abs(df_test_eval['price_numeric'] - df_test_eval['predicted'])
    df_test_eval = df_test_eval.sort_values('abs_err', ascending=False)
    print("\n[*] Top 10 dòng xe có sai số dự đoán lớn nhất:")
    print(df_test_eval[['brand', 'model_trim', 'year', 'price_numeric', 'predicted', 'abs_err']].head(10))

    print("\n" + "=" * 50)
    print("🚀 KẾT QUẢ TỐI ƯU HÓA NÂNG CAO ĐÍCH THỰC (PRODUCTION READY)")
    print("=" * 50)
    print(f"-> Hệ số xác định R² thực tế: {r2 * 100:.2f}%")
    print(f"-> Sai số MAE mới: {mae:.2f} Triệu VNĐ")
    print(f"-> Sai số RMSE mới: {rmse:.2f} Triệu VNĐ")
    print(f"-> Tỷ lệ biến động RMSE/MAE: {ratio:.2f}")
    print("=" * 50)

    print("\n--- [GIÁM SÁT PHÂN PHỐI FINAL_MODEL_FEATURE] ---")
    fmf_counts = df['final_model_feature'].value_counts()
    print(f"Tổng số category: {len(fmf_counts)}")
    print(f"Category có < 5 mẫu: {(fmf_counts < 5).sum()} ({(fmf_counts < 5).sum() / len(fmf_counts) * 100:.1f}%)")
    print(f"Category có < 10 mẫu: {(fmf_counts < 10).sum()}")
    print(f"Category chỉ có 1 mẫu: {(fmf_counts == 1).sum()}")
    print("\nTop 10 category hiếm nhất:")
    print(fmf_counts.tail(10))
    print("--------------------------------------------------\n")

    # ----------------------------------------------------------------------------------
    # SÀNG LỌC KÈO THƠM THƯƠNG MẠI (COMMERCIAL DEALS)
    # ----------------------------------------------------------------------------------
    print("[*] Đang kích hoạt Engine quét cấu trúc giá tìm Kèo Thơm Thương Mại...")

    # Tạo bản sao tập Test để đối chiếu kết quả dự đoán
    test_df = X_test.copy()
    test_df['price_numeric'] = y_test.values
    test_df['predicted_price'] = np.round(y_pred_final, 1)

    # Tính toán lại các biên độ chênh lệch thực tế giữa giá rao bán và giá AI định giá
    test_df['price_diff_absolute'] = test_df['price_numeric'] - test_df['predicted_price']
    test_df['price_diff_pct'] = test_df['price_diff_absolute'] / test_df['predicted_price']

    # BỘ LỌC KÈO THƠM CHUẨN CÔNG NGHIỆP:
    # - Giá thực tế rẻ hơn giá AI định giá từ 8% đến 25%
    # - Số tiền tiết kiệm được từ 15 Triệu đến 120 Triệu VNĐ
    # - Xe không bị tai nạn/ngập nước (Cam kết từ mô tả) và thông tin rõ ràng
    sweet_deals = test_df[
        (test_df['price_diff_pct'] <= -0.08) & (test_df['price_diff_pct'] >= -0.25) &
        (test_df['price_diff_absolute'] <= -15) & (test_df['price_diff_absolute'] >= -120) &
        (test_df['brand'] != 'Unknown') &
        (test_df['is_verified_accident_free'] == 1)
    ].copy()

    # Sắp xếp theo tỷ lệ giảm giá hời nhất
    sweet_deals = sweet_deals.sort_values(by='price_diff_pct', ascending=True)

    # Lấy Top 30 xe có giá hời nhất thị trường
    top_30_deals = sweet_deals.head(30)

    # Đổi tên cột phục vụ tương thích ngược (Backward Compatibility) nếu cần
    if 'model_trim' in top_30_deals.columns:
        top_30_deals = top_30_deals.rename(columns={'model_trim': 'model'})

    # Kết xuất dữ liệu xuống tầng Gold phục vụ làm sản phẩm đầu ra thương mại
    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    gold_deals_path = os.path.join(gold_dir, 'gold_deals.csv')

    top_30_deals.to_csv(gold_deals_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Đã trích xuất thành công {len(top_30_deals)} Kèo Thơm siêu chuẩn vào tệp: {gold_deals_path}")

if __name__ == "__main__":
    execute_advanced_training_pipeline()