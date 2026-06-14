import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def transform_log1p(x):
    return np.log1p(x.astype(float))

def load_master_msrp_dictionary(base_dir):
    dict_path = os.path.join(base_dir, 'data', 'silver', 'brand_model_dictionary.json')
    if not os.path.exists(dict_path):
        return {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_heuristic_guardrail(df_input, y_pred, master_dict):
    """
    Guardrail Layer:
    Bảo vệ xung lực xe sang dựa trên quy đổi mệnh giá MSRP chuẩn
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
        car_age = float(row.get('car_age', 0.0))

        msrp_cap = msrp_lookup.get(model_trim_raw) or msrp_lookup.get(model_core_raw)

        if msrp_cap and isinstance(msrp_cap, (int, float)) and msrp_cap > 100.0:
            # LOGIC GUARDRAIL ĐỘNG SIẾT CHẶT THEO KHẤU HAO THỜI GIAN
            # Xe 0-1 tuổi: Trần 125% MSRP. Xe 10 tuổi: Trần siết về 75% MSRP.
            dynamic_ratio = max(1.25 - (car_age * 0.05), 0.75)
            max_allowed_price = msrp_cap * dynamic_ratio

            if y_pred_bounded[idx] > max_allowed_price:
                y_pred_bounded[idx] = max_allowed_price
                adjusted_count += 1

        if y_pred_bounded[idx] < 10.0:
            y_pred_bounded[idx] = 10.0

    if adjusted_count > 0:
        print(f"[Dynamic Guardrail] Đã tối ưu hóa áp trần động theo tuổi đời cho {adjusted_count} dòng xe.")

    return y_pred_bounded

def build_feature_preprocessor():

    log_odo_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(transform_log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    numeric_features = ['car_age', 'odo_per_year', 'is_first_owner', 'has_upgrades', 
                        'is_verified_accident_free', 'is_brand_new', 'high_usage', 'is_luxury_brand']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    low_cardinality_features = ['brand', 'transmission', 'fuel', 'origin', 'condition', 'age_bucket']
    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    high_cardinality_features = ['final_model_feature']
    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('target_enc', TargetEncoder(
            categories='auto',
            target_type='continuous',
            cv=5,
            random_state=42
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
    return preprocessor

def train_stacking_core(X_train, y_train_log, X_test, preprocessor, meta_alpha=10.0):
    """
    Logic cốt lõi Stacking Ensemble 5-Fold Out-Of-Fold cho từng phân hệ
    """
    X_train_encoded = preprocessor.fit_transform(X_train, y_train_log)
    X_test_encoded = preprocessor.transform(X_test)

    # Khởi tạo bộ ba nguyên tử Tầng 0
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective='reg:pseudohubererror', random_state=42, n_jobs=-1)
    lgb = LGBMRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
    cat = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, random_seed=42, verbose=0)

    base_models = {'XGBoost': xgb, 'LightGBM': lgb, 'CatBoost': cat}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros((X_train_encoded.shape[0], len(base_models)))
    fitted_models = {name: [] for name in base_models.keys()}
    model_names = list(base_models.keys())

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train_log)):
        X_tr, X_va = X_train_encoded[train_idx], X_train_encoded[val_idx]
        y_tr, y_va = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

        for idx, name in enumerate(model_names):
            from sklearn.base import clone
            instance = clone(base_models[name])
            instance.fit(X_tr, y_tr)
            oof_predictions[val_idx, idx] = instance.predict(X_va)
            fitted_models[name].append(instance)

    # Tối ưu hóa Meta-Learner ở Tầng 1
    meta_learner = Ridge(alpha=meta_alpha)
    meta_learner.fit(oof_predictions, y_train_log)

    # Dự đoán trên tập Test
    meta_features_test = np.zeros((X_test_encoded.shape[0], len(base_models)))
    for idx, name in enumerate(model_names):
        fold_preds = np.column_stack([model.predict(X_test_encoded) for model in fitted_models[name]])
        meta_features_test[:, idx] = np.mean(fold_preds, axis=1)

    y_pred_log = meta_learner.predict(meta_features_test)
    
    # Đóng gói các cấu phần mô hình đã fitted thành một dictionary tổng hợp
    fitted_pipeline_objects = {
        "fitted_base_models": fitted_models,
        "fitted_meta_learner": meta_learner,
        "model_names": model_names
    }
    
    return np.expm1(y_pred_log), fitted_pipeline_objects

def execute_advanced_training_pipeline():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    gold_path = os.path.join(BASE_DIR, 'data', 'gold', 'gold_cars_train.csv')
    
    if not os.path.exists(gold_path):
        gold_path = os.path.join(BASE_DIR, 'data', 'silver', 'silver_cars_clean.csv')
        print(f"[!] Chạy chế độ Fallback liên kết luồng dữ liệu: {gold_path}")

    df = pd.read_csv(gold_path)

    # Dynamic Categorical Fallback chống nhãn hiếm cực đoan
    def is_generic_trim(model_trim, model_core):
        t = str(model_trim).lower().strip().replace('-', '').replace(' ', '')
        c = str(model_core).lower().strip().replace('-', '').replace(' ', '')
        return t == c or len(t) <= 2

    df['trim_counts'] = df.groupby(['brand', 'model_trim'])['id'].transform('count')
    df['trim_is_generic'] = df.apply(lambda r: is_generic_trim(r['model_trim'], r['model_core']), axis=1)
    df['final_model_feature'] = np.where(
        (df['trim_counts'] >= 5) & (~df['trim_is_generic']),
        df['brand'].astype(str) + "_" + df['model_trim'].astype(str),
        df['brand'].astype(str) + "_" + df['model_core'].astype(str)
    )
    df = df.drop(columns=['trim_counts', 'trim_is_generic'])

    # Map giá trị MSRP từ Dictionary để thực hiện phân luồng chia để trị
    master_dict = load_master_msrp_dictionary(BASE_DIR)
    msrp_lookup = {}
    for brand_key, brand_data in master_dict.items():
        trims_data = brand_data.get("trims", {})
        for core_key, core_trims in trims_data.items():
            for trim_name, msrp_val in core_trims.items():
                if msrp_val and isinstance(msrp_val, (int, float)):
                    msrp_lookup[trim_name.lower().strip()] = float(msrp_val)

    # Tính median giá thực tế theo brand từ chính data - dùng làm fallback khi khuyết MSRP
    brand_median_price = df.groupby('brand')['price_numeric'].median()

    df['msrp_proxy'] = df.apply(
        lambda r: msrp_lookup.get(str(r['model_trim']).lower().strip())
                  or msrp_lookup.get(str(r['model_core']).lower().strip())
                  or float(brand_median_price.get(r['brand'], 500.0)),
        axis=1
    )

    # Debug
    print("\n--- [KIỂM TRA MSRP PROXY THEO BRAND] ---")
    brand_routing = df.groupby('brand').agg(
        n=('id', 'count'),
        msrp_proxy_median=('msrp_proxy', 'median'),
        price_median=('price_numeric', 'median')
    ).sort_values('msrp_proxy_median', ascending=False).head(15)
    print(brand_routing)
    n_model_b = (df['msrp_proxy'] > 1500).sum()
    print(f"\nNgưỡng 1500tr: Model A={len(df)-n_model_b}, Model B={n_model_b}")
    n_model_b2 = (df['msrp_proxy'] > 2000).sum()
    print(f"Ngưỡng 2000tr: Model A={len(df)-n_model_b2}, Model B={n_model_b2}")
    print("------------------------------------------")

    # --- FEATURE ENGINEERING NGHIỆP VỤ ĐẶC TRƯNG ---
    # 1. Cờ xe chạy dịch vụ
    df['high_usage'] = (df['odo_per_year'] > 25000).astype(int)

    # Phân đoạn tuổi xe
    # Chia làm 3 nhóm: Lướt (0-3 năm), Trung hạn (4-6 năm), Đời sâu (7-10 năm)
    df['age_bucket'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 11], labels=['Luot', 'Trung_Han', 'Doi_Sau']).astype(str)

    # Phân nhóm xe sang phổ thông
    luxury_list = ['mercedes benz', 'bmw', 'audi', 'lexus', 'porsche', 'volvo']
    df['is_luxury_brand'] = df['brand'].str.lower().isin(luxury_list).astype(int)

    # ----------------------------------------------------------------------------------
    # CHIẾN THUẬT CHIA ĐỂ TRỊ: BẺ ĐÔI DỮ LIỆU THÀNH 2 PHÂN HỆ ĐỘC LẬP
    # ----------------------------------------------------------------------------------
    MSRP_THRESHOLD = 1500.0  # Ngưỡng chia cắt phân khúc: 1.5 Tỷ VNĐ
    df_A = df[df['msrp_proxy'] <= MSRP_THRESHOLD].copy().reset_index(drop=True)
    df_B = df[df['msrp_proxy'] > MSRP_THRESHOLD].copy().reset_index(drop=True)

    print(f"\n[HỆ THỐNG DUAL-CORE AI] Đã chia đôi luồng toán học:")
    print(f" -> Model A (Phổ thông <= 1.5 Tỷ): {len(df_A)} dòng dữ liệu đậm đặc.")
    print(f" -> Model B (Hạng sang > 1.5 Tỷ)  : {len(df_B)} dòng dữ liệu phi tuyến tính.")

    # Tách tập kiểm thử độc lập đồng bộ cho cả 2 mô hình chống rò rỉ dữ liệu
    X_A = df_A.drop(columns=['price_numeric'])
    y_A = df_A['price_numeric']
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)
    
    X_B = df_B.drop(columns=['price_numeric'])
    y_B = df_B['price_numeric']
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

    # Bản sao ghi nhớ chỉ mục để trích xuất đánh giá cuối cùng
    orig_test_index_A = X_test_A.index.copy()
    orig_test_index_B = X_test_B.index.copy()

    X_train_A, y_train_A, X_test_A, y_test_A = X_train_A.reset_index(drop=True), y_train_A.reset_index(drop=True), X_test_A.reset_index(drop=True), y_test_A.reset_index(drop=True)
    X_train_B, y_train_B, X_test_B, y_test_B = X_train_B.reset_index(drop=True), y_train_B.reset_index(drop=True), X_test_B.reset_index(drop=True), y_test_B.reset_index(drop=True)

    # Chuẩn bị Log-Target và Kỹ nghệ đặc trưng độc lập cho từng phân hệ
    y_train_A_log = np.log1p(y_train_A)
    y_train_B_log = np.log1p(y_train_B)
    
    preprocessor_A = build_feature_preprocessor()
    preprocessor_B = build_feature_preprocessor()

    # --- KHỞI CHẠY HUẤN LUYỆN MODEL A (XE PHỔ THÔNG) ---
    print("\n>>> [MODEL A] Đang huấn luyện Stacking 5-Fold cho xe Phổ thông...")
    y_pred_A_raw, pipeline_objs_A = train_stacking_core(X_train_A, y_train_A_log, X_test_A, preprocessor_A, meta_alpha=10.0)
    y_pred_A_final = apply_heuristic_guardrail(X_test_A, y_pred_A_raw, master_dict)

    # --- KHỞI CHẠY HUẤN LUYỆN MODEL B (XE SANG) ---
    print("\n>>> [MODEL B] Đang huấn luyện Stacking 5-Fold cho xe Hạng sang...")
    # Tăng trọng số phạt alpha=25.0 để kiểm soát các biến động Option dị biệt của Porsche/Lexus
    y_pred_B_raw, pipeline_objs_B = train_stacking_core(X_train_B, y_train_B_log, X_test_B, preprocessor_B, meta_alpha=25.0)
    y_pred_B_final = apply_heuristic_guardrail(X_test_B, y_pred_B_raw, master_dict)

    # ----------------------------------------------------------------------------------
    # HỢP NHẤT KẾT QUẢ
    # ----------------------------------------------------------------------------------
    df_eval_A = df_A.loc[orig_test_index_A].copy()
    df_eval_A['predicted'] = y_pred_A_final
    
    df_eval_B = df_B.loc[orig_test_index_B].copy()
    df_eval_B['predicted'] = y_pred_B_final

    df_total_eval = pd.concat([df_eval_A, df_eval_B], ignore_index=True)
    df_total_eval['abs_err'] = abs(df_total_eval['price_numeric'] - df_total_eval['predicted'])
    df_total_eval['abs_pct_err'] = (df_total_eval['abs_err'] / df_total_eval['price_numeric']) * 100

    # ĐÁNH GIÁ CHỈ SỐ TOÀN DIỆN
    r2_all = r2_score(df_total_eval['price_numeric'], df_total_eval['predicted'])
    mae_all = mean_absolute_error(df_total_eval['price_numeric'], df_total_eval['predicted'])
    rmse_all = np.sqrt(mean_squared_error(df_total_eval['price_numeric'], df_total_eval['predicted']))
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ TỐI ƯU KIẾN TRÚC DUAL-CORE AI SẠCH BÓNG BIAS")
    print("=" * 60)
    print(f"-> Hệ số xác định R² toàn sàn: {r2_all * 100:.2f}%")
    print(f"-> Sai số MAE toàn sàn      : {mae_all:.2f} Triệu VNĐ")
    print(f"-> Sai số RMSE toàn sàn     : {rmse_all:.2f} Triệu VNĐ")
    print(f"-> Tỷ lệ biến động RMSE/MAE  : {rmse_all / mae_all:.2f}")
    print("=" * 60)

    # Phân rã hiệu năng theo từng phân khúc giá để kiểm chứng
    print("\n--- [SAI SỐ TƯƠNG ĐỐI THEO PHÂN KHÚC GIÁ THỰC TẾ] ---")
    bins = [0, 300, 500, 800, 1200, 2000, 99999]
    labels = ['<300tr', '300-500tr', '500-800tr', '800tr-1.2ty', '1.2-2ty', '>2ty']
    df_total_eval['segment'] = pd.cut(df_total_eval['price_numeric'], bins=bins, labels=labels)

    segment_stats = df_total_eval.groupby('segment', observed=True).agg(
        so_luong=('price_numeric', 'count'),
        mape_pct=('abs_pct_err', 'mean'),
        mae_trieu=('abs_err', 'mean')
    )
    print(segment_stats)

    print(f"\nMAPE Tổng thể hệ thống: {df_total_eval['abs_pct_err'].mean():.2f}%")
    print(f"Tỷ lệ xe đạt độ chính xác cao (Sai số < 10%): {((df_total_eval['abs_pct_err'] < 10).mean() * 100):.1f}%")
    print(f"Tỷ lệ xe đạt độ chính xác trung bình (Sai số < 15%): {((df_total_eval['abs_pct_err'] < 15).mean() * 100):.1f}%")

    print("\n--- [KIỂM TRA HIỆU NĂNG MẪU XE PHỔ THÔNG KHỐT LÕI] ---")
    for common_model in ['vios', 'city', 'fadil', 'morning']:
        mask = df_total_eval['model_trim'].str.lower().str.contains(common_model, na=False)
        if mask.any():
            print(f"\n--- PHÂN TÍCH THỰC NGHIỆM: {common_model.upper()} ---")
            print(df_total_eval[mask][['brand', 'model_trim', 'year', 'price_numeric', 'predicted', 'abs_pct_err']].head(3))

    # ----------------------------------------------------------------------------------
    # SÀNG LỌC KÈO THƠM THƯƠNG MẠI (COMMERCIAL ENGINE)
    # ----------------------------------------------------------------------------------
    print("\n[*] Đang kích hoạt Engine quét cấu trúc giá tìm Kèo Thơm Thương Mại...")
    
    # Chuẩn bị dữ liệu test tổng hợp
    df_total_eval['predicted_price'] = np.round(df_total_eval['predicted'], 1)
    df_total_eval['price_diff_absolute'] = df_total_eval['price_numeric'] - df_total_eval['predicted_price']
    df_total_eval['price_diff_pct'] = df_total_eval['price_diff_absolute'] / df_total_eval['predicted_price']

    sweet_deals = df_total_eval[
        (df_total_eval['price_diff_pct'] <= -0.08) & (df_total_eval['price_diff_pct'] >= -0.25) &
        (df_total_eval['price_diff_absolute'] <= -15) & (df_total_eval['price_diff_absolute'] >= -120) &
        (df_total_eval['brand'] != 'Unknown') &
        (df_total_eval['is_verified_accident_free'] == 1)
    ].copy()

    top_30_deals = sweet_deals.sort_values(by='price_diff_pct', ascending=True).head(30)
    
    if 'model_trim' in top_30_deals.columns:
        top_30_deals = top_30_deals.rename(columns={'model_trim': 'model'})

    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    gold_deals_path = os.path.join(gold_dir, 'gold_deals.csv')
    os.makedirs(gold_dir, exist_ok=True)
    
    top_30_deals.to_csv(gold_deals_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Đã kết xuất thành công {len(top_30_deals)} Kèo Thơm siêu chuẩn vào sản phẩm: {gold_deals_path}")

    print("\n[*] Đang tiến hành đóng gói bộ não AI Dual-Core ra file vật lý...")
    import pickle
    
    models_dir = os.path.join(BASE_DIR, 'models_prod')
    os.makedirs(models_dir, exist_ok=True)
    
    # Đóng gói toàn bộ phân hệ A (Xe phổ thông)
    payload_model_A = {
        "preprocessor": preprocessor_A,
        "stacking_core": pipeline_objs_A,
        "msrp_threshold": MSRP_THRESHOLD
    }
    with open(os.path.join(models_dir, 'dual_core_model_A.pkl'), 'wb') as f:
        pickle.dump(payload_model_A, f)
        
    # Đóng gói toàn bộ phân hệ B (Xe hạng sang)
    payload_model_B = {
        "preprocessor": preprocessor_B,
        "stacking_core": pipeline_objs_B,
        "msrp_threshold": MSRP_THRESHOLD
    }
    with open(os.path.join(models_dir, 'dual_core_model_B.pkl'), 'wb') as f:
        pickle.dump(payload_model_B, f)
        
    print(f"[SUCCESS] AI core đã được đóng gói an toàn tại thư mục: {models_dir}")
    print(" -> File 1: dual_core_model_A.pkl (Chuyên trị xe Phổ thông)")
    print(" -> File 2: dual_core_model_B.pkl (Chuyên trị xe Hạng sang)")

if __name__ == "__main__":
    execute_advanced_training_pipeline()