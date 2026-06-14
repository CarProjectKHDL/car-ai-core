import os
import json
import pickle
import numpy as np
import pandas as pd

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
    Dynamic Guardrail Layer: Siết chặt trần giá theo khấu hao thời gian thực tế
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
            # Biên độ trần động siết dần từ 125% xuống 75% theo tuổi xe
            dynamic_ratio = max(1.25 - (car_age * 0.05), 0.75)
            max_allowed_price = msrp_cap * dynamic_ratio

            if y_pred_bounded[idx] > max_allowed_price:
                y_pred_bounded[idx] = max_allowed_price
                adjusted_count += 1

        if y_pred_bounded[idx] < 10.0:
            y_pred_bounded[idx] = 10.0

    if adjusted_count > 0:
        print(f"   [Dynamic Guardrail] Đã áp trần động an toàn cho {adjusted_count} dòng xe.")

    return y_pred_bounded

def load_production_model(models_dir, segment_letter):
    file_path = os.path.join(models_dir, f'dual_core_model_{segment_letter}.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def predict_with_stacking(df_input, model_payload):
    preprocessor = model_payload["preprocessor"]
    stacking_core = model_payload["stacking_core"]
    
    fitted_base_models = stacking_core["fitted_base_models"]
    fitted_meta_learner = stacking_core["fitted_meta_learner"]
    model_names = stacking_core["model_names"]
    
    # Kỹ nghệ chuyển đổi qua ma trận ColumnTransformer
    X_encoded = preprocessor.transform(df_input)
    
    # Tạo ma trận Meta-Features từ Tầng 0
    meta_features = np.zeros((X_encoded.shape[0], len(model_names)))
    for idx, name in enumerate(model_names):
        fold_preds = np.column_stack([model.predict(X_encoded) for model in fitted_base_models[name]])
        meta_features[:, idx] = np.mean(fold_preds, axis=1)
        
    # Dự đoán thông qua Meta-Learner Tầng 1
    pred_log = fitted_meta_learner.predict(meta_features)
    return np.expm1(pred_log)

def run_daily_sweet_deals_engine():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    gold_cars_path = os.path.join(BASE_DIR, 'data', 'gold', 'gold_cars_train.csv')
    models_dir = os.path.join(BASE_DIR, 'models_prod')
    
    if not os.path.exists(gold_cars_path):
        print(f"[-] Không tìm thấy tệp dữ liệu Gold tại: {gold_cars_path}")
        return

    df = pd.read_csv(gold_cars_path)
    print(f"[*] Đang khởi chạy Inference Engine trên tệp: {len(df)} dòng...")

    # Nạp từ điển MSRP và tính toán map động msrp_proxy
    master_dict = load_master_msrp_dictionary(BASE_DIR)
    msrp_lookup = {}
    for brand_key, brand_data in master_dict.items():
        trims_data = brand_data.get("trims", {})
        for core_key, core_trims in trims_data.items():
            for trim_name, msrp_val in core_trims.items():
                if msrp_val and isinstance(msrp_val, (int, float)):
                    msrp_lookup[trim_name.lower().strip()] = float(msrp_val)

    brand_median_price = df.groupby('brand')['price_numeric'].median()

    df['msrp_proxy'] = df.apply(
        lambda r: msrp_lookup.get(str(r['model_trim']).lower().strip())
                  or msrp_lookup.get(str(r['model_core']).lower().strip())
                  or float(brand_median_price.get(r['brand'], 500.0)),
        axis=1
    )

    df['high_usage'] = (df['odo_per_year'] > 25000).astype(int)
    df['age_bucket'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 11], labels=['Luot', 'Trung_Han', 'Doi_Sau']).astype(str)
    
    luxury_list = ['mercedes benz', 'bmw', 'audi', 'lexus', 'porsche', 'volvo']
    df['is_luxury_brand'] = df['brand'].str.lower().isin(luxury_list).astype(int)

    df['final_model_feature'] = np.where(
        df['is_dense_model'] == True,
        df['brand'].astype(str) + "_" + df['model_trim'].astype(str),
        df['brand'].astype(str) + "_" + df['model_core'].astype(str)
    )

    try:
        model_payload_A = load_production_model(models_dir, 'A')
        model_payload_B = load_production_model(models_dir, 'B')
    except Exception as e:
        print(f"[-] Thất bại khi nạp file pkl. Chi tiết: {e}")
        return

    # Phân luồng chia để trị
    MSRP_THRESHOLD = 1500.0
    df_A = df[df['msrp_proxy'] <= MSRP_THRESHOLD].copy().reset_index(drop=True)
    df_B = df[df['msrp_proxy'] > MSRP_THRESHOLD].copy().reset_index(drop=True)

    # Chạy dự đoán siêu tốc
    preds_A = predict_with_stacking(df_A, model_payload_A) if not df_A.empty else []
    preds_B = predict_with_stacking(df_B, model_payload_B) if not df_B.empty else []

    # Áp dụng lớp trần phòng vệ động
    if not df_A.empty: 
        df_A['predicted'] = apply_heuristic_guardrail(df_A, preds_A, master_dict)
    if not df_B.empty: 
        df_B['predicted'] = apply_heuristic_guardrail(df_B, preds_B, master_dict)

    # Hợp nhất và bóc tách kèo thơm
    df_total = pd.concat([df_A, df_B], ignore_index=True)
    df_total['predicted_price'] = np.round(df_total['predicted'], 1)
    df_total['price_diff_absolute'] = df_total['price_numeric'] - df_total['predicted_price']
    df_total['price_diff_pct'] = df_total['price_diff_absolute'] / df_total['predicted_price']

    sweet_deals = df_total[
        (df_total['price_diff_pct'] <= -0.08) & (df_total['price_diff_pct'] >= -0.25) &
        (df_total['price_diff_absolute'] <= -15) & (df_total['price_diff_absolute'] >= -120) &
        (df_total['brand'] != 'Unknown') & (df_total['is_verified_accident_free'] == 1)
    ].copy()

    top_30_deals = sweet_deals.sort_values(by='price_diff_pct', ascending=True).head(30)
    if 'model_trim' in top_30_deals.columns:
        top_30_deals = top_30_deals.rename(columns={'model_trim': 'model'})

    gold_deals_path = os.path.join(BASE_DIR, 'data', 'gold', 'gold_deals.csv')
    top_30_deals.to_csv(gold_deals_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Đã cập nhật xong 30 kèo Tthơm chuẩn hóa vào sản phẩm: {gold_deals_path}")

if __name__ == "__main__":
    run_daily_sweet_deals_engine()