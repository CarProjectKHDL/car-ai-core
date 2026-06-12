import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def run_market_eda():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    gold_path = os.path.join(gold_dir, 'gold_cars_train.csv')

    print("[*] Đang nạp tầng Gold để kết xuất biểu đồ phân tích thị trường...")
    if not os.path.exists(gold_path):
        print(f"[-] Chưa tìm thấy file Gold tại: {gold_path}. Vui lòng chạy process_gold.py trước!")
        return

    gold_df = pd.read_csv(gold_path)

    # Chart 1: Ma trận tương quan toán học (Bỏ location)
    plt.figure(figsize=(10, 8))
    corr_cols = ['price_numeric', 'year', 'car_age', 'odo_numeric', 'is_first_owner', 'has_upgrades', 'is_verified_accident_free', 'is_dense_model']
    sns.heatmap(gold_df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('MA TRẬN TƯƠNG QUAN TOÁN HỌC TẦNG GOLD (MÔ HÌNH MỚI)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(gold_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()

    # Chart 2: Đường cong khấu hao phân khúc hạng sang
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Rolls-Royce', 'Bentley']
    df_lux = gold_df[gold_df['brand'].isin(luxury_brands)]
    if not df_lux.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_lux, x='car_age', y='price_numeric', hue='brand', marker='o', errorbar=None, linewidth=2.5)
        plt.title('ĐƯỜNG CONG KHẤU HAO XE HẠNG SANG THEO TUỔI ĐỜI THỰC TẾ', fontsize=12, fontweight='bold')
        plt.xlabel('Tuổi đời xe (Năm)')
        plt.ylabel('Giá trị trung bình (Triệu VNĐ)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Hãng xe')
        plt.tight_layout()
        plt.savefig(os.path.join(gold_dir, 'luxury_depreciation_curve.png'), dpi=300)
        plt.close()

    print("[SUCCESS] Toàn bộ biểu đồ EDA phân tích thị trường được cập nhật sạch sẽ tại thư mục Gold.")

if __name__ == "__main__":
    run_market_eda()