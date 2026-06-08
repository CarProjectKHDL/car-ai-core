import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_market_eda():
    silver_path = os.path.join('data', 'silver', 'silver_cars_clean.csv')
    if not os.path.exists(silver_path):
        print("Chưa có file Silver")
        return

    df = pd.read_csv(silver_path)
    print(f"Đang phân tích EDA trên {len(df)} xe...")

    os.makedirs(os.path.join('data', 'gold'), exist_ok=True)

    # Biểu đồ 1: Ma trận tương quan (Correlation Matrix)
    plt.figure(figsize=(8, 6))
    corr_cols = ['price_numeric', 'year', 'car_age', 'odo_numeric', 'is_first_owner', 'has_upgrades']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Ma Trận Tương Quan Giữa Các Thuộc Tính Số Học')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'gold', 'correlation_matrix.png'))
    plt.close()
    print("Đã lưu")

    # Biểu đồ 2: Đường cong khấu hao theo Hãng (Depreciation Curve)
    # Lọc ra top 5 hãng xe phổ biến nhất tại VN
    top_brands = df['brand'].value_counts().head(5).index
    df_top_brands = df[df['brand'].isin(top_brands)]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_top_brands, x='car_age', y='price_numeric', hue='brand', marker='o', errorbar=None)
    plt.title('Đường Cong Khấu Hao Giá Xe Theo Tuổi Đời (Top 5 Hãng Phổ Biến)')
    plt.xlabel('Tuổi đời xe (Năm)')
    plt.ylabel('Giá xe trung bình (Triệu đồng)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'gold', 'depreciation_curve.png'))
    plt.close()
    print("Đã lưu")

if __name__ == "__main__":
    run_market_eda()