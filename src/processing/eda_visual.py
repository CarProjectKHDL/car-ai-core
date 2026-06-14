import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def run_market_eda():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) if 'src' in os.path.abspath(os.path.dirname(__file__)) else os.path.abspath(os.path.dirname(__file__))
    gold_dir = os.path.join(BASE_DIR, 'data', 'gold')
    gold_path = os.path.join(gold_dir, 'gold_cars_train.csv')
    reports_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    os.makedirs(reports_dir, exist_ok=True)

    print("[*] Đang nạp tầng Gold để xuất bản kho biểu đồ nghiệp vụ nâng cao...")
    if not os.path.exists(gold_path):
        print(f"[-] Chưa tìm thấy file Gold tại: {gold_path}.")
        return

    gold_df = pd.read_csv(gold_path)

    # Động tính các trường nghiệp vụ mới phục vụ đồ thị trực quan
    gold_df['high_usage'] = (gold_df['odo_per_year'] > 25000).map({True: 'Xe Chạy Dịch Vụ (>25k km/năm)', False: 'Xe Gia Đình (<25k km/năm)'})

    # CHART 1: MA TRẬN TƯƠNG QUAN SẠCH BÓNG NHIỄU BIẾN CŨ
    plt.figure(figsize=(10, 8))
    corr_cols = ['price_numeric', 'year', 'car_age', 'odo_numeric', 'odo_per_year', 
                 'is_first_owner', 'has_upgrades', 'is_verified_accident_free', 'is_brand_new']
    actual_corr_cols = [c for c in corr_cols if c in gold_df.columns]
    
    sns.heatmap(gold_df[actual_corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    plt.title('MA TRẬN TƯƠNG QUAN TOÁN HỌC CÁC BIẾN ĐẦU VÀO AI', fontsize=12, fontweight='bold')
    plt.tight_layout()
    chart1_path = os.path.join(reports_dir, 'correlation_matrix.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # CHART 2: ĐƯỜNG CONG KHẤU HAO CÁC THƯƠNG HIỆU XE SANG CỐT LÕI
    active_luxury_brands = ['Mercedes Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Volvo']
    df_lux = gold_df[gold_df['brand'].isin(active_luxury_brands)]
    if not df_lux.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_lux, x='car_age', y='price_numeric', hue='brand', marker='o', errorbar=None, linewidth=2.5)
        plt.title('ĐƯỜNG CONG KHẤU HAO PHI TUYẾN XE HẠNG SANG', fontsize=12, fontweight='bold')
        plt.xlabel('Tuổi xe (Năm)')
        plt.ylabel('Giá trị (Triệu VNĐ)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        chart2_path = os.path.join(reports_dir, 'luxury_depreciation_curve.png')
        plt.savefig(chart2_path, dpi=300)
        plt.close()

    # CHART 3: PHÂN PHỐI SẢN LƯỢNG GIÁ TOÀN SÀN VÀ TƯ DUY PHÂN HOẠCH
    plt.figure(figsize=(12, 6))
    sns.histplot(data=gold_df, x='price_numeric', kde=True, bins=50, color='darkblue', alpha=0.6)
    plt.axvline(x=1500.0, color='red', linestyle='--', linewidth=2, label='Ranh giới Phân Luồng Toán Học (1.5 Tỷ VNĐ)')
    plt.text(200, plt.gca().get_ylim()[1]*0.75, 'MODEL A\n(Xe Phổ Thông)', color='blue', fontsize=10, fontweight='bold')
    plt.text(1700, plt.gca().get_ylim()[1]*0.75, 'MODEL B\n(Xe Hạng Sang)', color='red', fontsize=10, fontweight='bold')
    plt.title('PHÂN PHỐI SẢN LƯỢNG GIÁ TOÀN SÀN VÀ TƯ DUY PHÂN HOẠCH', fontsize=12, fontweight='bold')
    plt.xlabel('Giá xe thực tế (Triệu VNĐ)')
    plt.ylabel('Sản lượng (Số lượng tin)')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart3_path = os.path.join(reports_dir, 'dual_core_routing_justification.png')
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    # CHART 4: KHẤU HAO THEO MỤC ĐÍCH SỬ DỤNG (HIGH USAGE BUCKET)
    # Khảo sát trên 4 hãng xe phổ dụng nhất Việt Nam
    common_brands = ['Toyota', 'Hyundai', 'Kia', 'Honda']
    df_common = gold_df[gold_df['brand'].isin(common_brands)]
    
    if not df_common.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_common, x='brand', y='price_numeric', hue='high_usage', palette='Set2')
        plt.title('PHÂN TÍCH ĐỘ LỆCH GIÁ: XE GIA ĐÌNH VS XE CHẠY DỊCH VỤ CÀY ODO', fontsize=12, fontweight='bold')
        plt.xlabel('Thương hiệu xe')
        plt.ylabel('Dải giá giao dịch (Triệu VNĐ)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        chart4_path = os.path.join(reports_dir, 'high_usage_impact_analysis.png')
        plt.savefig(chart4_path, dpi=300)
        plt.close()
        print(f"   [+] Đã xuất thêm biểu đồ 4 (Nghiệp vụ dịch vụ): {chart4_path}")

    print("[SUCCESS] Toàn bộ kho biểu đồ EDA báo cáo đã sẵn sàng tại reports/figures/!")

if __name__ == "__main__":
    run_market_eda()