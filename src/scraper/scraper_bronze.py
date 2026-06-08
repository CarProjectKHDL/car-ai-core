import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
from datetime import datetime

def get_raw_car_details(car_url, headers):
    """Thu thập dữ liệu văn bản thô nguyên bản từ trang chi tiết xe"""
    try:
        response = requests.get(car_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {}
        
        soup = BeautifulSoup(response.content, 'html.parser')
        raw_details = {}
        
        h1_tag = soup.find('h1')
        raw_details['raw_title'] = h1_tag.text.strip() if h1_tag else ""

        mapping = {
            'Năm sản xuất': 'raw_year',
            'Xuất xứ': 'raw_origin',
            'Tình trạng': 'raw_condition',
            'Dòng xe': 'raw_body_type',
            'Số Km đã đi': 'raw_odo',
            'Màu ngoại thất': 'raw_exterior_color',
            'Màu nội thất': 'raw_interior_color',
            'Số cửa': 'raw_doors',
            'Số chỗ ngồi': 'raw_seats',
            'Hộp số': 'raw_transmission',
            'Dẫn động': 'raw_drivetrain',
            'Nhiên liệu': 'raw_fuel',
        }
        for col in mapping.values():
            raw_details[col] = ""

        detail_box = soup.find('div', class_='box_car_detail')
        if detail_box:
            rows = detail_box.find_all('div', class_=lambda x: x and ('row' in x or 'line' in x))
            for row in rows:
                label_tag = row.find(['label', 'span'], class_=lambda x: x and 'label' in x) or row.find('label')
                value_tag = row.find('span', class_=lambda x: x and 'txt' in x) or row.find('span')
                
                if label_tag and value_tag:
                    lbl = label_tag.text.strip().replace(':', '')
                    if lbl in mapping:
                        raw_details[mapping[lbl]] = value_tag.text.strip()

        addr_tag = soup.find('span', class_='address') or soup.find('div', class_='car_location') or soup.find('p', class_='address')
        raw_details['raw_location'] = addr_tag.text.strip() if addr_tag else ""

        desc_div = soup.find('div', class_='des_txt')
        raw_details['raw_description'] = " ".join(desc_div.text.split()) if desc_div else ""

        return raw_details
    except Exception as e:
        print(f"   [Lỗi] Không thể lấy chi tiết xe {car_url}: {e}")
        return {}

def pipeline_bronze_cold_start(current_page):
    """
    Nhận vào chính xác số trang cần cào
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'vi-VN,vi;q=0.9',
        'Referer': 'https://bonbanh.com/'
    }
    
    bronze_file_path = os.path.join('data', 'bronze', 'bronze_cars_raw.csv')
    
    os.makedirs(os.path.dirname(bronze_file_path), exist_ok=True)
    
    existing_ids = set()
    if os.path.exists(bronze_file_path):
        try:
            old_df = pd.read_csv(bronze_file_path, usecols=['id'])
            existing_ids = set(old_df['id'].astype(str).tolist())
        except Exception:
            pass

    all_batch_data = []
    url = f"https://bonbanh.com/oto/page,{current_page}"
    print(f"\n=== [BRONZE INGEST] Đang quét danh sách trang: {current_page} ===")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            car_items = soup.find_all('li', class_=lambda x: x and 'car-item' in x)
            if not car_items:
                car_items = soup.find_all('div', class_=lambda x: x and 'car-item' in x)

            print(f"[+] Tìm thấy {len(car_items)} tin đăng tại trang {current_page}.")
            
            for item in car_items:
                a_tag = item.find('a', href=True)
                if not a_tag: continue
                    
                car_url = "https://bonbanh.com/" + a_tag['href']
                try:
                    car_id = car_url.split('-')[-1].replace('.html', '').strip()
                except Exception:
                    continue
                    
                if car_id in existing_ids:
                    print(f"   [-] Skip ID {car_id}: Đã tồn tại trong kho lưu trữ Bronze.")
                    continue
                    
                print(f"   [+] Cào mới ID {car_id} -> {car_url}")
                raw_details = get_raw_car_details(car_url, headers)
                
                if raw_details:
                    base_info = {
                        'id': car_id,
                        'url': car_url,
                        '_ingested_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '_source_system': 'bonbanh_scraper_v1'
                    }
                    base_info.update(raw_details)
                    all_batch_data.append(base_info)
                    existing_ids.add(car_id)
                    
                time.sleep(random.uniform(0.6, 1.4))
    except Exception as e:
        print(f"[Lỗi] Kết nối trang danh sách {current_page} thất bại: {e}")
        
    if all_batch_data:
        new_df = pd.DataFrame(all_batch_data)
        if not os.path.exists(bronze_file_path):
            new_df.to_csv(bronze_file_path, index=False, encoding='utf-8-sig')
        else:
            new_df.to_csv(bronze_file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"[Thành công] Đã lưu xong dữ liệu trang {current_page}.")