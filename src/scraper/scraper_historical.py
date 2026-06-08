import os
import time
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.scraper.scraper_bronze import pipeline_bronze_cold_start

if __name__ == "__main__":
    print("================================================================")
    print("CÀO FULL ĐA LUỒNG THỦ CÔNG")
    print("================================================================")
    
    # Nhận tham số START và END trực tiếp
    # python src/scraper/scraper_historical.py <trang_bat_dau> <trang_ket_thuc>
    if len(sys.argv) == 3:
        START_PAGE = int(sys.argv[1])
        END_PAGE = int(sys.argv[2])
    else:
        # Mặc định nếu không truyền tham số
        START_PAGE = 1
        END_PAGE = 50
        
    print(f"[*] Tiến trình song song đang xử lý vùng trang: {START_PAGE} -> {END_PAGE}")
    
    for page in range(START_PAGE, END_PAGE + 1):
        try:
            pipeline_bronze_cold_start(current_page=page)
            
            time.sleep(random.uniform(1.5, 2.5))
        except KeyboardInterrupt:
            print(f"\n[Dừng] Đã dừng tiến trình tại trang {page}.")
            break
        except Exception as e:
            print(f"Gặp lỗi tại trang {page}: {e}")
            time.sleep(5)
            continue