import os
import json
import re
from bs4 import BeautifulSoup

def build_hybrid_offline_dictionary():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bonbanh_html_path = os.path.join(base_dir, "bonbanh.html")
    vcar_html_path = os.path.join(base_dir, "vcar.html")
    output_path = os.path.join(base_dir, "data", "silver", "brand_model_dictionary.json")
    
    if not os.path.exists(bonbanh_html_path):
        print(f"[LỖI] Không tìm thấy file '{bonbanh_html_path}'.")
        return
    if not os.path.exists(vcar_html_path):
        print(f"[LỖI] Không tìm thấy file '{vcar_html_path}'.")
        return

    # ----------------------------------------------------------------------
    # PHA 1: TRÍCH XUẤT HÃNG & CORE MODEL TỪ BONBANH OFFLINE
    # ----------------------------------------------------------------------
    print("[*] Pha 1: Đang bóc tách Dòng xe cốt lõi (Core Models) từ bonbanh.html...")
    bonbanh_cores = {}
    
    with open(bonbanh_html_path, "r", encoding="utf-8") as f:
        soup_bb = BeautifulSoup(f.read(), "html.parser")
        
    nav_ul = soup_bb.find("ul", id="primary-nav")
    if nav_ul:
        menu_items = nav_ul.find_all("li", class_="menuparent")
        for item in menu_items:
            brand_tag = item.find("a", class_="mtop-item")
            if brand_tag:
                brand_name = brand_tag.text.strip()
                brand_key = brand_name.lower()
                
                model_tags = item.find_all("a", class_=lambda x: x and "bbl" in x)
                cores = []
                for m_tag in model_tags:
                    c_name = m_tag.text.strip()
                    if c_name and c_name != brand_name and c_name not in cores:
                        cores.append(c_name)
                        
                if brand_name:
                    bonbanh_cores[brand_key] = {
                        "official_brand": brand_name,
                        "core_models": cores
                    }
        print(f"   [+] Thu hoạch thành công {len(bonbanh_cores)} hãng xe Core từ file Bonbanh offline.")
    else:
        print("   [-] Không tìm thấy khối menu ul#primary-nav trong file bonbanh.html.")
        return

    # ----------------------------------------------------------------------
    # PHA 2: TRÍCH XUẤT BIẾN THỂ + GIÁ XE MỚI (MSRP) TỪ VCAR
    # ----------------------------------------------------------------------
    print("\n[*] Pha 2: Đang đọc file vcar.html để xây dựng cây phân cấp + giá trần...")
    
    with open(vcar_html_path, "r", encoding="utf-8") as f:
        soup_vc = BeautifulSoup(f.read(), "html.parser")
        
    rows = soup_vc.find_all("tr", class_=lambda x: x and "banggiaxe-item" in x)
    if not rows:
        rows = soup_vc.find_all("tr", attrs={"data-dongxe-name": True})
        
    print(f"   [+] Tìm thấy {len(rows)} biến thể xe trong file V-Car HTML.")
    
    hybrid_dict = {}
    
    for row in rows:
        tds = row.find_all("td")
        if len(tds) >= 6:
            b_name = tds[0].text.strip()
            m_raw = tds[1].text.strip()
            v_name = tds[2].text.strip()
            price_new_text = tds[5].text.strip()
            
            msrp = 0
            if "tỷ" in price_new_text.lower():
                matches = re.findall(r'\d+\,?\d*', price_new_text)
                if matches:
                    msrp = float(matches[0].replace(',', '.')) * 1000
            elif "triệu" in price_new_text.lower() or price_new_text.replace('.', '').replace(',', '').isdigit():
                matches = re.findall(r'\d+\,?\d*', price_new_text)
                if matches:
                    msrp = float(matches[0].replace(',', '.'))
            
            m_clean = re.sub(r'\d{4}', '', m_raw).strip()
            b_key = b_name.lower()
            
            if "mercedes" in b_key or "benz" in b_key:
                b_key = "mercedes"
                b_name = "Mercedes-Benz"
            elif "bmw" in b_key:
                b_key = "bmw"
                b_name = "BMW"
            
            if b_key not in hybrid_dict:
                bb_info = bonbanh_cores.get(b_key, {"official_brand": b_name, "core_models": []})
                hybrid_dict[b_key] = {
                    "official_brand": bb_info["official_brand"],
                    "core_models": bb_info["core_models"],
                    "trims": {}
                }
            
            m_key = m_clean.lower()
            if m_key not in hybrid_dict[b_key]["trims"]:
                hybrid_dict[b_key]["trims"][m_key] = {}
                
            if m_clean not in hybrid_dict[b_key]["core_models"]:
                hybrid_dict[b_key]["core_models"].append(m_clean)
                
            full_trim_name = f"{m_clean} {v_name}".strip()
            hybrid_dict[b_key]["trims"][m_key][full_trim_name.lower()] = msrp

    for b_key, b_val in bonbanh_cores.items():
        if b_key not in hybrid_dict:
            hybrid_dict[b_key] = {
                "official_brand": b_val["official_brand"],
                "core_models": b_val["core_models"],
                "trims": {m.lower(): {m.lower(): 0} for m in b_val["core_models"]}
            }
        else:
            for core in b_val["core_models"]:
                if core not in hybrid_dict[b_key]["core_models"]:
                    hybrid_dict[b_key]["core_models"].append(core)
                core_lower = core.lower()
                if core_lower not in hybrid_dict[b_key]["trims"]:
                    hybrid_dict[b_key]["trims"][core_lower] = {core_lower: 0}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_dict, f, ensure_ascii=False, indent=4)
        
    print(f"\n[HYBRID DICTIONARY + MSRP CREATED] Thành công!")
    print(f"[+] Đã lưu tại: {output_path}")
    print(f"[+] Tổng số hãng xe: {len(hybrid_dict)}")

if __name__ == "__main__":
    build_hybrid_offline_dictionary()