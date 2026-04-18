import os
import json
import time
import base64
from tqdm import tqdm
import requests

# --- CẤU HÌNH ---
MISTRAL_API_KEY = "5sgmNc547G9hP38oSAG0se0YbFOduPQO"
MODEL_ID = "pixtral-12b-2409"
INPUT_FOLDER = "train" 
OUTPUT_FILE = "qwen3_dataset_final.jsonl"

def call_mistral_vision(prompt, img_path):
    url = "https://api.mistral.ai/v1/chat/completions"
    try:
        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ],
            "temperature": 0.2, # Tăng nhẹ để AI "đoán" chữ tốt hơn ở ảnh mờ
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"\n❌ Lỗi API: {e}")
        return None

def clean_value(val):
    """Hàm biến đổi mọi kiểu dữ liệu (Dict, List) về String thuần túy"""
    if val is None or val == "":
        return "N/A"
    if isinstance(val, dict):
        # Nếu là dict, ưu tiên lấy value đầu tiên (thường là nội dung chính)
        return clean_value(next(iter(val.values())))
    if isinstance(val, list):
        # Nếu là list, nối các phần tử lại
        return ", ".join([str(i) for i in val])
    return str(val).strip()

def main():
    json_path = os.path.join(INPUT_FOLDER, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print("❌ Không tìm thấy file _annotations.coco.json")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Đọc danh sách ảnh đã xử lý để chạy tiếp (resume)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for l in f:
                try: processed_ids.add(json.loads(l)['id'])
                except: continue

    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
    queue = [img for img in coco['images'] if f"img_{img['id']}" not in processed_ids]

    print(f"🚀 Mistral API (Pixtral): Đang xử lý {len(queue)} ảnh còn lại...")

    # Prompt ép AI trả về String và tập trung OCR
    prompt = """Extract info from this Vietnamese ID card: id, name, birth, origin, address. 
    Return ONLY a JSON object where values are plain strings. 
    No nested objects. Be accurate with OCR."""

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for img_info in tqdm(queue, desc="Xử lý"):
            img_id = f"img_{img_info['id']}"
            img_path = os.path.join(INPUT_FOLDER, img_info['file_name'])

            if not os.path.exists(img_path): continue

            # Lấy tọa độ từ file COCO
            anns = [a for a in coco['annotations'] if a['image_id'] == img_info['id']]
            coords_map = {cat_map[a['category_id']]: [int(a['bbox'][1]*1000/640), int(a['bbox'][0]*1000/640), 
                          int((a['bbox'][1]+a['bbox'][3])*1000/640), int((a['bbox'][0]+a['bbox'][2])*1000/640)] 
                          for a in anns if cat_map[a['category_id']] not in ['cccd', 'card', 'title']}

            while True:
                res_text = call_mistral_vision(prompt, img_path)
                if res_text:
                    try:
                        info = json.loads(res_text)
                        assistant_msg = ""
                        
                        # Chỉ lấy 5 trường quan trọng nhất
                        for k in ['id', 'name', 'birth', 'origin', 'address']:
                            if k in coords_map:
                                raw_val = info.get(k, "N/A")
                                val = clean_value(raw_val) # Làm sạch dữ liệu
                                assistant_msg += f"{k}: <|box_open|>{coords_map[k]}<|box_close|>{val}\n"

                        res_line = {
                            "id": img_id,
                            "image": os.path.abspath(img_path),
                            "conversations": [
                                {"role": "user", "content": "Trích xuất thông tin CCCD."},
                                {"role": "assistant", "content": assistant_msg.strip()}
                            ]
                        }
                        f_out.write(json.dumps(res_line, ensure_ascii=False) + '\n')
                        f_out.flush()
                        time.sleep(1.2) # Tránh Rate Limit của gói Free
                        break
                    except Exception as e:
                        print(f"\n⚠️ Lỗi parse JSON {img_id}: {e}. Thử lại...")
                        time.sleep(2); continue
                else:
                    print("\n⏳ API không phản hồi, đang chờ...")
                    time.sleep(10)

if __name__ == "__main__":
    main()