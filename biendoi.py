import os
import json
import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

# Cấu hình logging để tắt các thông báo không cần thiết
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# --- CẤU HÌNH ---
INPUT_FOLDER = "train" 
OUTPUT_FILE = "qwen3_dataset_raw.jsonl" # File thô trước khi qua Gemini

# Khởi tạo OCR (Sử dụng các tham số ổn định nhất)
ocr = PaddleOCR(lang='vi', use_textline_orientation=True)

def coco_to_qwen_vl_coords(bbox, img_w=640, img_h=640):
    """Chuyển tọa độ COCO [x,y,w,h] sang chuẩn Qwen-VL [ymin, xmin, ymax, xmax] thang 1000"""
    x, y, w, h = bbox
    xmin = int(max(0, x) * 1000 / img_w)
    ymin = int(max(0, y) * 1000 / img_h)
    xmax = int(min(img_w, x + w) * 1000 / img_w)
    ymax = int(min(img_h, y + h) * 1000 / img_h)
    return [ymin, xmin, ymax, xmax]

def main():
    json_path = os.path.join(INPUT_FOLDER, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file JSON tại: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Đọc danh sách đã xử lý để tránh làm lại (Resume)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for l in f:
                try: processed_ids.add(json.loads(l)['id'])
                except: continue

    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
    queue = [img for img in coco['images'] if f"img_{img['id']}" not in processed_ids]

    print(f"🚀 Chế độ tạo Dataset Thô: Đang xử lý {len(queue)} ảnh...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8', buffering=1) as f_out:
        for img_info in tqdm(queue, desc="Xử lý"):
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_path = os.path.join(INPUT_FOLDER, file_name)

            if not os.path.exists(img_path):
                continue

            # Đọc ảnh an toàn (hỗ trợ đường dẫn Windows)
            img_array = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                continue

            raw_texts = {}
            coords_map = {}
            anns = [a for a in coco['annotations'] if a['image_id'] == img_id]

            # Nếu ảnh không có nhãn thì bỏ qua
            if not anns:
                continue

            for ann in anns:
                label = cat_map[ann['category_id']]
                # Bỏ qua các nhãn bao quát
                if label in ['cccd', 'card', 'title']:
                    continue
                
                # Tính toán tọa độ
                coords_map[label] = coco_to_qwen_vl_coords(ann['bbox'])
                
                # Thực hiện OCR thô cho vùng đó
                x, y, w, h = map(int, ann['bbox'])
                crop = image[max(0, y):y+h, max(0, x):x+w]
                
                text_val = ""
                if crop.size > 0:
                    try:
                        # Tiền xử lý nhanh: phóng to 2 lần
                        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        res = ocr.ocr(crop, cls=True)
                        if res and res[0]:
                            text_val = " ".join([line[1][0] for line in res[0]])
                    except:
                        pass
                raw_texts[label] = text_val if text_val.strip() else "..."

            # Tạo nội dung assistant (Tọa độ + OCR thô)
            assistant_msg = ""
            for k in ['id', 'name', 'birth', 'origin', 'address']:
                if k in coords_map:
                    val = raw_texts.get(k, "...")
                    assistant_msg += f"{k}: <|box_open|>{coords_map[k]}<|box_close|>{val}\n"

            if assistant_msg:
                res_line = {
                    "id": f"img_{img_id}",
                    "image": os.path.abspath(img_path),
                    "conversations": [
                        {"role": "user", "content": "Trích xuất thông tin CCCD."},
                        {"role": "assistant", "content": assistant_msg.strip()}
                    ]
                }
                f_out.write(json.dumps(res_line, ensure_ascii=False) + '\n')

    print(f"✅ Đã xong! File thô lưu tại: {OUTPUT_FILE}")
    print("Bây giờ bạn có thể dùng file này để train thử hoặc chạy script Gemini để làm sạch chữ sau.")

if __name__ == "__main__":
    main()