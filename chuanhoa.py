import json
import random
import os

# --- CẤU HÌNH ---
INPUT_FILE = "qwen3_dataset_final.jsonl" # File kết quả từ bước trước
TRAIN_FILE = "cccd_train.jsonl"
VAL_FILE = "cccd_val.jsonl"
TRAIN_RATIO = 0.8 # 90% cho training

def is_valid_data(content):
    """Kiểm tra xem dữ liệu có bị N/A ở các trường quan trọng không"""
    # Nếu có quá 2 trường bị N/A thì loại bỏ để đảm bảo chất lượng dataset
    na_count = content.count("N/A")
    if na_count > 2:
        return False
    return True

def convert_to_qwen_format(line):
    """Chuyển đổi sang định dạng hội thoại chuẩn của Qwen-VL"""
    data = json.loads(line)
    
    # Lấy nội dung phản hồi của AI (assistant)
    assistant_text = data['conversations'][1]['content']
    
    # Cấu trúc lại theo format "messages" mà các thư viện như SWIFT hay LLaMA-Factory yêu cầu
    new_format = {
        "id": data["id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": "Trích xuất thông tin CCCD."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
    }
    return new_format

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy file {INPUT_FILE}. Hãy chạy script lấy data trước nhé!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 Tổng số mẫu ban đầu: {len(lines)}")

    # 1. Lọc dữ liệu sạch
    clean_data = []
    for line in lines:
        try:
            data = json.loads(line)
            assistant_content = data['conversations'][1]['content']
            if is_valid_data(assistant_content):
                clean_data.append(convert_to_qwen_format(line))
        except:
            continue

    print(f"🧹 Số mẫu sau khi lọc sạch (bỏ N/A): {len(clean_data)}")

    # 2. Trộn ngẫu nhiên
    random.seed(42) # Giữ kết quả trộn cố định
    random.shuffle(clean_data)

    # 3. Chia tập Train/Val
    split_idx = int(len(clean_data) * TRAIN_RATIO)
    train_set = clean_data[:split_idx]
    val_set = clean_data[split_idx:]

    # 4. Lưu file
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Đã tách xong!")
    print(f"   - File Train: {TRAIN_FILE} ({len(train_set)} mẫu)")
    print(f"   - File Val: {VAL_FILE} ({len(val_set)} mẫu)")

if __name__ == "__main__":
    main()