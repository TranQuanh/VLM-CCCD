import os

# --- CẤU HÌNH SIÊU TINH GỌN CHO SWIFT V5+ ---
MODEL_ID = "qwen/Qwen3-VL-8B-Instruct" 
TRAIN_DATASET = "cccd_train.jsonl"
VAL_DATASET = "cccd_val.jsonl"
OUTPUT_DIR = "output/qwen3_8b_cccd_3050"

# CHỈ GIỮ LẠI CÁC THAM SỐ CỐT LÕI MÀ SWIFT KHÔNG THỂ TỪ CHỐI
train_cmd = f"""
swift sft \
    --model {MODEL_ID} \
    --dataset {TRAIN_DATASET} \
    --val_dataset {VAL_DATASET} \
    --output_dir {OUTPUT_DIR} \
    --quant_method bnb \
    --quant_bits 4 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_steps 100 \
    --eval_steps 50 \
    --save_steps 50 \
    --max_length 512 \
    --max_pixels 153600 \
    --gradient_checkpointing true \
    --optimizer adamw_bnb_8bit \
    --dataloader_num_workers 2
"""

if __name__ == "__main__":
    print("🚀 Đợt tổng tấn công cuối cùng: Loại bỏ sft_type và dtype...")
    os.system(train_cmd)