# Hướng Dẫn Training LoRA

## Chuẩn Bị Dữ Liệu

### 1. Download Datasets

**COCO 2017**:
```bash
# Download từ https://cocodataset.org/#download
# Chỉ cần train images (118k images)
```

**WikiArt**:
```bash
# Download từ https://www.wikiart.org/
# Chọn 3-5 phong cách, mỗi phong cách 50-100 ảnh
```

### 2. Organize Data

```
data/
├── coco/
│   └── train/
│       └── *.jpg
└── wikiart/
    ├── monet/
    │   └── *.jpg
    ├── ukiyo-e/
    │   └── *.jpg
    └── pop-art/
        └── *.jpg
```

### 3. Create Pairs (Optional)

Có thể tạo file CSV với content-style pairs hoặc random pairing trong training.

## Training

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Configure

Chỉnh sửa `src/configs/lora_config.yaml`:
- `content_dir`: Đường dẫn đến COCO images
- `style_dir`: Đường dẫn đến WikiArt images
- `lora_rank`: Rank của LoRA (default: 4)
- `learning_rate`: Learning rate (default: 1e-4)
- `batch_size`: Batch size (default: 2)

### 3. Train

```bash
python src/train_lora.py \
    --config src/configs/lora_config.yaml \
    --style_name monet
```

### 4. Monitor

Checkpoints được lưu tại `results/lora_checkpoints/{style_name}/`

## Tips

- Bắt đầu với rank=4, sau đó tăng lên 8 hoặc 16 nếu cần chất lượng tốt hơn
- Learning rate 1e-4 thường tốt, có thể thử 5e-5 hoặc 2e-4
- Training 5k-8k steps thường đủ cho mỗi phong cách
- Sử dụng mixed precision (fp16) để tiết kiệm memory

