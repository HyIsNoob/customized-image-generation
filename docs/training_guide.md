# Hướng Dẫn Training LoRA

## Chuẩn Bị Dữ Liệu

### 1. Add Datasets trên Kaggle

**COCO 2017**:
- Vào https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
- Click "New Notebook" hoặc "Add to notebook"
- Dataset path: `/kaggle/input/coco-2017-dataset/coco2017/train2017/`

**WikiArt**:
- Vào https://www.kaggle.com/datasets/steubk/wikiart
- Click "Add to notebook"
- Dataset path: `/kaggle/input/wikiart/wikiart/`

### 2. Cấu Hình Đường Dẫn

Trên Kaggle, datasets đã có sẵn tại:
```
/kaggle/input/coco-2017-dataset/coco2017/train2017/  # Content images
/kaggle/input/wikiart/wikiart/                        # Style images (cần kiểm tra cấu trúc thư mục)
/kaggle/working/lora_checkpoints/                     # Output checkpoints
```

### 3. Create Pairs (Optional)

Có thể tạo file CSV với content-style pairs hoặc random pairing trong training.

## Training

### 1. Setup Environment trên Kaggle

```python
# Clone repository
!git clone https://github.com/HyIsNoob/customized-image-generation.git
%cd customized-image-generation

# Cài đặt dependencies
!pip install -r requirements.txt

# Bật Internet và GPU trong Settings
```

### 2. Configure

Chỉnh sửa `src/configs/lora_config.yaml` hoặc override trong notebook:
```python
config = {
    'content_dir': '/kaggle/input/coco-2017-dataset/coco2017/train2017',
    'style_dir': '/kaggle/input/wikiart/wikiart',
    'output_dir': '/kaggle/working/lora_checkpoints',
    'lora_rank': 4,
    'learning_rate': 1e-4,
    'batch_size': 2,
}
```

### 3. Train

```bash
python src/train_lora.py \
    --config src/configs/lora_config.yaml \
    --style_name monet
```

### 4. Monitor

Checkpoints được lưu tại `/kaggle/working/lora_checkpoints/{style_name}/`

**Lưu ý**: 
- Download checkpoints về máy hoặc upload lên Google Drive
- Kaggle output có giới hạn, nên download thường xuyên

## Tips

- Bắt đầu với rank=4, sau đó tăng lên 8 hoặc 16 nếu cần chất lượng tốt hơn
- Learning rate 1e-4 thường tốt, có thể thử 5e-5 hoặc 2e-4
- Training 5k-8k steps thường đủ cho mỗi phong cách
- Sử dụng mixed precision (fp16) để tiết kiệm memory

