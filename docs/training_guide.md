# Hướng Dẫn Training LoRA & DreamBooth

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

## Training LoRA

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

---

## Training DreamBooth

### 1. Chuẩn Bị Dữ Liệu

- **Instance images**: 10-20 ảnh/style (resize 512x512)
- **Captions**: Mỗi ảnh chứa token riêng, ví dụ `a sks style painting`
- **Class images (prior)**: 200 ảnh chung chung (có thể dùng COCO hoặc generate)

Lưu tại:
```
/kaggle/working/dreambooth/{style}/instance_images/
/kaggle/working/dreambooth/{style}/class_images/
```

### 2. Cài Đặt Môi Trường

```python
!pip install accelerate==0.27.2 transformers==4.39.3 diffusers[torch]==0.27.0 bitsandbytes==0.43.0 xformers
```

### 3. Chạy DreamBooth Training Script

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/kaggle/working/dreambooth/sks_style/instance_images"
export CLASS_DIR="/kaggle/working/dreambooth/sks_style/class_images"
export OUTPUT_DIR="/kaggle/working/dreambooth_checkpoints/sks_style"

accelerate launch src/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks style painting" \
  --class_prompt="a painting" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5e-6 \
  --prior_loss_weight=1.0 \
  --mixed_precision="fp16"
```

### 4. Lưu Ý Khi Train DreamBooth

- Sử dụng gradient checkpointing (`--gradient_checkpointing`) để giảm memory
- Nếu thiếu class images, dùng flag `--with_prior_preservation --prior_generation_prompt="a painting"`
- Luôn lưu checkpoint giữa chừng (`--checkpointing_steps`)
- Checkpoint output ~2-3GB, cần tải về máy

### 5. Đánh Giá

- Generate ảnh với prompt: `"a sks style painting of {content}"` (có thể sử dụng img2img)
- So sánh metrics (FID, LPIPS, SSIM) với LoRA để đánh giá

