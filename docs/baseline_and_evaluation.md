# Baseline và Chiến Lược Đánh Giá

## 1. BASELINE - Phương Pháp So Sánh

### 1.1. Baseline Chính: Stable Diffusion v1.5 Gốc

**Mô hình baseline**: `runwayml/stable-diffusion-v1-5` (không fine-tune)

**Cách sử dụng**:
- Download model từ Hugging Face (đã được train sẵn, không cần train từ đầu)
- Sử dụng text prompt để generate ảnh
- Không có style transfer cụ thể, chỉ dựa vào prompt

**Ví dụ prompt**:
```
"a painting in the style of Monet, a landscape with water lilies"
```

**Mục đích so sánh**:
- Xem LoRA fine-tuning có cải thiện chất lượng style transfer không
- So sánh FID, LPIPS, SSIM giữa baseline và LoRA models

### 1.2. Baseline Phụ: Phương Pháp Style Transfer Truyền Thống

**Các phương pháp**:
- AdaIN (Adaptive Instance Normalization)
- SANet (Style-Aware Normalization)

**Mục đích**: So sánh với các phương pháp style transfer cũ để chứng minh ưu điểm của SD + LoRA

### 1.3. Fine-tuning Baseline 1: LoRA

- Train ~4-8M parameters
- Checkpoint 4-8MB/style
- Training 2-3 giờ/style (Kaggle T4/P100)
- Ưu tiên sử dụng khi cần mở rộng nhanh nhiều phong cách

### 1.4. Fine-tuning Baseline 2: DreamBooth

- Train một phần UNet (và tùy chọn text encoder) với prior preservation
- Checkpoint 2-3GB/style
- Training 4-6 giờ/style (Kaggle T4/P100)
- Chất lượng cao, phù hợp làm đối chứng với LoRA

| So sánh nhanh | LoRA | DreamBooth |
|---------------|------|------------|
| Params train | ~4-8M | ~100-200M |
| Checkpoint | 4-8MB | 2-3GB |
| Training time | 2-3 giờ | 4-6 giờ |
| GPU memory | 12-16GB | 16-20GB |
| Điểm mạnh | Nhanh, nhẹ, dễ quản lý | Giữ style rất tốt, ít ảnh |

---

## 2. MODEL - Nguồn Gốc và Cách Train

### 2.1. Base Model: Stable Diffusion v1.5

**Nguồn gốc**:
- Model đã được train sẵn bởi RunwayML
- Download từ Hugging Face: `runwayml/stable-diffusion-v1-5`
- **KHÔNG train từ đầu**, chỉ download và sử dụng

**Cấu trúc model**:
- VAE Encoder/Decoder: ~85M params (giữ nguyên, không train)
- UNet: ~860M params (chỉ train một phần nhỏ với LoRA)
- CLIP Text Encoder: ~123M params (không sử dụng trong project này)

**Code download model**:
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
```

### 2.2. LoRA Fine-Tuning

**Cách train**:
1. Load base model SD v1.5 (đã download)
2. Thêm LoRA layers vào UNet attention layers
3. **CHỈ train LoRA weights** (~4-8M params), không train toàn bộ UNet
4. Train trên style images từ WikiArt với trigger words (ví dụ: "contemporary_realism style")

**Pipeline Training**:
- Input: Content image (COCO) + Style image (WikiArt) + Trigger word
- Model học cách generate ảnh có style tương ứng với trigger word
- Sau khi train xong, mỗi style có 1 LoRA checkpoint riêng

**Pipeline training**:

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# 1. Load base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 2. Setup LoRA config
lora_config = LoraConfig(
    r=4,  # rank
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # attention layers
    alpha=32
)

# 3. Apply LoRA to UNet
pipe.unet = get_peft_model(pipe.unet, lora_config)

# 4. Train chỉ LoRA weights
# Freeze tất cả weights khác, chỉ train LoRA
for name, param in pipe.unet.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# 5. Training loop với style images
# Loss: L2 + LPIPS + StyleLoss
```

**Hyperparameters**:
- Rank (r): 4 (số lượng tham số LoRA)
- Learning rate: 1e-4
- Batch size: 2-4
- Steps: 5,000-8,000 per style
- Optimizer: AdamW
- Scheduler: Cosine

**Kết quả**:
- Mỗi style → 1 LoRA checkpoint (~4-8MB)
- Có thể load/unload LoRA để switch giữa các styles

### 2.3. Inference Pipeline - Cách Sử Dụng Sau Khi Train

**Input**: 1 ảnh content + Chọn style class

**Output**: Ảnh đã được style transfer

**Pipeline chi tiết**:

**Cách 1: Image-to-Image (Khuyến nghị)**
```python
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

# 1. Load base model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 2. Load LoRA checkpoint cho style đã chọn
pipe.unet = PeftModel.from_pretrained(pipe.unet, "path/to/lora_checkpoint")

# 3. Input: Content image
content_image = load_image("content.jpg")

# 4. Generate với trigger word
prompt = "contemporary_realism style"  # Trigger word tương ứng với style
output = pipe(
    prompt=prompt,
    image=content_image,
    strength=0.7,  # Độ mạnh của style transfer (0-1)
    num_inference_steps=50
).images[0]
```

**Cách 2: Text-to-Image với LoRA**
```python
# 1. Load base model + LoRA
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet = PeftModel.from_pretrained(pipe.unet, "path/to/lora_checkpoint")

# 2. Generate với prompt đơn giản
prompt = "a landscape in contemporary_realism style"
output = pipe(prompt=prompt).images[0]
```

**Tóm tắt**:
1. **Tải SD model về**: Model text-to-image gốc (không train từ đầu)
2. **Thêm LoRA**: Fine-tune LoRA để học style từ WikiArt
3. **Inference**: Input 1 ảnh + chọn style → Load LoRA tương ứng → Generate → Output ảnh đã đổi style

**Lưu ý**:
- Mỗi style cần 1 LoRA checkpoint riêng
- Có thể switch giữa các styles bằng cách load/unload LoRA
- Trigger word (ví dụ: "contemporary_realism") kích hoạt style tương ứng

---

## 3. EVALUATION - Đánh Giá Chất Lượng

### 3.1. Metrics Sử Dụng

#### 3.1.1. FID (Fréchet Inception Distance)

**Mục đích**: Đo độ "thật" và chất lượng của ảnh sinh ra

**Nguyên lý**:
- Extract features từ Inception v3 model
- So sánh phân phối features của ảnh generated vs ảnh thật (style images)
- FID thấp = ảnh generated giống ảnh thật hơn

**Công thức**:
```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^(1/2))
```
- μ₁, μ₂: mean của features
- Σ₁, Σ₂: covariance matrix

**Target**: FID < 60 (càng thấp càng tốt)

**Cách đánh giá**:
- FID < 30: Rất tốt, ảnh rất giống thật
- FID 30-60: Tốt, chấp nhận được
- FID 60-100: Trung bình
- FID > 100: Kém, cần cải thiện

#### 3.1.2. LPIPS (Learned Perceptual Image Patch Similarity)

**Mục đích**: Đo sự tương đồng perceptual giữa output và style image

**Nguyên lý**:
- Sử dụng pre-trained CNN (AlexNet/VGG) để extract features
- So sánh features giữa output và style image
- Phù hợp với cảm nhận của con người hơn pixel-level metrics

**Target**: LPIPS < 0.3 (càng thấp càng tốt)

**Cách đánh giá**:
- LPIPS < 0.2: Rất giống style
- LPIPS 0.2-0.3: Giống style
- LPIPS 0.3-0.5: Hơi khác style
- LPIPS > 0.5: Khác nhiều so với style

#### 3.1.3. SSIM (Structural Similarity Index)

**Mục đích**: Đo độ giữ cấu trúc content image

**Nguyên lý**:
- So sánh cấu trúc (luminance, contrast, structure) giữa output và content image
- Đảm bảo output vẫn giữ được bố cục của content

**Target**: SSIM > 0.7 (càng cao càng tốt)

**Cách đánh giá**:
- SSIM > 0.8: Giữ rất tốt cấu trúc content
- SSIM 0.7-0.8: Giữ tốt cấu trúc
- SSIM 0.5-0.7: Mất một phần cấu trúc
- SSIM < 0.5: Mất nhiều cấu trúc

#### 3.1.4. Content Loss & Style Loss

**Content Loss**:
- MSE trên VGG19 feature maps (layer conv4_2)
- Đảm bảo output giữ nội dung của content image

**Style Loss**:
- MSE trên Gram matrices từ VGG19 (multiple layers)
- Đảm bảo output có style giống style image

**Sử dụng trong training**: Làm loss function để optimize model

### 3.2. Test Set

**Content images**: 100-200 ảnh từ COCO val2017 (diverse: landscape, portrait, object, etc.)

**Style images**: 10-20 ảnh đại diện cho mỗi style từ WikiArt

**Evaluation protocol**:
1. Generate output cho mỗi (content, style) pair
2. Tính metrics cho tất cả outputs
3. Report mean ± std cho mỗi metric

### 3.3. So Sánh với Baseline

**Bảng so sánh**:

| Metric | Baseline (SD v1.5) | LoRA Fine-tuned | Cải thiện |
|--------|-------------------|-----------------|-----------|
| FID | ~80-100 | < 60 | ↓ 20-40 |
| LPIPS | ~0.4-0.5 | < 0.3 | ↓ 0.1-0.2 |
| SSIM | ~0.6-0.7 | > 0.7 | ↑ 0.1 |
| Inference Time | ~3-5s | ~3-5s | ≈ |

**Kết luận**:
- LoRA fine-tuning cải thiện chất lượng style transfer
- Giữ được tốc độ inference
- Model size nhỏ hơn (chỉ cần lưu LoRA weights)

---

## 4. HYPERPARAMETER TUNING

### 4.1. Các Hyperparameters Quan Trọng

**LoRA Rank (r)**:
- r = 4: Default, cân bằng giữa chất lượng và số params
- r = 8: Chất lượng tốt hơn nhưng nhiều params hơn
- r = 2: Ít params nhưng có thể kém chất lượng

**Learning Rate**:
- 1e-4: Default
- 5e-5: Conservative, training chậm hơn nhưng ổn định
- 2e-4: Aggressive, có thể không ổn định

**Batch Size**:
- 2-4: Phù hợp với GPU memory
- Lớn hơn → training nhanh hơn nhưng cần nhiều memory

### 4.2. Quy Trình Tuning

1. **Start với default**: r=4, lr=1e-4, batch_size=2
2. **Train 1 style** với default config
3. **Evaluate** trên test set
4. **Adjust** nếu metrics không đạt target:
   - FID cao → tăng r hoặc lr
   - LPIPS cao → tăng style loss weight
   - SSIM thấp → tăng content loss weight
5. **Repeat** cho đến khi đạt target metrics

---

## 5. NGUYÊN LÝ ĐÁNH GIÁ TỔNG THỂ

### 5.1. Đánh Giá Tốt Hay Xấu

**Tốt**:
- FID < 60: Ảnh generated giống ảnh thật
- LPIPS < 0.3: Style transfer thành công
- SSIM > 0.7: Giữ được cấu trúc content
- Inference < 5s: Đủ nhanh cho ứng dụng

**Xấu**:
- FID > 100: Ảnh không tự nhiên
- LPIPS > 0.5: Style transfer không thành công
- SSIM < 0.5: Mất quá nhiều cấu trúc content
- Inference > 10s: Quá chậm

### 5.2. Trade-offs

**Chất lượng vs Tốc độ**:
- LoRA rank cao → chất lượng tốt hơn nhưng inference chậm hơn một chút
- Batch size lớn → training nhanh nhưng cần nhiều memory

**Content vs Style**:
- Content loss weight cao → giữ content tốt nhưng style yếu
- Style loss weight cao → style mạnh nhưng mất content

**Cân bằng**: α·ContentLoss + β·StyleLoss với α=1, β=10 (ưu tiên style)

---

## 6. IMPLEMENTATION CHECKLIST

### Baseline Setup
- [ ] Download SD v1.5 từ Hugging Face
- [ ] Test inference với text prompt
- [ ] Generate baseline outputs cho test set
- [ ] Tính baseline metrics (FID, LPIPS, SSIM)

### LoRA Training
- [ ] Setup LoRA config (r=4)
- [ ] Train LoRA cho 3-5 styles
- [ ] Save LoRA checkpoints
- [ ] Generate outputs với LoRA models

### Evaluation
- [ ] Implement FID calculation
- [ ] Implement LPIPS calculation
- [ ] Implement SSIM calculation
- [ ] Run evaluation trên test set
- [ ] So sánh với baseline
- [ ] Tạo evaluation report

### Hyperparameter Tuning
- [ ] Tune LoRA rank (r=2, 4, 8)
- [ ] Tune learning rate (5e-5, 1e-4, 2e-4)
- [ ] Tune loss weights (α, β, γ)
- [ ] Chọn best config cho mỗi style

---

## 7. TÓM TẮT

**Baseline**: Stable Diffusion v1.5 gốc (download từ Hugging Face, không train)

**Model Training**: 
- Base model: Download sẵn
- LoRA: Train chỉ ~4-8M params trên style images

**Evaluation**:
- Metrics: FID, LPIPS, SSIM, Content Loss, Style Loss
- Target: FID < 60, LPIPS < 0.3, SSIM > 0.7
- So sánh với baseline để chứng minh cải thiện

**Hyperparameter Tuning**:
- Tune rank, learning rate, loss weights
- Đánh giá bằng metrics trên test set

