# Kiến Trúc Mô Hình

## Stable Diffusion v1.5

Stable Diffusion là một mô hình diffusion được train trên latent space thay vì pixel space, giúp giảm đáng kể chi phí tính toán.

### Components

1. **VAE (Variational Autoencoder)**
   - Encoder: Chuyển ảnh từ pixel space (512×512×3) sang latent space (64×64×4)
   - Decoder: Chuyển từ latent space về pixel space

2. **UNet**
   - Denoising network hoạt động trong latent space
   - Nhận input: noisy latent + timestep + condition
   - Output: predicted noise để denoise

3. **Text Encoder (CLIP)**
   - Không sử dụng trong project này (no-prompt approach)

## LoRA (Low-Rank Adaptation)

### Ý Tưởng

Thay vì fine-tune toàn bộ UNet (860M parameters), LoRA chỉ thêm các low-rank matrices vào attention layers.

### Công Thức

```
W' = W + α·A·B
```

Trong đó:
- W: Weight matrix gốc (d × d)
- A: Low-rank matrix (d × r), r << d
- B: Low-rank matrix (r × d)
- α: Scaling factor

### Ưu Điểm

- Giảm số tham số train từ ~860M xuống ~4-8M
- Training nhanh hơn 10-20 lần
- Dễ quản lý nhiều style (mỗi style 1 checkpoint ~10-20MB)
- Có thể kết hợp nhiều LoRA

### Tại Sao Kết Hợp SD Với LoRA?

**Vấn đề của Full Fine-tuning**:
- Stable Diffusion v1.5 có ~860M parameters
- Fine-tune toàn bộ tốn nhiều tài nguyên:
  - GPU memory: ~24GB (cần GPU lớn như A100)
  - Training time: Vài ngày cho 1 style
  - Checkpoint size: ~3-4GB mỗi style
  - Khó quản lý nhiều styles (5 styles = 15-20GB)

**Giải pháp LoRA**:
- Chỉ train ~4-8M parameters (giảm 99% so với full fine-tuning)
- Training nhanh: 2-3 giờ thay vì vài ngày
- Checkpoint nhỏ: ~4-8MB mỗi style (thay vì 3-4GB)
- Tiết kiệm GPU memory: Có thể train trên GPU nhỏ hơn (T4, P100)
- Dễ quản lý: Mỗi style 1 file LoRA nhỏ, dễ switch giữa các styles

**So sánh**:

| Phương pháp | Parameters | Checkpoint Size | Training Time | GPU Memory |
|-------------|-----------|----------------|---------------|------------|
| **Full Fine-tune** | 860M | ~3-4GB | Vài ngày | ~24GB |
| **LoRA (r=4)** | ~4-8M | ~4-8MB | 2-3 giờ | ~12GB |

**Kết luận**:
- SD: Model mạnh, đã được train sẵn, có khả năng generate ảnh tốt
- LoRA: Cách hiệu quả để adapt SD cho style cụ thể mà không cần train lại toàn bộ
- Kết hợp: Tận dụng sức mạnh của SD + training nhanh/gọn của LoRA

### Fine-tune Target

Chỉ fine-tune các attention layers trong UNet:
- Cross-attention layers
- Self-attention layers

## DreamBooth Fine-Tuning

### Khái Niệm

DreamBooth fine-tune Stable Diffusion bằng cách học một token riêng biệt (ví dụ: `sks style`) và kết hợp prior preservation để giữ khả năng sinh ảnh gốc.

### Quy Trình

1. Thu thập 10-20 ảnh cho phong cách cần học
2. Gán caption chứa token đặc biệt (ví dụ: "a sks style painting")
3. Chuẩn bị prior dataset (ảnh random) với prompt chung ("a painting")
4. Fine-tune UNet (và tùy chọn text encoder) để học token mới

### Thông Số Tham Khảo

- Learning rate: 5e-6 – 1e-5
- Batch size: 1-2 (sử dụng gradient accumulation)
- Training steps: 800-1,200 (kèm prior preservation)
- Optimizer: AdamW8bit (bitsandbytes)
- Mixed precision: fp16

### Kết Quả

- Checkpoint: ~2-3GB/style
- Thời gian train: 4-6 giờ trên Kaggle T4/P100
- Chất lượng cao, phù hợp làm baseline so sánh với LoRA

### So Sánh Với LoRA

| Tiêu chí | LoRA | DreamBooth |
|---------|------|------------|
| Params train | ~4-8M | ~100-200M |
| Checkpoint | 4-8MB | 2-3GB |
| Training time | 2-3h/style | 4-6h/style |
| GPU memory | 12-16GB | 16-20GB |
| Ưu điểm | Nhanh, gọn, dễ quản lý nhiều style | Chất lượng cao, giữ style tốt |

## Training Pipeline

1. Encode content image → latent
2. Add noise theo schedule
3. UNet predict noise
4. Compute loss (L2 + LPIPS + StyleLoss)
5. Backprop và update LoRA weights

## Inference Pipeline

1. Encode content image → latent
2. Load LoRA checkpoint tương ứng với style
3. Denoise trong latent space với UNet + LoRA
4. Decode → styled image

