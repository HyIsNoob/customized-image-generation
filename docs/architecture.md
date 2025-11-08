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

### Fine-tune Target

Chỉ fine-tune các attention layers trong UNet:
- Cross-attention layers
- Self-attention layers

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

