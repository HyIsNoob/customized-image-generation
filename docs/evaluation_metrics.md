# Evaluation Metrics

## FID (Fréchet Inception Distance)

Đo khoảng cách giữa phân phối của ảnh thật và ảnh sinh ra.

**Công thức**:
```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^(1/2))
```

**Target**: FID < 60

## LPIPS (Learned Perceptual Image Patch Similarity)

Đo khoảng cách perceptual giữa hai ảnh, phù hợp với cảm nhận của con người.

**Target**: LPIPS < 0.3

## SSIM (Structural Similarity Index)

Đo độ tương đồng về cấu trúc giữa ảnh output và content.

**Target**: SSIM > 0.7

## Content Loss

MSE trên feature maps từ VGG19.

## Style Loss

MSE trên Gram matrices từ VGG19.

## Inference Time

Thời gian để generate 1 ảnh (512×512).

**Target**: < 5s/image

## Comparative Reporting

- Báo cáo riêng metrics cho LoRA và DreamBooth
- Bảng tổng hợp: mean ± std cho từng metric và từng phong cách
- Thêm cột chi phí: thời gian train, kích thước checkpoint, GPU memory

