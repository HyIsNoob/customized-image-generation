# Evaluation Metrics

Thư mục này chứa kết quả đánh giá metrics cho các models.

## Files

- `dreambooth_metrics.csv`: Metrics cho DreamBooth models (CSV format)
- `dreambooth_metrics.json`: Metrics cho DreamBooth models (JSON format)

## Metrics

### FID (Fréchet Inception Distance)
- **Mục đích**: Đo độ "thật" và chất lượng của ảnh sinh ra
- **So sánh**: Generated images vs Style images (WikiArt)
- **Target**: FID < 60 (càng thấp càng tốt)
- **Interpretation**:
  - FID < 30: Rất tốt
  - FID 30-60: Tốt
  - FID 60-100: Trung bình
  - FID > 100: Kém

### LPIPS (Learned Perceptual Image Patch Similarity)
- **Mục đích**: Đo sự tương đồng perceptual giữa output và style image
- **So sánh**: Generated images vs Style images
- **Target**: LPIPS < 0.3 (càng thấp càng tốt)
- **Interpretation**:
  - LPIPS < 0.2: Rất giống style
  - LPIPS 0.2-0.3: Giống style
  - LPIPS 0.3-0.5: Hơi khác style
  - LPIPS > 0.5: Khác nhiều so với style

### SSIM (Structural Similarity Index)
- **Mục đích**: Đo độ giữ cấu trúc content image
- **So sánh**: Generated images vs Content images
- **Target**: SSIM > 0.7 (càng cao càng tốt)
- **Interpretation**:
  - SSIM > 0.8: Giữ rất tốt cấu trúc content
  - SSIM 0.7-0.8: Giữ tốt cấu trúc content
  - SSIM 0.5-0.7: Giữ một phần cấu trúc
  - SSIM < 0.5: Mất nhiều cấu trúc content

## Models Đánh Giá

1. **baseline**: Stable Diffusion v1.5 gốc (không fine-tune)
2. **dreambooth_contemporaryrealism**: DreamBooth fine-tuned cho Contemporary_Realism
3. **dreambooth_newrealism**: DreamBooth fine-tuned cho New_Realism

## Cách Sử Dụng

### Xem kết quả:
```python
import pandas as pd
df = pd.read_csv("results/metrics/dreambooth_metrics.csv")
print(df)
```

### So sánh với baseline:
- FID improvement = baseline_fid - model_fid (số dương = tốt hơn)
- LPIPS improvement = baseline_lpips - model_lpips (số dương = tốt hơn)
- SSIM change = model_ssim - baseline_ssim (số dương = tốt hơn)

## Lưu ý

- Metrics được tính trên 5-20 samples (tùy theo số lượng style images có sẵn)
- Khi có LoRA models, sẽ thêm vào evaluation và so sánh
- Xem notebook `04_Evaluation_Metrics.ipynb` để biết chi tiết implementation

