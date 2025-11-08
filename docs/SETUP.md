# Hướng Dẫn Setup Dự Án

## 1. Setup Environment

### Trên Local Machine

```bash
pip install -r requirements.txt
```

### Trên Kaggle Notebooks

```python
# Clone repository
!git clone https://github.com/HyIsNoob/customized-image-generation.git
%cd customized-image-generation

# Cài đặt dependencies
!pip install -r requirements.txt
```

**Lưu ý**: 
- Bật Internet trong Settings → Internet
- Bật GPU trong Settings → Accelerator → GPU
- Add datasets: Add Data → Search "coco-2017-dataset" và "wikiart" → Add

## 2. Cấu Hình

Chỉnh sửa `src/configs/lora_config.yaml`:

```yaml
content_dir: "path/to/coco/train"
style_dir: "path/to/wikiart"
output_dir: "results/lora_checkpoints"
```

## 3. Add Datasets trên Kaggle

### COCO 2017

1. Vào https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
2. Click "New Notebook" hoặc "Add to notebook"
3. Dataset sẽ có sẵn tại `/kaggle/input/coco-2017-dataset/`

### WikiArt

1. Vào https://www.kaggle.com/datasets/steubk/wikiart
2. Click "Add to notebook" (chọn notebook đã tạo)
3. Dataset sẽ có sẵn tại `/kaggle/input/wikiart/`

**Cấu trúc đường dẫn trên Kaggle:**
```
/kaggle/input/coco-2017-dataset/coco2017/train2017/  # Content images
/kaggle/input/wikiart/wikiart/                       # Style images
/kaggle/working/                                     # Output directory
```

## 4. Git Workflow

### Tạo Branch Mới

```bash
git checkout -b your-branch-name
```

Ví dụ:
- `data-eda` (Khang Hy)
- `lora-training` (Minh Quốc)
- `demo-integration` (Thành Phát)

### Commit và Push

```bash
git add .
git commit -m "Description of changes"
git push origin your-branch-name
```

### Tạo Pull Request

1. Vào GitHub repository
2. Click "New Pull Request"
3. Chọn branch của bạn → `main`
4. Mô tả changes
5. Request review từ team lead

## 7. Cấu Trúc Làm Việc

### Notebooks

Tất cả notebooks đặt trong `notebooks/`:
- `00_Data_EDA.ipynb` - Khang Hy
- `01_LoRA_Training.ipynb` - Minh Quốc
- `02_Inference_Pipeline.ipynb` - Thành Phát
- `03_Demo_Application.ipynb` - Thành Phát
- `04_Evaluation_Metrics.ipynb` - Khang Hy
- `05_Results_Analysis.ipynb` - Khang Hy

### Source Code

Code Python đặt trong `src/`:
- `src/models/` - Model implementations
- `src/utils/` - Utility functions
- `src/configs/` - Configuration files
- `src/train_lora.py` - Training script
- `src/infer.py` - Inference script
- `src/demo.py` - Demo application

## 5. Lưu Ý

- **KHÔNG commit** datasets, model weights, large files
- Chỉ commit: notebooks, scripts, configs, documentation
- Models và data lưu trên Google Drive
- Sync thường xuyên với `main` branch

## 6. Troubleshooting

### Lỗi Permission

Nếu không push được, kiểm tra:
1. Đã accept invitation chưa?
2. Đang ở đúng branch chưa?
3. Remote đã setup đúng chưa?

### Lỗi Import

```bash
pip install --upgrade -r requirements.txt
```

### Lỗi GPU trên Kaggle

Đảm bảo Settings → Accelerator → GPU được chọn (P100 hoặc T4).

