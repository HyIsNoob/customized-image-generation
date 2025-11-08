# Hướng Dẫn Setup Dự Án

## 1. Clone Repository

```bash
git clone https://github.com/HyIsNoob/customized-image-generation.git
cd customized-image-generation
```

## 2. Add Collaborators

Team lead (Nguyễn Khang Hy) cần add các thành viên vào repository:

1. Vào https://github.com/HyIsNoob/customized-image-generation
2. Vào Settings → Collaborators
3. Add username GitHub của các thành viên:
   - **Nguyễn Minh Quốc** (23521304) - Cần username GitHub
   - **Phan Đức Thành Phát** (23521149) - Cần username GitHub

Sau khi được add, các thành viên sẽ nhận email invitation và cần accept.

## 3. Setup Environment

### Trên Local Machine

```bash
pip install -r requirements.txt
```

### Trên Google Colab

```python
!git clone https://github.com/HyIsNoob/customized-image-generation.git
%cd customized-image-generation
!pip install -r requirements.txt
```

## 4. Cấu Hình

Chỉnh sửa `src/configs/lora_config.yaml`:

```yaml
content_dir: "path/to/coco/train"
style_dir: "path/to/wikiart"
output_dir: "results/lora_checkpoints"
```

## 5. Download Datasets

### COCO 2017

1. Vào https://cocodataset.org/#download
2. Download "2017 Train images" (18GB)
3. Extract vào `data/coco/train/`

### WikiArt

1. Download từ https://www.wikiart.org/ hoặc sử dụng dataset có sẵn
2. Organize theo phong cách:
   ```
   data/wikiart/
   ├── monet/
   ├── ukiyo-e/
   ├── pop-art/
   └── ...
   ```

## 6. Git Workflow

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

## 8. Lưu Ý

- **KHÔNG commit** datasets, model weights, large files
- Chỉ commit: notebooks, scripts, configs, documentation
- Models và data lưu trên Google Drive
- Sync thường xuyên với `main` branch

## 9. Troubleshooting

### Lỗi Permission

Nếu không push được, kiểm tra:
1. Đã accept invitation chưa?
2. Đang ở đúng branch chưa?
3. Remote đã setup đúng chưa?

### Lỗi Import

```bash
pip install --upgrade -r requirements.txt
```

### Lỗi GPU trên Colab

Đảm bảo Runtime → Change runtime type → GPU được chọn.

