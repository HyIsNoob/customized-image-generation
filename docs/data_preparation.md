# HƯỚNG DẪN CHUẨN BỊ DỮ LIỆU CHO STABLE DIFFUSION + LORA

---

## Bước 1. Tải và Setup Datasets trên Kaggle

Nguồn dữ liệu:
- **COCO 2017**: [https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- **WikiArt**: [https://www.kaggle.com/datasets/steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart)

### Trên Kaggle Notebook:

1. Tạo notebook mới trên Kaggle
2. Add datasets:
   - Vào link COCO 2017 → Click "Add to notebook"
   - Vào link WikiArt → Click "Add to notebook"
3. Enable GPU trong Settings → Accelerator → GPU (P100 hoặc T4)
4. Enable Internet trong Settings

Cấu trúc đường dẫn trên Kaggle:
```
/kaggle/input/coco-2017-dataset/coco2017/
├── train2017/          # 118,287 images (dùng cho training)
└── val2017/            # 5,000 images (dùng cho validation)

/kaggle/input/wikiart/
├── Contemporary_Realism/    # 481 images
├── New_Realism/             # 314 images
├── Synthetic_Cubism/        # 216 images
├── Analytical_Cubism/       # 110 images
└── Action_painting/         # 98 images
```

---

## Bước 2. Chạy EDA Notebook

Chạy notebook `notebooks/00-data-eda.ipynb` để:
- Phân tích đặc điểm datasets (size, aspect ratio, file size)
- Chọn styles phù hợp cho training (đã chọn 5 styles)
- Tạo file `selected_styles.csv` với danh sách styles đã chọn

Output từ EDA:
- `selected_styles.csv`: Danh sách 5 styles đã chọn
- `eda_report.md`: Báo cáo tổng hợp EDA
- Các visualization: `coco_stats.png`, `wikiart_style_distribution.png`, etc.

---

## Bước 3. Preprocessing cho LoRA Training

### 3.1. Resize Images

Stable Diffusion yêu cầu ảnh có kích thước 512x512. Cần resize tất cả ảnh:

**COCO (Content images):**
- Resize về 512x512 (có thể crop center hoặc pad để giữ aspect ratio)
- Lưu vào `/kaggle/working/coco_processed/train/` và `/kaggle/working/coco_processed/val/`

**WikiArt (Style images):**
- Resize về 512x512 cho các styles đã chọn
- Lưu vào `/kaggle/working/wikiart_processed/{style_name}/`

### 3.2. Tạo Caption/Trigger Words

Mỗi style cần có trigger word (từ khóa kích hoạt) để LoRA có thể nhận diện:

| Style | Trigger Word | Caption Template |
|-------|-------------|------------------|
| Contemporary_Realism | `contemporary_realism` | "a painting in contemporary realism style" |
| New_Realism | `new_realism` | "a painting in new realism style" |
| Synthetic_Cubism | `synthetic_cubism` | "a painting in synthetic cubism style" |
| Analytical_Cubism | `analytical_cubism` | "a painting in analytical cubism style" |
| Action_painting | `action_painting` | "a painting in action painting style" |

Lưu caption vào file `.txt` cùng tên với ảnh (ví dụ: `image.jpg` → `image.txt`)

### 3.3. Tổ Chức Dữ Liệu cho LoRA Training

Cấu trúc thư mục sau khi preprocessing:

```
/kaggle/working/
├── coco_processed/
│   ├── train/              # Content images 512x512
│   └── val/                # Content images 512x512
└── wikiart_processed/
    ├── Contemporary_Realism/
    │   ├── image1.jpg
    │   ├── image1.txt      # Caption: "a painting in contemporary realism style"
    │   ├── image2.jpg
    │   └── image2.txt
    ├── New_Realism/
    ├── Synthetic_Cubism/
    ├── Analytical_Cubism/
    └── Action_painting/
```

---

## Bước 4. Data Loading cho Training

### 4.1. Sử dụng Diffusers Dataset

LoRA training với Stable Diffusion sử dụng `diffusers` library:

```python
from diffusers import StableDiffusionPipeline
from datasets import load_dataset

# Load content images (COCO)
content_dataset = load_dataset("imagefolder", data_dir="/kaggle/working/coco_processed/train")

# Load style images (WikiArt) - mỗi style một dataset riêng
style_datasets = {}
for style in selected_styles:
    style_datasets[style] = load_dataset(
        "imagefolder", 
        data_dir=f"/kaggle/working/wikiart_processed/{style}"
    )
```

### 4.2. DataLoader Setup

```python
from torch.utils.data import DataLoader

# Tạo DataLoader cho content images
content_loader = DataLoader(
    content_dataset["train"],
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Tạo DataLoader cho style images (có thể combine hoặc train riêng từng style)
```

---

## Bước 5. Validation và Quality Checks

Trước khi training, kiểm tra:

1. **Corrupted Images**: Đảm bảo tất cả ảnh có thể mở được
2. **Image Count**: Kiểm tra số lượng ảnh mỗi style đủ cho training
3. **Caption Files**: Đảm bảo mỗi ảnh style có file caption tương ứng
4. **Image Size**: Tất cả ảnh đã resize về 512x512

---

## Tóm Tắt

| Bước | Mục tiêu | Công cụ | Kết quả |
|------|----------|---------|---------|
| 1 | Tải và setup datasets | Kaggle Datasets | Datasets có sẵn tại `/kaggle/input/` |
| 2 | Phân tích dữ liệu | EDA Notebook | `selected_styles.csv`, `eda_report.md` |
| 3 | Preprocessing | Resize, tạo captions | `/kaggle/working/coco_processed/`, `/kaggle/working/wikiart_processed/` |
| 4 | Data loading | Diffusers, DataLoader | Sẵn sàng cho training |
| 5 | Quality checks | Validation scripts | Đảm bảo chất lượng dữ liệu |

---

## Lưu Ý

- **Không cần sampling**: LoRA có thể train với số lượng ảnh khác nhau (98-481 images/style)
- **Resize về 512x512** là bắt buộc cho Stable Diffusion
- **Caption files** quan trọng cho LoRA training để model học được style
- **Kaggle output có giới hạn**: Nên download checkpoints và processed data về máy hoặc upload lên Google Drive

---


