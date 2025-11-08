# Customized Image Generation - Style Transfer via Stable Diffusion + LoRA Fine-Tuning

## Tổng Quan Dự Án

### Giới Thiệu

Dự án **Customized Image Generation** nghiên cứu và triển khai phương pháp chuyển đổi phong cách nghệ thuật cho ảnh sử dụng Stable Diffusion kết hợp với kỹ thuật LoRA (Low-Rank Adaptation) fine-tuning. Hệ thống cho phép người dùng cung cấp ảnh nội dung (content image) và chọn phong cách nghệ thuật (style class) để tự động tạo ra ảnh mới giữ nguyên bố cục nhưng mang đặc trưng phong cách đã chọn.

### Thông Tin Môn Học

- **Môn học**: Các Kỹ Thuật Học Sâu và Ứng Dụng – CS431.Q12
- **Giảng viên**: Nguyễn Vinh Tiệp & Chế Quang Huy
- **Thời gian**: 2 tuần (Deadline: 22/11/2025)

### Thành Viên Nhóm

1. **Nguyễn Khang Hy** (2352662) - Team Lead
2. **Nguyễn Minh Quốc** (23521304)
3. **Phan Đức Thành Phát** (23521149)

---

## Động Lực và Mục Tiêu

### Vấn Đề Hiện Tại

- Các hệ thống AI sinh ảnh hiện tại (DALL-E, Midjourney) yêu cầu prompt văn bản chính xác, khó kiểm soát kết quả
- Style transfer truyền thống (AdaIN, SANet) có giới hạn về chất lượng và độ tự nhiên
- Fine-tuning toàn bộ Stable Diffusion tốn tài nguyên và thời gian

### Giải Pháp Đề Xuất

- Sử dụng Stable Diffusion v1.5 làm base model
- Fine-tuning bằng LoRA chỉ trên UNet attention layers
- Không cần text prompt, chỉ cần content image + style class
- Nhẹ, nhanh, dễ mở rộng phong cách mới

### Mục Tiêu

- Fine-tune thành công 3-5 phong cách nghệ thuật (Monet, Ukiyo-e, Pop Art, Sketch, Minimalism)
- Ảnh sinh ra đạt FID < 60, LPIPS thấp, SSIM cao
- Demo chạy ổn định, thời gian inference < 5s/ảnh
- Model gọn < 1 tỉ tham số, training < vài ngày

---

## Bài Toán

### Phát Biểu Bài Toán

Fine-tune mô hình Stable Diffusion để sinh ảnh theo phong cách cụ thể (style class) dựa trên ảnh gốc (content image). Mô hình học phân phối có điều kiện p(x | style), cho phép tạo ra ảnh mới giữ bố cục content nhưng mang đặc trưng của style.

### Input/Output

**Input:**
- Content_Image: Ảnh gốc giữ bố cục và nội dung chính
- Style_Class hoặc Style_Image: Lựa chọn phong cách từ thư viện có sẵn hoặc upload ảnh phong cách
- Tùy chọn: style_strength, mask vùng áp style

**Output:**
- Ảnh mới giữ bố cục content và mang phong cách tương ứng

### Ràng Buộc

1. **Content Preservation**: Giữ cấu trúc và bố cục của ảnh gốc
2. **Style Transfer**: Tái tạo texture, màu sắc, họa tiết của ảnh style
3. **Efficiency**: Model gọn, training nhanh, inference nhanh

---

## Kiến Trúc Mô Hình

### Stable Diffusion v1.5

- **Base Model**: `runwayml/stable-diffusion-v1-5`
- **Components**:
  - VAE Encoder/Decoder: Encode/decode giữa pixel space và latent space
  - UNet: Denoising network trong latent space
  - CLIP Text Encoder: (Không sử dụng trong project này)

### LoRA (Low-Rank Adaptation)

**Ý tưởng**: Thay vì fine-tune toàn bộ UNet, chỉ thêm các low-rank matrices vào attention layers

**Công thức**: `W' = W + α·A·B` với A ∈ R^(d×r), B ∈ R^(r×d), r << d

**Ưu điểm**:
- Giảm số tham số train từ ~860M xuống ~4-8M
- Training nhanh hơn 10-20 lần
- Dễ quản lý nhiều style (mỗi style 1 checkpoint)
- Có thể kết hợp nhiều LoRA

**Fine-tune Target**: UNet attention layers (cross-attention và self-attention)

---

## Pipeline Chi Tiết

### 1. Chuẩn Bị Dữ Liệu

**Content Dataset**: COCO 2017
- 118k train images, 5k val images
- Ảnh thực tế đời thường, bố cục tự nhiên
- Resize về 512x512

**Style Dataset**: WikiArt
- 3-5 phong cách nghệ thuật
- 50-100 ảnh/phong cách
- Các phong cách: Monet, Ukiyo-e, Pop Art, Sketch, Minimalism

### 2. Fine-tune LoRA

**Cấu hình**:
- Base model: `runwayml/stable-diffusion-v1-5`
- Fine-tune target: UNet attention layers
- Rank: 4
- Learning rate: 1e-4
- Batch size: 2-4
- Steps: 5,000-8,000/phong cách
- Optimizer: AdamW
- Scheduler: Cosine
- Training time: 2-3 giờ/phong cách (Colab T4/A100)

**Loss Function**:
```
L_total = α·L2 + β·LPIPS + γ·StyleLoss
```
- L2 loss: Tái tạo chi tiết ảnh
- LPIPS: Duy trì độ tự nhiên theo cảm nhận người nhìn
- Style loss (Gram matrix): Giữ họa tiết, màu sắc của style

### 3. Inference

1. Encode Content_Image → latent vector (VAE encoder)
2. Load LoRA checkpoint tương ứng với style đã chọn
3. Áp dụng LoRA weights vào UNet
4. Denoise trong latent space với UNet
5. Decode → ảnh mới mang phong cách đã học (VAE decoder)

---

## Dataset

| Loại | Tên Dataset | Quy Mô | Ghi Chú |
|------|------------|--------|---------|
| **Content** | COCO 2017 | 118k train, 5k val | Ảnh thực tế đời thường |
| **Style** | WikiArt | 3-5 phong cách, 50-100 ảnh/phong cách | Tranh nghệ thuật |

---

## Phân Công Công Việc

### Nguyễn Khang Hy (2352662) - Team Lead

**Trách nhiệm chính**:
- Quản lý dự án: Timeline, phân công, theo dõi tiến độ
- Tích hợp: Đảm bảo các phần code hoạt động cùng nhau
- Documentation: README, báo cáo cuối kỳ, presentation

**Công việc kỹ thuật**:
1. **EDA & Data Analysis**:
   - Phân tích dataset COCO và WikiArt
   - Thống kê phân phối, visualize samples
   - Identify potential issues

2. **Evaluation Framework**:
   - Implement metrics: FID, LPIPS, SSIM
   - Content loss, Style loss calculation
   - Inference time benchmark
   - Create test suite với diverse samples

3. **Results & Reporting**:
   - Tổng hợp kết quả training
   - So sánh các phong cách
   - Viết báo cáo cuối kỳ

**Deliverables**:
- Notebook: `00_Data_EDA.ipynb`
- Notebook: `04_Evaluation_Metrics.ipynb`
- Notebook: `05_Results_Analysis.ipynb`
- Script: `eval_utils.py`, `eval.py`
- Evaluation report
- Final report

---

### Nguyễn Minh Quốc (23521304) - LoRA Training Specialist

**Trách nhiệm chính**:
- Fine-tuning LoRA cho các phong cách nghệ thuật
- Tối ưu pipeline huấn luyện
- Hyperparameter tuning

**Công việc kỹ thuật**:
1. **LoRA Implementation**:
   - Implement LoRA layers cho UNet
   - Setup training pipeline với diffusers library
   - Loss function implementation (L2 + LPIPS + StyleLoss)

2. **Training & Optimization**:
   - Fine-tune LoRA cho 3-5 phong cách
   - Hyperparameter tuning (rank, learning rate, batch size)
   - Monitoring training progress
   - Save/load LoRA checkpoints

3. **Data Pipeline**:
   - Prepare training data (content-style pairs)
   - Data augmentation
   - DataLoader implementation

**Deliverables**:
- Notebook: `01_LoRA_Training.ipynb`
- Script: `src/models/lora.py`
- Script: `src/train_lora.py`
- Config: `src/configs/lora_config.yaml`
- Trained LoRA checkpoints (3-5 styles)
- Training logs và metrics

---

### Phan Đức Thành Phát (23521149) - Integration & Demo

**Trách nhiệm chính**:
- Tích hợp mô hình vào inference pipeline
- Xây dựng giao diện demo
- Visualization kết quả

**Công việc kỹ thuật**:
1. **Inference Pipeline**:
   - Implement inference script với LoRA
   - Load content image và style LoRA
   - Generate styled images
   - Optimize inference speed

2. **Demo Application**:
   - Gradio interface trên Colab/Hugging Face
   - Upload content image
   - Chọn style class (dropdown)
   - Chỉnh style_strength, mask nếu cần
   - Hiển thị kết quả và tải về

3. **Visualization**:
   - So sánh content/style/output side-by-side
   - Loss curves, training progress plots
   - Quality comparison giữa các phong cách

**Deliverables**:
- Notebook: `02_Inference_Pipeline.ipynb`
- Notebook: `03_Demo_Application.ipynb`
- Script: `src/infer.py`
- Script: `src/demo.py`
- Demo app (Gradio)
- Demo video/screenshots

---

## Timeline (2 Tuần)

### Week 1: Setup & Training

**Ngày 1-2: Setup & Data Preparation**
- [ ] Setup GitHub repo, cấu trúc thư mục
- [ ] Khang Hy: Download và EDA datasets
- [ ] Minh Quốc: Setup LoRA training environment
- [ ] Thành Phát: Research inference pipeline

**Ngày 3-5: LoRA Training**
- [ ] Minh Quốc: Implement LoRA training pipeline
- [ ] Minh Quốc: Fine-tune 3-5 phong cách (parallel nếu có GPU)
- [ ] Khang Hy: Setup evaluation framework
- [ ] Thành Phát: Implement inference script

**Ngày 6-7: Integration & Testing**
- [ ] Thành Phát: Tích hợp inference pipeline
- [ ] Khang Hy: Test evaluation metrics
- [ ] Minh Quốc: Tối ưu training nếu cần
- [ ] Toàn team: Testing end-to-end

### Week 2: Evaluation & Demo

**Ngày 8-9: Evaluation & Analysis**
- [ ] Khang Hy: Evaluate tất cả LoRA models
- [ ] Khang Hy: So sánh metrics giữa các phong cách
- [ ] Thành Phát: Visualization kết quả
- [ ] Minh Quốc: Fine-tune nếu cần cải thiện

**Ngày 10-11: Demo & Documentation**
- [ ] Thành Phát: Hoàn thành demo app
- [ ] Khang Hy: Draft báo cáo
- [ ] Toàn team: Testing demo
- [ ] Record demo video

**Ngày 12-14: Finalization**
- [ ] Khang Hy: Hoàn thành báo cáo
- [ ] Toàn team: Review và chỉnh sửa
- [ ] Chuẩn bị presentation
- [ ] Final submission (22/11/2025)

---

## Cấu Trúc Thư Mục

```
customized-image-generation/
│
├── README.md                          # File mô tả toàn bộ dự án
├── .gitignore                         # Loại bỏ checkpoints, datasets, models
├── requirements.txt                   # Danh sách dependencies
│
├── notebooks/                         # Nơi làm việc chính
│   ├── 00_Data_EDA.ipynb              # EDA và phân tích dữ liệu
│   ├── 01_LoRA_Training.ipynb         # Huấn luyện LoRA cho các phong cách
│   ├── 02_Inference_Pipeline.ipynb   # Inference với LoRA
│   ├── 03_Demo_Application.ipynb     # Giao diện demo Gradio
│   ├── 04_Evaluation_Metrics.ipynb    # Tính FID, LPIPS, SSIM
│   └── 05_Results_Analysis.ipynb      # Phân tích và so sánh kết quả
│
├── src/
│   ├── models/                        # Chứa kiến trúc mạng
│   │   ├── lora.py                    # LoRA implementation
│   │   └── __init__.py
│   │
│   ├── utils/                         # Các hàm tiện ích
│   │   ├── data_utils.py              # Load ảnh, data augmentation
│   │   ├── train_utils.py              # Train loop helpers
│   │   ├── eval_utils.py               # FID, LPIPS, SSIM, Style loss
│   │   └── viz_utils.py                # Plot, visualize results
│   │
│   ├── configs/                        # File cấu hình siêu tham số
│   │   └── lora_config.yaml           # LoRA training config
│   │
│   ├── train_lora.py                  # Entry point huấn luyện
│   ├── infer.py                       # Entry point inference
│   ├── eval.py                        # Entry point evaluation
│   └── demo.py                        # Entry point demo app
│
├── docs/                              # Tài liệu chi tiết
│   ├── architecture.md                # Giải thích SD + LoRA
│   ├── training_guide.md              # Hướng dẫn training
│   └── evaluation_metrics.md         # Cách tính các chỉ số
│
└── results/                           # Kết quả mẫu
    ├── samples/                       # Output samples
    └── metrics/                       # Metrics và logs
```

---

## Tech Stack & Tools

### Development Environment

- **Primary**: Google Colab (GPU: T4/V100/A100)
- **Storage**: Google Drive (sync với Colab)
- **Version Control**: GitHub

### Core Libraries

```python
# Deep Learning
torch >= 2.0.0
torchvision >= 0.15.0
diffusers >= 0.21.0
transformers >= 4.30.0
accelerate >= 0.20.0

# Stable Diffusion
safetensors
peft  # LoRA implementation

# Computer Vision
opencv-python
Pillow
scikit-image

# Evaluation Metrics
pytorch-fid
lpips
torchmetrics

# Visualization
matplotlib
seaborn

# Demo
gradio >= 3.50.0

# Utils
numpy
pandas
tqdm
PyYAML
```

---

## Evaluation Metrics

### Objective Metrics

1. **FID Score**: Fréchet Inception Distance - đo độ "thật" của ảnh sinh ra
2. **LPIPS**: Learned Perceptual Image Patch Similarity - đo cảm nhận giữa output và style
3. **SSIM**: Structural Similarity Index - đo độ giữ cấu trúc content
4. **Content Loss**: MSE trên VGG feature maps
5. **Style Loss**: MSE trên Gram matrices
6. **Inference Time**: ms/image (512×512)

### Target Metrics

- FID < 60
- LPIPS < 0.3
- SSIM > 0.7
- Inference time < 5s/image

---

## Success Criteria

### Minimum (Pass)

- [ ] Fine-tune thành công ít nhất 3 phong cách
- [ ] FID < 100 trên test set
- [ ] Inference < 10s/image
- [ ] Demo app functional

### Target (Good)

- [ ] Fine-tune thành công 3-5 phong cách
- [ ] FID < 60, LPIPS < 0.3, SSIM > 0.7
- [ ] Inference < 5s/image
- [ ] Demo với style strength control
- [ ] Comprehensive evaluation report

### Stretch (Excellent)

- [ ] Fine-tune 5+ phong cách
- [ ] FID < 50
- [ ] Regional style transfer (mask)
- [ ] Multi-style blending
- [ ] Deploy trên Hugging Face Spaces

---

## Tài Liệu Tham Khảo

### Papers

1. **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
2. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. **Style Transfer**: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

### Implementation References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PEFT Library](https://github.com/huggingface/peft)
- [LoRA for Stable Diffusion](https://huggingface.co/docs/peft/task_guides/stable_diffusion)

### Datasets

- [COCO 2017](https://cocodataset.org/#download)
- [WikiArt](https://www.wikiart.org/)

---

## Checklist Tổng Hợp

### Setup Phase

- [x] Tạo GitHub repo
- [x] Setup cấu trúc thư mục
- [ ] Download datasets
- [ ] Setup Colab environment

### Development Phase

- [ ] Data EDA hoàn chỉnh
- [ ] LoRA training pipeline
- [ ] Fine-tune 3-5 phong cách
- [ ] Inference pipeline
- [ ] Evaluation framework

### Finalization Phase

- [ ] Comprehensive evaluation
- [ ] Demo app
- [ ] Documentation complete
- [ ] Final report
- [ ] Presentation slides
- [ ] Submission

---

## Liên Hệ

- **GitHub**: [Repository Link]
- **Issues**: Sử dụng GitHub Issues để báo lỗi và đề xuất

---

**Lưu ý**: Không commit datasets, model weights, large files lên GitHub. Chỉ commit notebooks, scripts, configs, documentation. Models và data lưu trên Google Drive.
