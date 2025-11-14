# Customized Image Generation - Style Transfer via Stable Diffusion + LoRA Fine-Tuning

## Tổng Quan Dự Án

### Giới Thiệu

Dự án **Customized Image Generation** nghiên cứu và triển khai phương pháp chuyển đổi phong cách nghệ thuật cho ảnh sử dụng Stable Diffusion kết hợp với kỹ thuật LoRA (Low-Rank Adaptation) fine-tuning. Hệ thống cho phép người dùng cung cấp ảnh nội dung (content image) và chọn phong cách nghệ thuật (style class) để tự động tạo ra ảnh mới giữ nguyên bố cục nhưng mang đặc trưng phong cách đã chọn.

### Thông Tin Môn Học

- **Môn học**: Các Kỹ Thuật Học Sâu và Ứng Dụng – CS431.Q12
- **Giảng viên**: Nguyễn Vinh Tiệp & Chế Quang Huy
- **Thời gian**: 2 tuần (Deadline: 22/11/2025)

### Thành Viên Nhóm

1. **Nguyễn Khang Hy** (2352662)
2. **Phan Đức Thành Phát** (23521149)
3. **Minh Quốc** (MSSV)

---

## Động Lực và Mục Tiêu

### Vấn Đề Hiện Tại

- Các hệ thống AI sinh ảnh hiện tại (DALL-E, Midjourney) yêu cầu prompt văn bản chính xác, khó kiểm soát kết quả
- Style transfer truyền thống (AdaIN, các phương pháp CNN-based) có giới hạn về chất lượng và độ tự nhiên
- Fine-tuning toàn bộ Stable Diffusion tốn tài nguyên và thời gian

### Giải Pháp Đề Xuất

- Sử dụng Stable Diffusion v1.5 làm base model
- Fine-tuning bằng **LoRA** (Low-Rank Adaptation) chỉ trên UNet attention layers
- Fine-tuning bằng **DreamBooth** với prior preservation
- Fine-tuning bằng **Textual Inversion** cho embedding tokens
- So sánh 3 phương pháp fine-tuning trên cùng dataset và metrics
- Nhẹ, nhanh, dễ mở rộng phong cách mới

### Mục Tiêu

- Fine-tune thành công 3-5 phong cách nghệ thuật
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

### 2a. Fine-tune LoRA

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

### 2b. Fine-tune DreamBooth

**Mục tiêu**: Fine-tune UNet với prior preservation để học phong cách nghệ thuật cụ thể. Do hạn chế về phần cứng (GPU memory trên Kaggle), chúng tôi chỉ fine-tune **attention layers** của UNet thay vì toàn bộ UNet.

**Lý do chỉ train attention layers**:
- **Hạn chế phần cứng**: Kaggle GPU (T4/P100) có ~16GB VRAM, không đủ để train full UNet (~860M parameters) với batch size hợp lý
- **Memory requirements**: 
  - Full UNet training: ~15-16GB VRAM (model + optimizer state + activations)
  - Attention layers only: ~5-6GB VRAM (chỉ ~30% parameters cần train)
- **Trade-off**: Giảm memory usage đáng kể nhưng vẫn giữ được khả năng học style transfer hiệu quả vì attention layers là phần quan trọng nhất trong UNet để học các đặc trưng style

**Cấu hình**:
- Base model: Stable Diffusion v1.5
- **Fine-tune target: Chỉ attention layers của UNet** (cross-attention và self-attention)
- Parameters train: ~30% của UNet (~260M parameters thay vì 860M)
- Input size: 256×256 (giảm từ 512 để tiết kiệm memory)
- Learning rate: 5e-6
- Batch size: 1 (với gradient accumulation 16)
- Steps: 1k-2k per style
- Optimizer: AdamW
- Loss: MSE loss + Prior preservation loss (weight=1.0)
- Training time: ~2-3 giờ/style trên Kaggle T4/P100

**Memory optimizations** (bắt buộc do hạn chế phần cứng):
- **CPU offloading**: VAE và Text Encoder ở CPU, chỉ move lên GPU khi encode
- **VAE slicing và tiling**: Chia VAE encoding thành các slice/tile nhỏ hơn
- **Attention slicing**: Chia attention mechanism thành các slice
- **Gradient checkpointing**: Trade computation for memory
- **Resolution reduction**: 512 → 256 để giảm memory cho activations
- **Gradient accumulation**: Batch size 1 với accumulation 16 để mô phỏng batch lớn hơn

**Kết quả**:
- Checkpoint: Chỉ lưu attention layers đã train (~260M parameters), có thể load vào base model
- Memory usage: ~5-6GB VRAM (thay vì ~15GB nếu train full UNet)
- Chất lượng: Vẫn đạt được style transfer tốt vì attention layers là phần quan trọng nhất
- **Lưu ý**: Nếu có GPU lớn hơn (A100 40GB+), có thể train full UNet để đạt chất lượng cao hơn

### 2c. Fine-tune Textual Inversion

**Mục tiêu**: Học một embedding mới trong CLIP text encoder đại diện cho phong cách (`sks style`) thay vì fine-tune toàn bộ UNet.

**Cấu hình**:
- Base model: `runwayml/stable-diffusion-v1-5`
- Modules train: Textual embedding (768 chiều) dành cho token mới
- Learning rate: 5e-4 – 1e-3
- Batch size: 1 (gradient accumulation 4)
- Steps: 500-1,000 per style
- Optimizer: AdamW
- Scheduler: Cosine/Constant
- Training time: < 1 giờ/style (Kaggle T4/P100)

**Yêu cầu thêm**:
- Captions chứa token đặc biệt (`sks style painting`)
- 10-20 instance images đã resize 512x512
- Theo dõi loss embedding để tránh overfit

**Kết quả**:
- Checkpoint embedding < 1MB/style (dễ chia sẻ)
- Có thể kết hợp với LoRA hoặc dùng riêng để generate ảnh theo phong cách

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

### Nguyễn Khang Hy (2352662) - DreamBooth Training & Evaluation

**Trách nhiệm chính**:
- Quản lý dự án: Timeline, phân công, theo dõi tiến độ
- Tích hợp: Đảm bảo các phần code hoạt động cùng nhau
- Documentation: README, báo cáo cuối kỳ, presentation

**Công việc kỹ thuật**:
1. **EDA & Data Analysis**:
   - Phân tích dataset COCO và WikiArt
   - Thống kê phân phối, visualize samples
   - Identify potential issues

2. **DreamBooth Training**:
   - Fine-tune DreamBooth cho 3-5 phong cách nghệ thuật
   - **Chỉ train attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
   - Tối ưu memory cho Kaggle GPU (CPU offloading, VAE slicing, attention slicing, resolution reduction)
   - Implement freeze/unfreeze logic để chỉ train attention layers
   - Hyperparameter tuning (learning rate, prior loss weight, steps)
   - Ghi nhận thời gian train, kích thước checkpoint, GPU usage
   - Save/load DreamBooth checkpoints (chỉ attention layers)

3. **Evaluation Framework**:
   - Implement metrics: FID, LPIPS, SSIM
   - Content loss, Style loss calculation
   - Inference time benchmark
   - Create test suite với diverse samples
   - So sánh LoRA vs DreamBooth vs Textual Inversion

4. **Results & Reporting**:
   - Tổng hợp kết quả training từ cả 3 phương pháp
   - So sánh các phong cách và các phương pháp fine-tuning
   - Viết báo cáo cuối kỳ

**Deliverables**:
- Notebook: `00_Data_EDA.ipynb`
- Notebook: `01b_DreamBooth_Training.ipynb`
- Notebook: `04_Evaluation_Metrics.ipynb`
- Notebook: `05_Results_Analysis.ipynb`
- Script: `src/train_dreambooth.py` (nếu cần)
- Script: `eval_utils.py`
- Script: `eval.py`
- Trained DreamBooth checkpoints (3-5 styles)
- Evaluation report
- Final report

---

### Phan Đức Thành Phát (23521149) - LoRA Training

**Trách nhiệm chính**:
- Fine-tuning LoRA cho các phong cách nghệ thuật
- Tối ưu pipeline huấn luyện
- Hyperparameter tuning
- Cung cấp inference pipeline ổn định cho toàn hệ thống

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

4. **Inference Support**:
   - Implement inference script và tối ưu tốc độ
   - Bàn giao checkpoints + hướng dẫn load LoRA cho pipeline chung
   - Hỗ trợ Minh Quốc tích hợp các lựa chọn LoRA trong demo

**Deliverables**:
- Notebook: `01a_LoRA_Training.ipynb`
- Notebook: `02_Inference_Pipeline.ipynb`
- Script: `src/models/lora.py`
- Script: `src/train_lora.py`
- Config: `src/configs/lora_config.yaml`
- Script: `src/infer.py`
- Script: `src/demo.py`
- Trained LoRA checkpoints (3-5 styles)
- Training logs và metrics

---

### Trần Minh Quốc (MSSV) - Textual Inversion & Demo

**Trách nhiệm chính**:
- Fine-tuning textual inversion embeddings cho từng phong cách
- Phát triển demo Gradio tích hợp lựa chọn mô hình (LoRA / DreamBooth / Textual Inversion)
- Phối hợp inference pipeline để hỗ trợ nhiều baseline

**Công việc kỹ thuật**:
1. **Textual Inversion Training**:
   - Chuẩn bị instance captions với token đặc biệt
   - Huấn luyện embedding trên SD v1.5 (500-1000 steps/style)
   - Quản lý checkpoint embeddings (.pt /.bin)
   - Ghi nhận thời gian train, kích thước checkpoint, GPU usage

2. **Demo & UX**:
   - Mở rộng notebook `03_Demo_Application.ipynb`
   - Cho phép người dùng chọn mô hình (LoRA / DreamBooth / Textual Inversion) + tham số (style strength, steps)
   - Tích hợp inference pipeline cho cả 3 phương pháp
   - Xuất bản hướng dẫn sử dụng/demo video

3. **Inference Integration**:
   - Cập nhật `src/infer.py` để hỗ trợ textual inversion weights
   - Đảm bảo compatibility với LoRA và DreamBooth outputs
   - Hỗ trợ load và switch giữa các model types

**Deliverables**:
- Notebook: `01c_Textual_Inversion_Training.ipynb`
- Notebook: `03_Demo_Application.ipynb`
- Textual inversion embedding checkpoints (mỗi style)
- Script: `src/train_textual_inversion.py`
- Demo app (Gradio) + video/screenshots

---

## Timeline (2 Tuần)

### Week 1: Setup & Training

**Ngày 1-2: Setup & Data Preparation**
- [ ] Hy: Download datasets, chạy EDA, chuẩn bị caption + trigger words
- [ ] Hy: Setup môi trường DreamBooth, kiểm tra GPU và dependencies, tối ưu memory
- [ ] Phát: Setup môi trường LoRA, kiểm tra GPU và dependencies
- [ ] Phát: Khởi tạo skeleton inference pipeline
- [ ] Minh Quốc: Chuẩn bị captions/token cho textual inversion, kiểm tra script training

**Ngày 3-5: LoRA & DreamBooth Training (song song)**
- [ ] Phát: Implement LoRA training pipeline với diffusers
- [ ] Phát: Fine-tune LoRA cho 3-5 phong cách (parallel nếu GPU cho phép)
- [ ] Hy: Implement DreamBooth training pipeline với memory optimizations
- [ ] Hy: Fine-tune DreamBooth cho 3-5 phong cách
- [ ] Hy: Hoàn thiện evaluation framework (FID, LPIPS, SSIM)
- [ ] Minh Quốc: Chạy thử textual inversion với 1 phong cách để kiểm tra config

**Ngày 3-7: Textual Inversion Training (song song)**
- [ ] Minh Quốc: Huấn luyện textual inversion cho 3-5 phong cách, log thời gian và VRAM
- [ ] Hy & Phát & Minh Quốc: Ghi lại thời gian train, kích thước checkpoint, GPU usage cho cả 3 phương pháp

**Ngày 6-7: Integration & Testing**
- [ ] Phát: Tích hợp inference pipeline và sinh mẫu kết quả
- [ ] Minh Quốc: Bắt đầu skeleton demo Gradio (chọn model, upload ảnh)
- [ ] Hy: Test evaluation metrics trên output hiện có
- [ ] Cả team: Kiểm thử end-to-end (content → styled image)

### Week 2: Evaluation & Demo

**Ngày 8-9: Evaluation & Analysis**
- [ ] Hy: Evaluate đầy đủ cả 3 phương pháp (LoRA, DreamBooth, Textual Inversion)
- [ ] Hy: Tính metrics: FID, LPIPS, SSIM, Content Loss, Style Loss cho từng phương pháp
- [ ] Hy: Lập bảng so sánh chi tiết: thời gian train, kích thước checkpoint, chất lượng output
- [ ] Phát: Visualization kết quả (content/style/output, loss curves) cho LoRA
- [ ] Minh Quốc: Visualization kết quả cho Textual Inversion
- [ ] Hy: Visualization kết quả cho DreamBooth
- [ ] Cả team: Fine-tune bổ sung nếu cần cải thiện chất lượng

**Ngày 10-11: Demo & Documentation**
- [ ] Minh Quốc: Hoàn thiện demo app (UI, inference, download) với 3 phương pháp
- [ ] Hy: Draft báo cáo + slide outline (bao gồm so sánh LoRA vs DreamBooth vs Textual Inversion)
- [ ] Cả team: Test demo, ghi nhận feedback
- [ ] Chuẩn bị clip demo (screen recording)

**Ngày 12-14: Finalization**
- [ ] Hy: Hoàn thiện báo cáo & evaluation report
- [ ] Minh Quốc: Chỉnh sửa demo theo feedback cuối
- [ ] Cả team: Review tổng thể, chuẩn bị presentation, final submission

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
│   ├── 00_Data_EDA.ipynb              # EDA và phân tích dữ liệu (Hy)
│   ├── 01a_LoRA_Training.ipynb         # LoRA training (Phát)
│   ├── 01b_DreamBooth_Training.ipynb   # DreamBooth training (Hy)
│   ├── 01c_Textual_Inversion_Training.ipynb  # Textual inversion (Minh Quốc)
│   ├── 02_Inference_Pipeline.ipynb    # Inference với các phương pháp (Phát)
│   ├── 03_Demo_Application.ipynb      # Giao diện demo Gradio (Minh Quốc)
│   ├── 04_Evaluation_Metrics.ipynb    # Tính FID, LPIPS, SSIM (Hy)
│   └── 05_Results_Analysis.ipynb      # Phân tích và so sánh kết quả (Hy)
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
│   ├── train_lora.py                  # Entry point huấn luyện LoRA
│   ├── train_dreambooth.py            # Entry point huấn luyện DreamBooth (nếu cần)
│   ├── train_textual_inversion.py     # Entry point textual inversion embeddings
│   ├── infer.py                       # Entry point inference (hỗ trợ cả 3 phương pháp)
│   ├── eval.py                        # Entry point evaluation
│   └── demo.py                        # Entry point demo app
│
├── docs/                              # Tài liệu chi tiết
│   ├── architecture.md                # Giải thích SD + LoRA
│   ├── training_guide.md              # Hướng dẫn training
│   └── evaluation_metrics.md         # Cách tính các chỉ số
│
└── results/                           # Kết quả mẫu
    ├── eda/                           # Kết quả EDA
    └── metrics/                       # Metrics và logs
    └── samples/                       # Output samples
```

---

## Tech Stack & Tools

### Development Environment

- **Primary**: Kaggle Notebooks (GPU: P100/T4)
- **Datasets**: Kaggle Datasets (COCO 2017, WikiArt)
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

## Baseline và Chiến Lược Đánh Giá

### Baseline

**Baseline chính**: Stable Diffusion v1.5 gốc (`runwayml/stable-diffusion-v1-5`)

- Model đã được train sẵn, download từ Hugging Face (không train từ đầu)
- Sử dụng text prompt để generate ảnh
- Không có style transfer cụ thể
- Mục đích: So sánh để chứng minh LoRA fine-tuning cải thiện chất lượng

**Baseline fine-tuning 1**: LoRA (Low-Rank Adaptation)
- Train ~4-8M parameters, checkpoint 4-8MB
- Ưu tiên lightweight, dễ triển khai nhiều style
- Training nhanh, memory efficient

**Baseline fine-tuning 2**: DreamBooth
- **Chỉ fine-tune attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
- Train ~30% parameters (~260M thay vì 860M full UNet)
- Checkpoint: Chỉ lưu attention layers đã train (nhỏ hơn full model)
- Training lâu hơn, memory usage ~5-6GB VRAM (với optimizations)
- Sử dụng prior preservation để tránh overfitting
- **Lưu ý**: Trong implementation này, không train full UNet do hạn chế phần cứng

**Baseline fine-tuning 3**: Textual Inversion
- Fine-tune embedding của token đặc biệt trong CLIP text encoder (~768 params)
- Checkpoint < 1MB, training 500-1000 steps/style, phù hợp cho Kaggle
- Rất nhẹ, training nhanh nhất

**Baseline tham khảo**: Stable Diffusion v1.5 gốc (không fine-tune)

Xem chi tiết tại: [`docs/baseline_and_evaluation.md`](docs/baseline_and_evaluation.md)

### Model Training

**Base Model**:
- Download từ Hugging Face: `runwayml/stable-diffusion-v1-5`
- **KHÔNG train từ đầu**, chỉ download và sử dụng
- Cấu trúc: VAE (~85M) + UNet (~860M) + CLIP (~123M, không dùng)

**LoRA Fine-Tuning** (Phát):
- Load base model SD v1.5
- Thêm LoRA layers vào UNet attention layers
- **CHỈ train LoRA weights** (~4-8M params), không train toàn bộ UNet
- Train trên style images từ WikiArt
- Mỗi style → 1 LoRA checkpoint (~4-8MB)

**DreamBooth Fine-Tuning** (Hy):
- Load base model SD v1.5
- **Chỉ fine-tune attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
  - Freeze tất cả parameters của UNet
  - Chỉ enable gradient cho attention layers (cross-attention và self-attention)
  - Train ~30% parameters (~260M thay vì 860M)
- Sử dụng prior preservation với class images để tránh overfitting
- Train trên instance images + class images từ WikiArt
- Memory optimizations: CPU offloading, VAE slicing, attention slicing, resolution 256
- Mỗi style → 1 DreamBooth checkpoint (chỉ lưu attention layers đã train)
- **Lưu ý**: Nếu có GPU lớn hơn (A100 40GB+), có thể train full UNet để đạt chất lượng cao hơn

**Textual Inversion Fine-Tuning** (Minh Quốc):
- Load base model SD v1.5
- Fine-tune embedding của special token trong CLIP text encoder
- Train trên style images với captions chứa special token
- Mỗi style → 1 embedding checkpoint (< 1MB)

**Hyperparameters** (tham khảo):
- LoRA: Rank=4, LR=1e-4, Batch=2-4, Steps=5k-8k
- DreamBooth: LR=5e-6, Batch=1, Steps=1k-2k, Prior loss weight=1.0
- Textual Inversion: LR=5e-4, Batch=1, Steps=500-1000

### Evaluation Strategy

**Metrics sử dụng**:
1. **FID**: Đo độ "thật" của ảnh (target: < 60)
2. **LPIPS**: Đo sự tương đồng style (target: < 0.3)
3. **SSIM**: Đo độ giữ cấu trúc content (target: > 0.7)
4. **Content Loss**: Giữ nội dung content image
5. **Style Loss**: Tái tạo phong cách style image
6. **Inference Time**: Tốc độ generate (target: < 5s/image)

**Test Set**:
- Content: 100-200 ảnh từ COCO val2017
- Style: 10-20 ảnh đại diện cho mỗi style

**So sánh với Baseline**:
- Generate outputs với baseline (SD v1.5 gốc - không fine-tune)
- Generate outputs với LoRA models
- Generate outputs với DreamBooth models
- Generate outputs với Textual Inversion embeddings
- So sánh metrics để chứng minh cải thiện và đánh giá trade-offs giữa các phương pháp

**Nguyên lý đánh giá**:
- FID thấp = ảnh giống thật hơn
- LPIPS thấp = style transfer thành công
- SSIM cao = giữ được cấu trúc content
- Cân bằng giữa content preservation và style transfer

Xem chi tiết tại: [`docs/baseline_and_evaluation.md`](docs/baseline_and_evaluation.md)

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

- [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart)

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
- [ ] DreamBooth training pipeline
- [ ] Textual inversion training pipeline
- [ ] Fine-tune 3-5 phong cách cho mỗi phương pháp
- [ ] Inference pipeline (hỗ trợ cả 3 phương pháp)
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

- **GitHub**: https://github.com/HyIsNoob/customized-image-generation
- **Issues**: Sử dụng GitHub Issues để báo lỗi và đề xuất