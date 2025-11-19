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
3. **Nguyễn Minh Quốc** (23521304)

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
- Ảnh sinh ra giữ bố cục content (SSIM > baseline) và thể hiện style (LPIPS vừa phải)
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
- Training nhanh: < 6 giờ thay vì vài ngày (thực tế: ~5-6 giờ cho 1 style)
- Checkpoint nhỏ: ~4-8MB mỗi style (thay vì 3-4GB)
- Tiết kiệm GPU memory: Có thể train trên GPU nhỏ hơn (T4, P100)
- Dễ quản lý: Mỗi style 1 file LoRA nhỏ, dễ switch giữa các styles

**So sánh**:

| Phương pháp | Parameters | Checkpoint Size | Training Time | GPU Memory |
|-------------|-----------|----------------|---------------|------------|
| **Full Fine-tune** | 860M | ~3-4GB | Vài ngày | ~24GB |
| **LoRA (r=4)** | ~4-8M | ~1-2GB | < 6 giờ | ~12GB |
| **DreamBooth (attention-only)** | ~260M (30% UNet) | ~3-4GB | ~12 giờ | ~5-6GB |

**Kết luận**:

- SD: Model mạnh, đã được train sẵn, có khả năng generate ảnh tốt
- LoRA: Cách hiệu quả nhất để adapt SD cho style cụ thể - training nhanh nhất (< 6h) với ít parameters nhất (~4-8M)
- DreamBooth: Training chậm hơn (~12h) dù chỉ train 30% parameters do phải load toàn bộ model và xử lý prior preservation
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
- Các phong cách: Contemporary_Realism, New_Realism, Synthetic_Cubism, Analytical_Cubism, Action_painting

### 2a. Fine-tune LoRA

**Cấu hình**:

- Base model: `runwayml/stable-diffusion-v1-5`
- Fine-tune target: UNet attention layers
- Rank: 4
- Learning rate: 1e-4
- Batch size: 2
- Steps: 1,500/phong cách
- Optimizer: AdamW
- Scheduler: Cosine

**Loss Function**:
```
L_total = α·L2 + β·LPIPS + γ·StyleLoss
```

- L2 loss: Tái tạo chi tiết ảnh
- LPIPS: Duy trì độ tự nhiên theo cảm nhận người nhìn
- Style loss (Gram matrix): Giữ họa tiết, màu sắc của style

### 2b. Fine-tune DreamBooth

**Mục tiêu**: Fine-tune UNet với prior preservation để học phong cách nghệ thuật cụ thể. Do hạn chế về phần cứng (GPU memory trên Kaggle), chúng em chỉ fine-tune **attention layers** của UNet thay vì toàn bộ UNet.

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
- Input size: 256 (giảm từ 512 để tiết kiệm memory)
- Learning rate: 1e-5
- Batch size: 1 (với gradient accumulation 16)
- Steps: 2k per style
- Optimizer: AdamW
- Loss: MSE loss + Prior preservation loss (weight=1.0)

**Memory optimizations** (bắt buộc do hạn chế phần cứng):

- **CPU offloading**: VAE và Text Encoder ở CPU, chỉ move lên GPU khi encode
- **VAE slicing và tiling**: Chia VAE encoding thành các slice/tile nhỏ hơn
- **Attention slicing**: Chia attention mechanism thành các slice
- **Gradient checkpointing**: Trade computation for memory
- **Resolution reduction**: 512 → 256 để giảm memory cho activations
- **Gradient accumulation**: Batch size 1 với accumulation 16 để mô phỏng batch lớn hơn

**Kết quả và Hạn chế**:

- Checkpoint: Chỉ lưu attention layers đã train (~260M parameters), có thể load vào base model
- Memory usage: ~5-6GB VRAM (thay vì ~15GB nếu train full UNet)
- Chất lượng: Style transfer hoạt động nhưng chưa mạnh như full UNet training

**Tại sao kết quả chưa tối ưu?**:

1. **Chỉ train attention layers (~30% parameters)**:
   - Attention layers: Điều khiển "what to attend to" (nội dung, style concept)
   - ResNet blocks: Điều khiển "how to process" (texture, brushstrokes, rendering details)
   - **Hệ quả**: Model học được style concept nhưng thiếu texture/brushstroke details
   
2. **Hạn chế phần cứng**:
   - Kaggle GPU: T4/P100 với 16GB VRAM
   - Full UNet training cần ~20-24GB VRAM (model + optimizer state + activations)
   - Không thể train full UNet → phải chấp nhận trade-off
   
3. **Hạn chế thời gian**:
   - Kaggle timeout: 12 giờ/session
   - Training 1 style: ~12 giờ (với attention-only, ~30% parameters)
   - **Lý do chậm hơn LoRA**: Dù chỉ train 30% parameters, DreamBooth vẫn phải:
     - Load toàn bộ UNet vào GPU (không chỉ attention layers)
     - Tính forward/backward qua toàn bộ UNet (chỉ update attention)
     - Xử lý prior preservation loss (class images) → tăng computation
     - Memory overhead cao hơn do phải giữ toàn bộ model
   - Không thể train lại nhiều lần để tối ưu hyperparameters
   - Kaggle weekly quota: Giới hạn số lần chạy GPU/week
   
4. **Resolution thấp**:
   - Input: 256×256 (thay vì 512×512) để tiết kiệm memory
   - Mất chi tiết texture và brushstrokes ở resolution thấp

### 2c. Fine-tune Textual Inversion

**Mục tiêu**: Học một embedding mới trong CLIP text encoder đại diện cho phong cách (`sks style`) thay vì fine-tune toàn bộ UNet.

**Cấu hình**:

- Base model: `runwayml/stable-diffusion-v1-5`
- Modules train: Textual embedding (768 chiều) dành cho token mới
- Learning rate: 5e-5
- Batch size: 1 (gradient accumulation 4)
- Steps: 400 per style
- Optimizer: AdamW
- Scheduler: Cosine/Constant

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
   - Fine-tune DreamBooth cho 2 phong cách nghệ thuật
   - **Chỉ train attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
   - Tối ưu memory cho Kaggle GPU (CPU offloading, VAE slicing, attention slicing, resolution reduction)
   - Implement freeze/unfreeze logic để chỉ train attention layers
   - Hyperparameter tuning (learning rate, prior loss weight, steps)
   - Ghi nhận thời gian train, kích thước checkpoint, GPU usage
   - Save/load DreamBooth checkpoints (chỉ attention layers)

3. **Evaluation Framework**:
   - CLIP-Based: CLIP-Content, CLIP-Style, Style Strength.
   - Load style reference images từ WikiArt
   - Inference time benchmark
   - Create test suite với diverse samples
   - So sánh LoRA vs DreamBooth vs Textual Inversion

4. **Results & Reporting**:
   - Tổng hợp kết quả training từ cả 3 phương pháp
   - So sánh các phong cách và các phương pháp fine-tuning
   - Viết báo cáo cuối kỳ

**Deliverables**:

- Notebook: `00-Data-EDA.ipynb`
- Notebook: `01b_DreamBooth_Training.ipynb`
- Notebook: `04a_Evaluation_Metrics_LoRA.ipynb`
- Notebook: `04b_Evaluation_Metrics_DreamBooth_TI.ipynb`
- Notebook: `05_Results_Analysis.ipynb`
- Trained DreamBooth checkpoints (2 styles)
- Evaluation report với 4 metrics
- Slide

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
- Notebook: `testInfer/01-LoRA-Inference-Test.ipynb`
- Trained LoRA checkpoints (5 styles)
- Training logs và metrics
- Thuyết trình

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
- Notebook: `testInfer/03-Textual-Inversion-Inference-Test.ipynb`
- Notebook: `03_Demo_Application.ipynb`
- Textual inversion embedding checkpoints (1 style)
- Demo app (Gradio) + video/screenshots
- Textual inversion embedding checkpoints 

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
│   ├── 00-Data-EDA.ipynb              # EDA và phân tích dữ liệu (Hy)
│   ├── 01a_LoRA_Training.ipynb         # LoRA training (Phát)
│   ├── 01b_DreamBooth_Training.ipynb   # DreamBooth training (Hy)
│   ├── 01c_Textual_Inversion_Training.ipynb  # Textual inversion (Minh Quốc)
│   ├── 04a_Evaluation_Metrics_LoRA.ipynb    # Đánh giá LoRA (Hy)
│   ├── 04b_Evaluation_Metrics_DreamBooth_TI.ipynb  # Đánh giá DreamBooth + TI (Hy)
│   └── 05_Results_Analysis.ipynb      # Phân tích và so sánh kết quả (Hy)
│
├── testInfer/                         # Inference test notebooks
│   ├── 01-LoRA-Inference-Test.ipynb    # Test inference LoRA
│   ├── 02-Dreambooth-Inference-Test.ipynb  # Test inference DreamBooth
│   └── 03-Textual-Inversion-Inference-Test.ipynb  # Test inference TI
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
- Checkpoint < 1MB, training 400 steps/style, phù hợp cho Kaggle
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

**Textual Inversion Fine-Tuning** (Minh Quốc):

- Load base model SD v1.5
- Fine-tune embedding của special token trong CLIP text encoder
- Train trên style images với captions chứa special token
- Mỗi style → 1 embedding checkpoint (< 1MB)

**Hyperparameters** (tham khảo):

- LoRA: Rank=4, LR=1e-4, Batch=2, Steps=1.5k
- DreamBooth: LR=1e-5, Batch=1, Steps=2k, Prior loss weight=0.6
- Textual Inversion: LR=5e-5, Batch=1, Steps=400

### Evaluation Strategy

**Metrics sử dụng**:

- **CLIP-content**: `1 - cos_sim(clip(output), clip(content))`.  Thấp hơn → output giữ semantic content tốt hơn (so với ảnh gốc).
- **Style Strength Score**: `clip_content / baseline_clip_content`. ≈1 nghĩa là áp style tương đương baseline, >1 nghĩa là áp style mạnh hơn (hy sinh content nhiều hơn).
- **CLIP-style**: `1 - cos_sim(clip(output), clip(style_reference))`. Thấp hơn → output giống style reference hơn theo CLIP.

**Additional Metrics**:

- **Inference Time**: < 5s/ảnh trên Kaggle P100/T4.

**Test Set**:

- **Content**: Tập con COCO val2017 (8 ảnh resized 256×256). Danh sách ảnh được cố định và chia sẻ giữa mọi notebook qua `content_paths.json`.
- **Style**: WikiArt images (10 ảnh/style). Danh sách ảnh cố định qua `style_paths.json` để LoRA và DreamBooth/TI dùng chung baseline.

**So sánh với Baseline**:

- Baseline: `runwayml/stable-diffusion-v1-5` chạy img2img cùng content images (dùng để chuẩn hóa Style Strength Score).
- DreamBooth: Contemporary_Realism, New_Realism (2 styles).
- LoRA: Action_painting, Analytical_Cubism, Contemporary_Realism, New_Realism, Synthetic_Cubism (5 styles).
- Textual Inversion: sks_style (1 style).

**Nguyên lý đánh giá**:

- **Content Preservation**: CLIP-content của model càng sát baseline càng tốt; Style Strength ≈1 nghĩa là độ thay đổi vừa đủ.
- **Style Quality**: CLIP-style giảm ⇒ mô hình áp đúng style, không cần phụ thuộc texture low-level.
- **Trade-off**: DreamBooth thường có Style Strength cao (áp style mạnh, khả năng mất content cao) trong khi LoRA/TI giữ content tốt hơn. Bộ CLIP metrics thể hiện trade-off này rõ ràng và nhất quán. LoRA vừa áp style tốt và giữ content tốt nhất, Cân bằng nhất trong cả 3 model.

---

## Evaluation Metrics

### Target Metrics

- **CLIP-content** gần với baseline (≤ baseline + 0.05) để đảm bảo giữ nội dung.
- **Style Strength Score** quanh 1 cho kết quả cân bằng; >1 biểu thị áp style mạnh hơn baseline (chấp nhận được nếu CLIP-style thấp).
- **CLIP-style** < 0.5 cho chất lượng style tốt (ngưỡng thực nghiệm).
- **Inference time < 5s/ảnh** với 256×256 trên Kaggle P100/T4.

### Lưu ý về Trade-off

- **LoRA**: Thường có Style Strength >1 (apply style mạnh) kèm CLIP-style thấp ⇒ phù hợp nếu ưu tiên style đồng thời cũng có CLIP-content thấp, tốt nhất để style transfer.
- **DreamBooth**: CLIP-Content cao (giữ content tệ).CLIP-style trung bình nhưng Style Strength lại cao cho thấy áp style mạnh vừa có khả năng mất nội dung nhưng style áp chỉ vừa đạt với style Reference.
- **Textual Inversion**: CLIP-content trung bình và CLIP-Style trung bình, phù hợp nhanh gọn nhẹ nhưng không quá tốt.
- Bộ metrics CLIP cho phép đánh giá đồng nhất giữa các notebook vì baseline và tập ảnh đã được cố định.

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

## Liên Hệ

- **GitHub**: https://github.com/HyIsNoob/customized-image-generation
- **Issues**: Sử dụng GitHub Issues để báo lỗi và đề xuất