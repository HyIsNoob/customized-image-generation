# DreamBooth Trained Models

## Tổng quan

Thư mục này chứa thông tin về các DreamBooth models đã được train cho 2 phong cách nghệ thuật:
- **Contemporary_Realism**
- **New_Realism**

## Cấu trúc Files

### Training Info
- `Contemporary_Realism_training_info.json`: Hyperparameters và thông tin training cho Contemporary_Realism
- `New_Realism_training_info.json`: Hyperparameters và thông tin training cho New_Realism

### Model Configs
- `configs/`: Chứa các config files (scheduler, UNet) cho từng style

### Model Index
- `Contemporary_Realism_model_index.json`: Cấu trúc model components
- `New_Realism_model_index.json`: Cấu trúc model components

## Models Lớn (Không có trong repo)

Các model checkpoints đầy đủ (~13GB mỗi style) được lưu tại:
- **Local**: `.myfile/results/archive/dreambooth_checkpoints/`
- **Kaggle Dataset**: Upload lên Kaggle dataset để sử dụng trong inference

### Tại sao không up lên GitHub?
- Mỗi checkpoint ~13GB (tổng ~26GB cho 2 styles)
- GitHub có giới hạn file size (100MB) và repo size
- Models có thể tải lại từ Kaggle dataset khi cần

### Cách sử dụng models:
1. Download từ Kaggle dataset (nếu đã upload)
2. Hoặc train lại từ notebook `01b_DreamBooth_Training.ipynb`
3. Load checkpoint vào inference pipeline (xem `testInfer/01_DreamBooth_Inference_Test.ipynb`)

## Training Details

### Contemporary_Realism
- **Steps**: 2000
- **Learning Rate**: 1e-5
- **Batch Size**: 1 (gradient accumulation: 16)
- **Resolution**: 256×256
- **Instance Images**: 40
- **Class Images**: 200
- **Checkpoint Size**: ~13.4 GB
- **Training Time**: ~4-6 giờ trên Kaggle T4/P100

### New_Realism
- **Steps**: 2000
- **Learning Rate**: 1e-5
- **Batch Size**: 1 (gradient accumulation: 16)
- **Resolution**: 256×256
- **Instance Images**: 40
- **Class Images**: 200
- **Checkpoint Size**: ~13.4 GB
- **Training Time**: ~4-6 giờ trên Kaggle T4/P100

## Lưu ý

- Models chỉ train **attention layers** của UNet (~30% parameters) do hạn chế GPU memory
- Xem README.md chính để biết thêm về limitations và trade-offs
- Inference test có sẵn tại `testInfer/01_DreamBooth_Inference_Test.ipynb`

