# Sample Images

Thư mục này chứa các ảnh được generate từ các models đã train.

## Cấu trúc

```
samples/
├── dreambooth/      # Ảnh từ DreamBooth models
│   ├── Contemporary_Realism/
│   │   ├── baseline_comparison.png
│   │   ├── test_samples/
│   │   └── parameter_optimization/
│   └── New_Realism/
│       ├── baseline_comparison.png
│       ├── test_samples/
│       └── parameter_optimization/
├── lora/            # Ảnh từ LoRA models
│   └── [style_name]/
└── comparisons/     # So sánh giữa các phương pháp
    ├── baseline_vs_dreambooth.png
    ├── baseline_vs_lora.png
    └── dreambooth_vs_lora.png
```

## Nội dung

### `dreambooth/`
- Ảnh inference từ DreamBooth models
- Comparison images (baseline vs DreamBooth)
- Test samples với các prompts khác nhau
- Parameter optimization results (guidance_scale, num_inference_steps)

### `lora/`
- Ảnh inference từ LoRA models
- Tổ chức theo style name

### `comparisons/`
- So sánh trực quan giữa các phương pháp
- Baseline vs Fine-tuned models
- DreamBooth vs LoRA vs Textual Inversion

## Cách tạo samples

### DreamBooth
Chạy notebook: `testInfer/01_DreamBooth_Inference_Test.ipynb`
- Output sẽ được lưu vào `/kaggle/working/inference_samples/` (trên Kaggle)
- Copy các ảnh quan trọng vào `samples/dreambooth/` để commit lên GitHub

### LoRA
Chạy notebook: `02_Inference_Pipeline.ipynb` (nếu có)
- Lưu samples vào `samples/lora/`

## Lưu ý

- Chỉ commit các ảnh **quan trọng** (comparisons, best samples)
- Không commit tất cả test images (quá nhiều files)
- Ưu tiên: comparison images, parameter optimization results, best samples
- Format: PNG hoặc JPG, resolution hợp lý (không quá lớn)

