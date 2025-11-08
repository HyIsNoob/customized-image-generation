
# Data EDA Report

## COCO Dataset
- Train images: 118287 (dùng cho training)
- Val images: 5000 (dùng cho validation/testing)
- Mean size: 576x486px
- Mean aspect ratio: 1.25

## WikiArt Dataset
- Total styles: 27
- Selected styles for training: 5
- Styles: Contemporary_Realism, New_Realism, Synthetic_Cubism, Analytical_Cubism, Action_painting
- Image count per style: 98-481 images

## Recommendations
- Resize all images to 512x512 for training
- Use train2017 (118287 images) cho training LoRA
- Use val2017 (5000 images) cho validation/testing sau khi train
- Use selected styles with 98-481 images each (total: 1219 style images)
- Consider data augmentation for style images if needed
