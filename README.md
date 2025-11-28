# Face Recognition Dataset - README

## Dataset Information
- **Total Images**: 276
- **Total Classes (People)**: 70
- **Images per Person**: 2-8 (mostly 4)
- **Train Split**: ~75% (207 images)
- **Validation Split**: ~25% (69 images)

## Setup Instructions

### 1. Prepare Dataset
Jalankan script untuk konversi gambar dan buat CSV:
```bash
cd Tubes
python prepare_dataset.py
```

Script ini akan:
- Konversi semua file HEIC/WebP/PNG ke JPG
- Membuat `train.csv` dengan kolom: filename, label, person_name
- Membuat `label_mapping.csv` untuk mapping nama ke ID

### 2. File yang Dihasilkan
- `Dataset/train.csv`: Dataset mapping
- `Dataset/label_mapping.csv`: Label ID mapping  
- `Dataset/Train/<nama>/`: Folder gambar per orang

## Usage dengan DataReader

```python
from datareader import FaceRecognition

# Load dataset
train_dataset = FaceRecognition(
    data_dir='Tubes/Dataset/Train',
    img_size=(224, 224),
    split='train'
)

val_dataset = FaceRecognition(
    data_dir='Tubes/Dataset/Train',
    img_size=(224, 224),
    split='val'
)

# Access data
image, label, filename = train_dataset[0]
print(f"Image shape: {image.shape}")  # torch.Size([3, 224, 224])
print(f"Label: {label}")  # ID kelas (0-69)
print(f"Filename: {filename}")  # path relatif
```

## Data Augmentation

### Training
- RandomResizedCrop (scale 0.9-1.0)
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness, contrast, saturation ±10%)
- ImageNet normalization

### Validation
- Resize to 224x224
- ImageNet normalization

## Known Issues

1. **1 corrupt image**: `Martua Kevin A.M.H.Lubis/IMG_20241208_223409_653 - MARTUA KEVIN ANDREAS MUAL H LUBIS.jpg`
   - File corrupt/tidak bisa dibaca
   - Total valid: 275/276 images

2. **HEIC files**: Beberapa file HEIC tidak terkonversi karena butuh `pillow-heif`:
   ```bash
   pip install pillow-heif
   ```

## Dataset Structure
```
Tubes/Dataset/
├── train.csv                 # Main dataset file
├── label_mapping.csv         # Label to name mapping
└── Train/
    ├── Abraham Ganda Napitu/
    │   ├── IMG_4951 - Abraham Ganda Napitu.jpeg
    │   ├── IMG_5357 - Abraham Ganda Napitu.jpeg
    │   └── ...
    ├── Ahmad Faqih Hasani/
    │   └── ...
    └── .../
```

## Class Distribution
- Mean: 4.0 images/class
- Std: 0.59
- Min: 2 images/class
- Max: 8 images/class

## Notes for Training

1. **Small dataset**: Gunakan heavy augmentation atau pre-trained models
2. **Class imbalance**: Minimal tapi ada (2-8 gambar per kelas)
3. **Stratified split**: Sudah dihandle otomatis, minimal 1 gambar per kelas di validation
4. **ImageNet normalization**: Cocok untuk transfer learning dari pretrained ViT/ResNet
