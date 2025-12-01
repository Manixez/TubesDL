# FaceNet Face Recognition App

Aplikasi web untuk face recognition menggunakan model FaceNet yang sudah di-train.

## 📋 Fitur

- **MTCNN Face Detection**: Deteksi wajah otomatis dengan alignment
- **InceptionResnetV1**: Pre-trained di VGGFace2 untuk face recognition
- **70 Persons**: Mengenali 70 orang yang berbeda
- **Web Interface**: Gradio-based interface yang mudah digunakan
- **Real-time Recognition**: Deteksi dan recognisi wajah secara real-time

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Pastikan virtual environment aktif
source ../DeepLearn/bin/activate

# Install requirements (sudah diinstall)
pip install gradio opencv-python face-recognition torch torchvision facenet-pytorch
```

### 2. Jalankan Aplikasi

```bash
# Basic usage
python run_with_facenet.py

# Dengan opsi kustom
python run_with_facenet.py --model-path ../best_facenet_model.pth --num-classes 70

# Share ke public (Gradio link)
python run_with_facenet.py --share

# Buka untuk semua network interfaces
python run_with_facenet.py --server-name 0.0.0.0 --server-port 7860
```

### 3. Akses Web Interface

Setelah running, buka browser ke:
```
http://127.0.0.1:7860
```

## 📁 Struktur File

```
APP/
├── facenet_adapter.py          # FaceNet model adapter
├── run_with_facenet.py         # Launcher script
├── config/
│   └── person_mapping.json     # Mapping label → person name
├── app.py                      # Base Gradio app (jika ada)
└── README_FACENET.md           # File ini
```

## 🔧 Komponen Utama

### FaceNetAdapter

Adapter untuk integrasi FaceNet model dengan Gradio app:

- `load_model()`: Load model weights dari `.pth` file
- `load_person_mapping()`: Load mapping label → person name
- `preprocess_image()`: Resize ke 160x160, normalisasi
- `predict_from_image()`: Prediksi dari cropped face image
- Returns: `(person_name, confidence_score)`

### FaceNetApp

Custom Gradio app untuk FaceNet:

- `process_frame()`: Deteksi wajah → extract face → predict → draw boxes
- Menampilkan nama dan confidence score untuk setiap wajah
- Warna hijau untuk wajah yang dikenali, merah untuk unknown

## 📊 Model Information

- **Architecture**: InceptionResnetV1 + Custom Classifier
- **Pre-trained**: VGGFace2 dataset
- **Input Size**: 160×160 RGB images
- **Output**: 70 classes (70 persons)
- **Total Parameters**: 28.5M (27.9M frozen, 645K trainable)
- **Model File**: `../best_facenet_model.pth` (110 MB)

## 🎯 Person Mapping

File `config/person_mapping.json` berisi mapping dari label ID ke nama orang:

```json
{
  "0": "Nasya Aulia Efendi",
  "1": "Abraham Ganda Napitu",
  "2": "Abu Bakar Siddiq Siregar",
  ...
  "69": "..."
}
```

Total: **70 orang**

## 💡 Tips Penggunaan

### Upload Image

1. Klik "Upload Image" di web interface
2. Pilih foto yang mengandung wajah
3. Sistem akan otomatis:
   - Deteksi semua wajah di foto
   - Kenali setiap wajah
   - Tampilkan nama dan confidence score

### Webcam Real-time

1. Klik "Use Webcam"
2. Izinkan akses webcam
3. Sistem akan real-time:
   - Deteksi wajah di video stream
   - Kenali setiap wajah
   - Draw bounding box dengan nama

### Confidence Threshold

Default threshold: **0.5** (50%)

Untuk mengubah:
```python
app.model_adapter.confidence_threshold = 0.7  # 70%
```

Jika confidence < threshold → "Unknown"

## 🐛 Troubleshooting

### Model Not Found

```
❌ Model file not found: ../best_facenet_model.pth
```

**Solusi**: Train model dulu dengan `train_facenet.py`

```bash
cd ..
python train_facenet.py
```

### Import Error

```
ModuleNotFoundError: No module named 'facenet_pytorch'
```

**Solusi**: Install facenet-pytorch

```bash
pip install facenet-pytorch
```

### CUDA Out of Memory

**Solusi**: Model otomatis deteksi GPU/CPU. Jika OOM, model akan fallback ke CPU.

Atau force CPU:
```python
# Di facenet_adapter.py, line ~30
self.device = torch.device('cpu')  # Force CPU
```

### Person Mapping Not Found

```
⚠️ Using default person mapping (Person_0, Person_1, ...)
```

**Solusi**: File `config/person_mapping.json` missing. Re-generate:

```bash
python -c "
from pathlib import Path
import json

csv_file = Path('../Dataset/label_mapping.csv')
mapping = {}
with open(csv_file, 'r') as f:
    next(f)
    for line in f:
        name, label = line.strip().split(',')
        mapping[label] = name

Path('config').mkdir(exist_ok=True)
with open('config/person_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)
"
```

## 📝 Advanced Usage

### Custom Confidence Threshold

```python
from facenet_adapter import FaceNetApp

app = FaceNetApp(
    model_path="../best_facenet_model.pth",
    num_classes=70
)

# Set custom threshold
app.model_adapter.confidence_threshold = 0.7

# Launch
app.launch()
```

### Batch Prediction

```python
from facenet_adapter import FaceNetAdapter
import numpy as np
from PIL import Image

# Load adapter
adapter = FaceNetAdapter("../best_facenet_model.pth", num_classes=70)

# Load images
faces = [
    np.array(Image.open("face1.jpg")),
    np.array(Image.open("face2.jpg")),
]

# Predict
for face in faces:
    name, conf = adapter.predict_from_image(face)
    print(f"{name}: {conf:.2%}")
```

### Get Face Embeddings

```python
# Get 512-dim embeddings untuk face verification
tensor = adapter.preprocess_image(face_image)
embedding = adapter.model.get_embedding(tensor)
print(embedding.shape)  # [1, 512]
```

## 📚 References

- **FaceNet Paper**: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- **InceptionResnetV1**: [Szegedy et al., 2016](https://arxiv.org/abs/1602.07261)
- **MTCNN**: [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)
- **facenet-pytorch**: [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- **Gradio**: [gradio.app](https://gradio.app)

## 📞 Support

Jika ada masalah, cek:

1. ✅ Model file exists: `best_facenet_model.pth`
2. ✅ Person mapping exists: `config/person_mapping.json`
3. ✅ Dependencies installed: `gradio`, `torch`, `facenet-pytorch`
4. ✅ Virtual environment activated

---

**Happy Face Recognition! 🎉**
