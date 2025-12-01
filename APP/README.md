# Face Recognition App - FaceNet

Aplikasi web untuk face recognition menggunakan FaceNet (InceptionResnetV1) dengan Gradio interface.

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source ../../DeepLearn/bin/activate

# 2. Run the app
python run_with_facenet.py

# 3. Open browser
# http://127.0.0.1:7860
```

## 📋 Features

- ✅ **FaceNet Model**: InceptionResnetV1 pretrained on VGGFace2
- ✅ **70 Persons**: Mengenali 70 orang berbeda
- ✅ **MTCNN Detection**: Automatic face detection & alignment
- ✅ **Web Interface**: Gradio-based UI
- ✅ **Real-time**: Webcam support
- ✅ **GPU Accelerated**: CUDA support

## 📁 Structure

```
APP/
├── facenet_adapter.py       # FaceNet model adapter
├── run_with_facenet.py      # Launcher script
├── config/
│   └── person_mapping.json  # Label → Name mapping (70 persons)
├── requirements.txt         # Dependencies
├── README.md               # This file
└── README_FACENET.md       # Detailed documentation
```

## ⚙️ Options

```bash
# Custom model path
python run_with_facenet.py --model-path /path/to/model.pth

# Share publicly
python run_with_facenet.py --share

# Custom port
python run_with_facenet.py --server-port 8080
```

## 📖 Full Documentation

See [README_FACENET.md](README_FACENET.md) for:
- Detailed usage guide
- Troubleshooting
- Advanced features
- API examples

## 📊 Model Info

- **Architecture**: InceptionResnetV1 + Classifier
- **Pretrained**: VGGFace2
- **Input**: 224×224 RGB
- **Classes**: 70 persons
- **Model**: `../best_facenet_model.pth` (110 MB)

---

**Ready to use! ��**
