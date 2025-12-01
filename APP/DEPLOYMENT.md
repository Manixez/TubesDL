# FaceNet Face Recognition - Deployment Guide

## 📦 Model Deployment

Model `.pth` file **TIDAK** disimpan di Git karena ukurannya besar (110+ MB).

### Cara Deploy:

#### Option 1: Download Model Terpisah
```bash
# 1. Clone repository
git clone <your-repo>
cd Tubes/APP

# 2. Download model dari cloud storage (Google Drive, S3, dll)
wget https://your-storage.com/best_facenet_model.pth
# atau
curl -o best_facenet_model.pth https://your-storage.com/model.pth

# 3. Jalankan aplikasi
python run_with_facenet.py
```

#### Option 2: Environment Variable
```bash
# 1. Upload model ke server
scp best_facenet_model.pth user@server:/models/

# 2. Set environment variable
export FACENET_MODEL_PATH=/models/best_facenet_model.pth

# 3. Jalankan aplikasi
python run_with_facenet.py
```

#### Option 3: Docker Volume Mount
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY APP/ /app/
RUN pip install -r requirements.txt

# Model will be mounted as volume
ENV FACENET_MODEL_PATH=/models/best_facenet_model.pth

CMD ["python", "run_with_facenet.py"]
```

```bash
# Run dengan volume mount
docker run -v /path/to/model:/models/best_facenet_model.pth \
           -p 7860:7860 \
           facenet-app
```

## 📍 Model Path Priority

Script mencari model dengan prioritas:
1. **CLI argument**: `--model-path /path/to/model.pth`
2. **Environment variable**: `FACENET_MODEL_PATH`
3. **Auto-detect** di lokasi berikut:
   - `Tubes/best_facenet_model.pth`
   - `APP/best_facenet_model.pth`
   - `./best_facenet_model.pth` (current directory)
   - `models/best_facenet_model.pth`

## 🚀 Production Setup

### Rekomendasi untuk Production:

1. **Upload model ke cloud storage**:
   - Google Cloud Storage
   - AWS S3
   - Azure Blob Storage
   
2. **Download saat deployment**:
   ```bash
   # Contoh dengan gsutil (Google Cloud)
   gsutil cp gs://your-bucket/best_facenet_model.pth APP/
   
   # Contoh dengan aws cli
   aws s3 cp s3://your-bucket/best_facenet_model.pth APP/
   ```

3. **Atau gunakan environment variable**:
   ```bash
   export FACENET_MODEL_PATH=/mnt/storage/models/facenet.pth
   ```

## 🔒 Security

- **JANGAN** commit file `.pth` ke Git
- Gunakan `.gitignore` untuk exclude model files
- Store model di private storage dengan access control
- Gunakan signed URLs untuk download model

## 📊 Model Info

- **File**: `best_facenet_model.pth`
- **Size**: ~110 MB
- **Architecture**: InceptionResnetV1 (VGGFace2)
- **Classes**: 70 persons
- **Input**: 160x160 RGB images

## 🛠️ Development vs Production

### Development:
```bash
# Model ada lokal di Tubes/
python train_facenet.py  # Train model
python APP/run_with_facenet.py  # Auto-detect model
```

### Production:
```bash
# Download model dari storage
wget $MODEL_URL -O APP/best_facenet_model.pth

# Atau set env variable
export FACENET_MODEL_PATH=/path/to/model.pth

# Run app
python APP/run_with_facenet.py
```

## 🐳 Docker Example

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY APP/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY APP/ .

# Create models directory
RUN mkdir -p /models

# Set environment
ENV FACENET_MODEL_PATH=/models/best_facenet_model.pth

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run app
CMD ["python", "run_with_facenet.py", "--server-name", "0.0.0.0"]
```

Run:
```bash
docker build -t facenet-app .
docker run -d \
  -p 7860:7860 \
  -v /path/to/model.pth:/models/best_facenet_model.pth:ro \
  facenet-app
```
