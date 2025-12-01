"""
Training FaceNet dengan Augmentasi yang Lebih Baik + WeightedRandomSampler
Berdasarkan saran: fokus pada layering augmentasi untuk meningkatkan performa
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from collections import Counter
from tqdm import tqdm
import json

# Suppress sklearn warnings tentang jumlah classes vs samples
warnings.filterwarnings('ignore', message='.*could represent a regression problem.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import datareader yang sudah ada
from datareader import FaceRecognition

# Import model FaceNet yang sudah ada
from facenet_pytorch import InceptionResnetV1

# ===========================================
# 1) KONFIGURASI
# ===========================================
DATASET_DIR = "Dataset/Train_Cropped"
MODEL_SAVE_PATH = "best_facenet_model.pth"
LOG_FILE = "train_log_facenet.txt"
IMG_SIZE = 160  # FaceNet menggunakan 160x160
BATCH_SIZE = 24
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📊 Augmentasi yang Lebih Baik + Class Weights")

# ===========================================
# 2) LOAD DATA MENGGUNAKAN DATAREADER
# ===========================================
# Augmentasi untuk training (disesuaikan dengan FaceNet 160x160)
train_tfms = T.Compose([
    # Crop dengan variasi tapi tidak terlalu ekstrim
    T.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.85, 1.00),   # Dinaikin sedikit, biar muka tidak terlalu kepotong
        ratio=(0.90, 1.10)
    ),
    
    T.RandomHorizontalFlip(p=0.5),
    
    # Warna: masih cukup kuat, tapi hue & saturation diturunin
    T.RandomApply([
        T.ColorJitter(
            brightness=0.20,   # Lebih halus
            contrast=0.20,
            saturation=0.20,
            hue=0.03           # Dikurangin dari 0.05
        )
    ], p=0.7),
    
    # Grayscale kecil
    T.RandomGrayscale(p=0.05),
    
    # Affine: rotasi/geser/zoom lebih kalem
    T.RandomAffine(
        degrees=8,               # Dikurangin dari 10
        translate=(0.04, 0.04),
        scale=(0.92, 1.08),
        shear=4
    ),
    
    # Perspektif kecil untuk variasi
    T.RandomApply([
        T.RandomPerspective(distortion_scale=0.15, p=1.0)
    ], p=0.3),
    
    # Blur ringan
    T.RandomApply([
        T.GaussianBlur(kernel_size=3)
    ], p=0.25),
    
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
    
    # Random erasing: area sedikit dipersempit
    T.RandomErasing(
        p=0.30,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    ),
])

# Validation transforms (simple)
val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

# Load dataset menggunakan FaceRecognition dari datareader.py
train_set = FaceRecognition(
    data_dir=DATASET_DIR,
    img_size=(IMG_SIZE, IMG_SIZE),
    transform=train_tfms,
    split='train'
)

val_set = FaceRecognition(
    data_dir=DATASET_DIR,
    img_size=(IMG_SIZE, IMG_SIZE),
    transform=val_tfms,
    split='val'
)

NUM_CLASSES = len(set(train_set.labels))  # Jumlah kelas unik

print(f"📂 Train: {len(train_set)} | Val: {len(val_set)} | Classes: {NUM_CLASSES}")

# ===========================================
# 3) DATALOADER (Sederhana, tanpa WeightedRandomSampler)
# ===========================================
# Hitung class distribution dari train_set.labels
counts = Counter(train_set.labels)
print(f"📊 Class distribution: {len(counts)} classes")
print(f"   Min samples per class: {min(counts.values())}")
print(f"   Max samples per class: {max(counts.values())}")
print(f"   Avg samples per class: {sum(counts.values())/len(counts):.1f}")

# Hitung class weights untuk CrossEntropyLoss
# Weight lebih tinggi untuk kelas yang jarang
# PENTING: Harus sesuai dengan NUM_CLASSES (69), bukan 512
class_weights = []
for class_id in range(NUM_CLASSES):
    if class_id in counts:
        class_weights.append(1.0 / counts[class_id])
    else:
        # Kelas tanpa sample (tidak seharusnya terjadi setelah remapping)
        class_weights.append(1.0)

class_weights = torch.FloatTensor(class_weights)
# Normalisasi weights agar sum = NUM_CLASSES
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

print(f"✅ Class weights computed: {len(class_weights)} weights (range: {class_weights.min():.2f} - {class_weights.max():.2f})")

# Simple DataLoader tanpa sampling tricks
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Simple shuffle
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"✅ DataLoader ready | train={len(train_set)} | val={len(val_set)} | classes={NUM_CLASSES}")

# ===========================================
# 4) MODEL - FACENET
# ===========================================
class FaceNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained='vggface2'):
        super().__init__()
        # Load pretrained FaceNet
        self.facenet = InceptionResnetV1(pretrained=pretrained)
        
        # Replace classifier
        self.facenet.logits = nn.Linear(512, num_classes)
        self.facenet.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.facenet(x)

model = FaceNetClassifier(NUM_CLASSES, pretrained='vggface2').to(DEVICE)
print(f"✅ Model FaceNet loaded (pretrained on VGGFace2)")

# ===========================================
# 5) LOSS & OPTIMIZER
# ===========================================
# Untuk sementara pakai CrossEntropyLoss tanpa weights
# (Augmentasi sudah cukup kuat untuk handle imbalance)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"✅ Loss: CrossEntropyLoss | Optimizer: Adam (lr={LEARNING_RATE})")

# Learning rate scheduler - Cosine Annealing dengan Warm Restarts
# Lebih agresif dan cocok untuk dataset kecil
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # Restart setiap 10 epoch
    T_mult=2,      # Setiap restart, periode dikali 2 (10, 20, 40, ...)
    eta_min=1e-6   # LR minimum
)

print(f"✅ Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

# ===========================================
# 6) TRAINING LOOP
# ===========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_data in pbar:
        # FaceRecognition returns (image, label, filename)
        images, labels = batch_data[0], batch_data[1]
        
        # Skip batch if ada label error (-1)
        if (labels == -1).any():
            continue
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100. * correct / max(total, 1)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validation"):
            # FaceRecognition returns (image, label, filename)
            images, labels = batch_data[0], batch_data[1]
            
            # Skip batch if ada label error (-1)
            if (labels == -1).any():
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100. * correct / max(total, 1)
    return epoch_loss, epoch_acc

# ===========================================
# 7) MAIN TRAINING
# ===========================================
best_val_acc = 0.0
train_history = []

with open(LOG_FILE, 'w') as log:
    log.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,LR\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Learning rate - update scheduler (step setiap epoch)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # CosineAnnealingWarmRestarts di-step setiap epoch
        
        # Log
        log_line = f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n"
        log.write(log_line)
        log.flush()
        
        # Print summary
        print(f"\n📊 Summary Epoch {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': NUM_CLASSES,
            }, MODEL_SAVE_PATH)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # History
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })

print(f"\n{'='*60}")
print(f"🎉 Training Complete!")
print(f"📈 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
print(f"📝 Log saved to: {LOG_FILE}")
print(f"{'='*60}")

# Save training history
with open('train_history_facenet.json', 'w') as f:
    json.dump(train_history, f, indent=2)
print(f"📊 Training history saved to: train_history_facenet.json")
