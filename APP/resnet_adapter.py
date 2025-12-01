"""
ResNet Adapter untuk Face Recognition App
Mengintegrasikan ResNet model dengan Gradio interface
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
import sys
import gradio as gr
from datetime import datetime
import pandas as pd
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ResNet model
from model_resnet import ResNet34Model
from app import FaceRecognitionApp


class ResNetAdapter:
    """Adapter untuk ResNet model"""
    
    def __init__(self, model_path: str, num_classes: int = 70):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load label mapping
        self.idx_to_name = self.load_label_mapping()
        
        # Load model
        print(f"📦 Loading ResNet model from: {model_path}")
        self.model = ResNet34Model(num_classes=num_classes, pretrained=False, device=self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, "
                  f"Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded")
        
        self.model.eval()
        
        # Transform untuk preprocessing (sama seperti saat training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"🔧 Device: {self.device}")
        print(f"👥 Number of classes: {num_classes}")
        print(f"📝 Label mapping: {len(self.idx_to_name)} names loaded")
    
    def load_label_mapping(self) -> dict:
        """Load label mapping dari CSV atau JSON"""
        # Coba load dari beberapa lokasi
        possible_paths = [
            Path("../Dataset/label_mapping.csv"),  # Dari APP/
            Path("Dataset/label_mapping.csv"),     # Dari Tubes/
            Path("config/person_mapping.json"),    # JSON mapping
            Path("../config/person_mapping.json"), # JSON dari APP/
        ]
        
        idx_to_name = {}
        
        # Try CSV first
        for csv_path in [p for p in possible_paths if p.suffix == '.csv']:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    # Detect column names
                    if 'label_id' in df.columns:
                        label_col = 'label_id'
                    elif 'label' in df.columns:
                        label_col = 'label'
                    else:
                        continue
                    
                    # Create mapping
                    for _, row in df.iterrows():
                        idx_to_name[int(row[label_col])] = row['person_name']
                    
                    print(f"✅ Label mapping loaded from: {csv_path}")
                    return idx_to_name
                except Exception as e:
                    print(f"⚠️  Error loading {csv_path}: {e}")
        
        # Try JSON
        for json_path in [p for p in possible_paths if p.suffix == '.json']:
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    # Convert string keys to int
                    idx_to_name = {int(k): v for k, v in data.items()}
                    print(f"✅ Label mapping loaded from: {json_path}")
                    return idx_to_name
                except Exception as e:
                    print(f"⚠️  Error loading {json_path}: {e}")
        
        # Fallback: use generic names
        print(f"⚠️  No label mapping found, using generic names")
        return {i: f"Person_{i}" for i in range(self.num_classes)}
    
    def detect_and_recognize_face(self, image: np.ndarray) -> tuple:
        """
        Detect dan recognize wajah dari image
        
        Args:
            image: Input image (numpy array BGR dari OpenCV)
        
        Returns:
            (face_locations, face_names, confidences)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces menggunakan OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_locations = []
        face_names = []
        confidences = []
        
        # Process setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Crop face
            face_img = rgb_image[y:y+h, x:x+w]
            
            # Convert ke PIL Image
            pil_img = Image.fromarray(face_img)
            
            # Transform
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                conf_score = confidence.item()
            
            # Format lokasi wajah (top, right, bottom, left) - format face_recognition
            face_locations.append((y, x+w, y+h, x))
            
            # Get person name dari mapping
            person_name = self.idx_to_name.get(predicted_class, f"Unknown_{predicted_class}")
            face_names.append(person_name)
            confidences.append(conf_score)
        
        return face_locations, face_names, confidences


class ResNetApp(FaceRecognitionApp):
    """Face Recognition App dengan ResNet model"""
    
    def __init__(self, model_path: str, num_classes: int = 70):
        # Initialize parent app
        super().__init__(use_custom_model=True)
        
        # Set custom model
        self.resnet_adapter = ResNetAdapter(model_path, num_classes)
        self.model_info = {
            "name": "ResNet-34",
            "type": "custom",
            "classes": num_classes,
            "path": model_path
        }
        
        print(f"✅ ResNet App initialized")
        print(f"   Model: {self.model_info['name']}")
        print(f"   Classes: {self.model_info['classes']}")
    
    def recognize_face(self, frame: np.ndarray) -> Tuple:
        """
        Override recognize_face untuk menggunakan ResNet model
        
        Args:
            frame: Input frame dari webcam (numpy array BGR)
        
        Returns:
            (face_locations, face_names)
        """
        # Resize frame untuk processing lebih cepat
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Detect and recognize
        face_locations, face_names, confidences = \
            self.resnet_adapter.detect_and_recognize_face(small_frame)
        
        # Scale back face locations
        face_locations = [(top*2, right*2, bottom*2, left*2) 
                         for (top, right, bottom, left) in face_locations]
        
        # Format names dengan confidence
        formatted_names = [f"{name} ({conf*100:.1f}%)" 
                          for name, conf in zip(face_names, confidences)]
        
        return face_locations, formatted_names
    
    def process_and_recognize(self, image: np.ndarray) -> Tuple:
        """
        Process image dan return hasil dengan bounding box
        
        Args:
            image: Input image dari webcam
        
        Returns:
            (annotated_image, report_text)
        """
        if image is None:
            return None, "❌ Tidak ada gambar untuk diproses!"
        
        # Detect and recognize
        face_locations, face_names = self.recognize_face(image)
        
        # Draw bounding boxes
        result_image = image.copy()
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle
            cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label
            cv2.rectangle(result_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(result_image, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Generate report
        report = f"🔍 Deteksi Wajah\n"
        report += f"{'='*50}\n"
        report += f"📊 Jumlah wajah terdeteksi: {len(face_locations)}\n\n"
        
        if len(face_locations) > 0:
            report += "👤 Daftar wajah:\n"
            for i, name in enumerate(face_names, 1):
                report += f"  {i}. {name}\n"
        else:
            report += "⚠️  Tidak ada wajah yang terdeteksi\n"
        
        # Save snapshot
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = snapshot_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        report += f"\n💾 Snapshot disimpan: {filename}"
        
        return result_image, report
    
    def launch(self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
        """Launch Gradio interface"""
        
        with gr.Blocks(title="Face Recognition - ResNet") as demo:
            gr.Markdown(
                """
                # 🎭 Sistem Pengenalan Wajah - ResNet Model
                
                Ambil snapshot dari kamera, dan sistem akan mendeteksi dan mengenali wajah menggunakan ResNet-34.
                
                **Model Info:**
                - Architecture: ResNet-34
                - Classes: 70 orang
                - Accuracy: ~61%
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 Input Kamera")
                    camera_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Ambil Snapshot dari Kamera"
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🔍 Proses & Rekognisi", variant="primary", scale=2)
                        reset_btn = gr.Button("🔄 Reset", scale=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Hasil Analisis")
                    result_image = gr.Image(
                        label="Hasil Deteksi Wajah",
                        type="numpy"
                    )
                    recognition_report = gr.Textbox(
                        label="📋 Laporan Pengenalan",
                        lines=8,
                        interactive=False
                    )
            
            gr.Markdown("### 💡 Cara Penggunaan:")
            gr.Markdown(
                """
                1. Klik tombol kamera untuk mengambil snapshot
                2. Klik tombol "Proses & Rekognisi" untuk mendeteksi dan mengenali wajah
                3. Hasil akan ditampilkan dengan kotak hijau dan nama person
                4. Snapshot otomatis disimpan di folder `snapshots/`
                """
            )
            
            # Button callbacks
            process_btn.click(
                fn=self.process_and_recognize,
                inputs=camera_input,
                outputs=[result_image, recognition_report]
            )
            
            reset_btn.click(
                fn=lambda: (None, None, ""),
                inputs=[],
                outputs=[camera_input, result_image, recognition_report]
            )
        
        print(f"\n🚀 Launching Gradio interface...")
        print(f"   URL: http://{server_name}:{server_port}")
        
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True
        )


if __name__ == "__main__":
    # Test adapter
    app = ResNetApp(
        model_path="../checkpoints/resnet34_frozen_best.pth",
        num_classes=70
    )
    print("✅ ResNet adapter test successful!")
