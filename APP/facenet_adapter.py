"""
FaceNet Model Adapter untuk Face Recognition App

Adapter khusus untuk model FaceNet yang sudah di-train
"""

import sys
from pathlib import Path

# Add parent directory to path untuk import model_facenet
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple
import json
from torchvision import transforms

from model_facenet import FaceNetModel
from app import FaceRecognitionApp


class FaceNetAdapter:
    """
    Adapter untuk FaceNet model (InceptionResnetV1 + MTCNN)
    """
    
    def __init__(self, model_path: str, num_classes: int = 70):
        """
        Initialize FaceNet adapter
        
        Args:
            model_path: Path ke model weights (.pth file)
            num_classes: Jumlah kelas/orang yang dikenali
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.person_mapping = {}
        self.confidence_threshold = 0.05  # LOWERED for debugging (was 0.5)
        
        # Initialize transforms untuk preprocessing
        # Gunakan ImageNet normalization (sama seperti training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Model akan resize ke 160x160 internally
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model and person mapping
        self.load_model()
        self.load_person_mapping()
        
        print(f"✅ FaceNet Adapter initialized")
        print(f"   Device: {self.device}")
        print(f"   Num Classes: {self.num_classes}")
    
    def load_model(self):
        """Load FaceNet model dari checkpoint"""
        try:
            # Load checkpoint first to auto-detect num_classes
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Auto-detect num_classes from checkpoint if not provided
            detected_classes = None
            for key in checkpoint.keys():
                if 'classifier.5.weight' in key:
                    detected_classes = checkpoint[key].shape[0]
                    print(f"🔍 Auto-detected num_classes from checkpoint: {detected_classes}")
                    break
            
            # Update num_classes if detected and different
            if detected_classes is not None and detected_classes != self.num_classes:
                print(f"⚠️  Updating num_classes: {self.num_classes} → {detected_classes}")
                self.num_classes = detected_classes
            
            # Create model with correct num_classes
            self.model = FaceNetModel(
                num_classes=self.num_classes,
                pretrained=True,  # Creates facenet with classify=False
                device=self.device
            )
            
            # Filter out facenet.logits (8631 classes dari VGGFace2 pretrained)
            # Kita hanya pakai yang sudah kita train: MTCNN + facenet backbone + classifier
            state_dict = {}
            for key, value in checkpoint.items():
                # Skip facenet.logits yang 8631 classes (dari pretrained)
                if 'facenet.logits' not in key:
                    state_dict[key] = value
            
            # Load with strict=False to ignore missing/extra keys
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Set to evaluation mode
            self.model.eval()
            self.model = self.model.to(self.device)
            
            print(f"✅ FaceNet model loaded from: {self.model_path}")
            print(f"   Loaded {len(state_dict)} layers")
            print(f"   Num Classes: {self.num_classes}")
            if missing_keys:
                print(f"   ⚠️  Missing keys: {len(missing_keys)} (expected for facenet.logits)")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_person_mapping(self):
        """
        Load mapping dari label ID ke person name
        
        Format: {0: "Abraham Ganda Napitu", 1: "Abu Bakar Siddiq Siregar", ...}
        """
        # Try to load from JSON first
        mapping_file = Path(__file__).parent / "config" / "person_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                # Convert string keys to int if needed
                self.person_mapping = {int(k) if k.isdigit() else k: v for k, v in data.items()}
            print(f"✅ Person mapping loaded from {mapping_file}")
        else:
            # Try to load from label_mapping.csv
            csv_file = Path(__file__).parent.parent / "Dataset" / "label_mapping.csv"
            
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                self.person_mapping = dict(zip(df['label'], df['person_name']))
                
                # Save to JSON for future use
                mapping_file.parent.mkdir(parents=True, exist_ok=True)
                with open(mapping_file, 'w') as f:
                    json.dump({str(k): v for k, v in self.person_mapping.items()}, f, indent=2)
                
                print(f"✅ Person mapping loaded from {csv_file}")
                print(f"   Saved to {mapping_file} for future use")
            else:
                # Default: Create numbered mapping
                self.person_mapping = {i: f"Person_{i}" for i in range(self.num_classes)}
                print(f"⚠️  Using default person mapping (Person_0, Person_1, ...)")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image untuk FaceNet
        
        Args:
            image: RGB image dari face detection (numpy array)
        
        Returns:
            Preprocessed tensor [1, 3, 160, 160]
        """
        # Convert numpy array ke PIL Image
        if isinstance(image, np.ndarray):
            # Ensure RGB
            if image.shape[-1] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Convert BGR to RGB if needed (from OpenCV)
            # Assume input is already RGB from face_recognition
            
            image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_from_image(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict dari face image (bukan encoding)
        
        Args:
            face_image: Cropped face image (RGB, numpy array)
        
        Returns:
            Tuple[person_name, confidence]
        """
        if self.model is None:
            return "Model_Not_Loaded", 0.0
        
        try:
            # Preprocess image
            tensor = self.preprocess_image(face_image)
            
            # Predict
            with torch.no_grad():
                # Get embeddings first (512-dim)
                embeddings = self.model.get_embedding(tensor)  # [1, 512]
                
                # Apply classifier
                outputs = self.model.classifier(embeddings)  # [1, 70]
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted = torch.max(probs, 1)
                
                person_id = predicted.item()
                confidence_score = confidence.item()
            
            # Map to person name
            person_name = self.person_mapping.get(person_id, f"Unknown_{person_id}")
            
            # Debug: Print top predictions
            top5_conf, top5_idx = torch.topk(probs, min(5, probs.size(1)), dim=1)
            print(f"\n🔍 Debug - Top 5 Predictions:")
            for i in range(min(5, top5_idx.size(1))):
                idx = top5_idx[0, i].item()
                conf = top5_conf[0, i].item()
                name = self.person_mapping.get(idx, f"Unknown_{idx}")
                print(f"   {i+1}. {name}: {conf:.4f} ({conf*100:.2f}%)")
            
            # Check threshold (lower for debugging)
            if confidence_score < self.confidence_threshold:
                # Don't set to Unknown yet, show actual prediction with warning
                person_name = f"{person_name} (Low Confidence)"
            
            return person_name, confidence_score
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return "Error", 0.0
    
    def predict(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Predict dari face encoding (untuk compatibility dengan base class)
        
        Note: FaceNet tidak menggunakan face encoding 128-dim dari face_recognition,
        tapi langsung dari image. Method ini untuk compatibility saja.
        
        Args:
            face_encoding: Not used, kept for interface compatibility
        
        Returns:
            Tuple[person_name, confidence]
        """
        # Untuk FaceNet, kita tidak bisa predict dari encoding
        # Karena FaceNet butuh raw image untuk extract features sendiri
        return "Use_predict_from_image", 0.0


class FaceNetApp(FaceRecognitionApp):
    """
    Face Recognition App dengan FaceNet model
    
    Override untuk menggunakan predict_from_image instead of predict
    """
    
    def __init__(self, model_path: str, num_classes: int = 70):
        """
        Initialize FaceNet app
        
        Args:
            model_path: Path ke model weights (.pth)
            num_classes: Jumlah orang yang dikenali
        """
        # Initialize parent FaceRecognitionApp
        super().__init__(use_custom_model=True)
        
        # Create FaceNet adapter
        self.model_adapter = FaceNetAdapter(model_path, num_classes)
        
        # Update model info
        self.model_info = {
            "name": "FaceNet",
            "version": "1.0",
            "type": "facenet",
            "architecture": "InceptionResnetV1",
            "pretrained": "VGGFace2",
            "num_classes": num_classes,
            "model_path": str(model_path)
        }
        self.save_config()
        
        print(f"✅ FaceNetApp initialized")
        print(f"📊 Model Info: {self.model_info}")
    
    def launch(self, share=False, server_name="127.0.0.1", server_port=7860):
        """
        Launch Gradio interface
        
        Args:
            share: Create public Gradio link
            server_name: Server address
            server_port: Server port
        """
        import gradio as gr
        
        def process_image(image):
            """Process uploaded image"""
            if image is None:
                return None, "No image provided"
            
            # Process frame
            result = self.process_frame(image)
            
            # Get face count
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            import face_recognition
            face_locations = face_recognition.face_locations(rgb_frame)
            
            report = f"Detected {len(face_locations)} face(s)"
            
            return result, report
        
        # Create interface
        with gr.Blocks(title="FaceNet Face Recognition") as demo:
            gr.Markdown("# 🎭 FaceNet Face Recognition")
            gr.Markdown(f"**Model**: {self.model_info['architecture']} | **Classes**: {self.model_info['num_classes']}")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image", type="numpy")
                    process_btn = gr.Button("🔍 Recognize Faces", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Result")
                    output_text = gr.Textbox(label="Detection Report", lines=3)
            
            process_btn.click(
                fn=process_image,
                inputs=[input_image],
                outputs=[output_image, output_text]
            )
            
            gr.Markdown("### 💡 Tips:")
            gr.Markdown("- Upload image containing faces\n- System will detect and recognize all faces\n- Green box = recognized, Red box = unknown")
        
        # Launch
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )
    
    def process_frame(self, frame):
        """
        Override process_frame untuk menggunakan FaceNet MTCNN + classifier
        """
        import face_recognition
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Process each face
        for (top, right, bottom, left) in face_locations:
            # Extract face region
            face_image = rgb_frame[top:bottom, left:right]
            
            # Predict menggunakan FaceNet
            name, confidence = self.model_adapter.predict_from_image(face_image)
            
            # Draw box dan label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Label dengan nama dan confidence
            label = f"{name} ({confidence:.2%})"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame


# ============================================================
# QUICK START
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FaceNet Model Adapter - Testing")
    print("=" * 60)
    
    # Path ke model - check both possible locations
    script_dir = Path(__file__).parent  # APP folder
    
    # Try parent folder first (when run from APP)
    model_path = script_dir.parent / "best_facenet_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print(f"💡 Please train the model first using train_facenet.py")
        sys.exit(1)
    
    # Create app
    print("\n📦 Creating FaceNet app...")
    app = FaceNetApp(model_path=model_path, num_classes=70)
    
    # Test dengan sample image
    print("\n🧪 Testing dengan sample image...")
    sample_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    name, confidence = app.model_adapter.predict_from_image(sample_image)
    print(f"   Prediction: {name} (confidence: {confidence:.2%})")
    
    print("\n✅ FaceNet adapter test completed!")
    print("=" * 60)
    print("\n💡 To run the web app:")
    print("   python run_with_facenet.py")
    print("=" * 60)
