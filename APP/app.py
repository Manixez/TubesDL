import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import json
from typing import Tuple, List, Dict, Any, Optional

# Optional import - untuk demo menggunakan builtin model
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False
    print("⚠️  face_recognition not installed - use custom model adapter instead")

class FaceRecognitionApp:
    """
    Aplikasi Face Recognition yang modular dan mudah diintegrasikan dengan model custom.
    
    Untuk mengintegrasikan dengan model Anda:
    1. Override method recognize_face() atau set_custom_model()
    2. Atau replace face_recognition dengan model Anda
    """
    
    def __init__(self, use_custom_model=False, custom_model=None):
        self.known_face_encodings = []
        self.known_face_names = []
        self.captured_image = None
        self.process_this_frame = True
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Model configuration
        self.use_custom_model = use_custom_model
        self.custom_model = custom_model
        self.model_info = {
            "name": "face_recognition (ageitgey)",
            "version": "1.3.0",
            "type": "builtin"
        }
        
        # Create output directory for snapshots
        self.snapshot_dir = Path("snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Create reference faces directory
        self.reference_dir = Path("reference_faces")
        self.reference_dir.mkdir(exist_ok=True)
        
        # Create config directory
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Tolerance setting untuk flexibility
        self.tolerance = self.load_config().get("tolerance", 0.6)
        self.model_type = self.load_config().get("model_type", "hog")
        
        self.load_reference_faces()
    
    def load_config(self) -> Dict:
        """Load configuration from config.json"""
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_config(self) -> None:
        """Save configuration to config.json"""
        config = {
            "tolerance": self.tolerance,
            "model_type": self.model_type,
            "model_info": self.model_info
        }
        config_file = self.config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def set_custom_model(self, model, model_name: str = "Custom Model") -> None:
        """
        Set custom model untuk recognition.
        
        Args:
            model: Model object yang sudah di-load
            model_name: Nama model untuk logging
        
        Example:
            custom_model = load_your_model("path/to/model.h5")
            app.set_custom_model(custom_model, "YourModelName")
        """
        self.custom_model = model
        self.use_custom_model = True
        self.model_info = {
            "name": model_name,
            "version": "custom",
            "type": "custom"
        }
        self.save_config()
        print(f"✅ Custom model '{model_name}' berhasil di-set")
    
    def load_reference_faces(self):
        """Load reference faces from reference_faces directory"""
        if not HAS_FACE_RECOGNITION or self.use_custom_model:
            return
        
        # Hanya load jika face_recognition tersedia
        for person_name in os.listdir(self.reference_dir):
            person_path = self.reference_dir / person_name
            if person_path.is_dir():
                for image_name in os.listdir(person_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = person_path / image_name
                        try:
                            image = face_recognition.load_image_file(str(image_path))
                            face_encodings = face_recognition.face_encodings(image)
                            if face_encodings:
                                self.known_face_encodings.append(face_encodings[0])
                                self.known_face_names.append(person_name)
                        except Exception as e:
                            print(f"⚠️ Error loading {image_path}: {e}")
    
    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize single face. Override ini untuk menggunakan custom model.
        
        Args:
            face_encoding: Face encoding dari model
            
        Returns:
            Tuple[name, confidence]: (Nama, Confidence score 0-1)
        
        Example override:
            def recognize_face(self, face_encoding):
                prediction = self.custom_model.predict(face_encoding)
                return prediction['name'], prediction['confidence']
        """
        if self.use_custom_model and self.custom_model:
            # Panggil custom model Anda di sini
            # return self.custom_model.predict(face_encoding)
            pass
        
        # Default: gunakan face_recognition library (jika tersedia)
        if not HAS_FACE_RECOGNITION or len(self.known_face_encodings) == 0:
            return "Demo_Model", 0.85
        
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding,
            tolerance=self.tolerance
        )
        
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        name = "Unknown"
        confidence = 0.0
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
        
        return name, confidence
    
    def process_image(self, image):
        """Process image to detect and recognize faces"""
        if image is None:
            return None, "Silakan ambil snapshot terlebih dahulu"
        
        if not HAS_FACE_RECOGNITION:
            # Demo mode - jika face_recognition tidak tersedia
            # Return dummy results untuk testing
            self.face_locations = [(100, 300, 200, 150)]  # Dummy face location
            self.face_names = [("Demo_Person", 0.85)]
            result_image = self.draw_face_boxes(image)
            return result_image, "✅ Demo Mode\n\n🤖 Model Capstone Anda akan dijalankan di sini\n\n✨ Silakan setup custom_model_adapter.py"
        
        # Resize image for faster processing
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        self.face_locations = face_recognition.face_locations(rgb_small_image, model=self.model_type)
        self.face_encodings = face_recognition.face_encodings(rgb_small_image, self.face_locations)
        
        self.face_names = []
        
        # Recognize each face
        for face_encoding in self.face_encodings:
            name, confidence = self.recognize_face(face_encoding)
            self.face_names.append((name, confidence))
        
        # Draw results on image
        result_image = self.draw_face_boxes(image)
        
        # Generate report
        report = self.generate_report()
        
        return result_image, report
    
    def draw_face_boxes(self, image):
        """Draw boxes around detected faces"""
        result_image = image.copy()
        scale_factor = 4  # Because we resized image by 0.25
        
        for (top, right, bottom, left), (name, confidence) in zip(self.face_locations, self.face_names):
            # Scale back up
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor
            
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = name if name != "Unknown" else "Unknown"
            if confidence > 0:
                label += f" ({confidence:.2%})"
            
            cv2.rectangle(result_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                result_image,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return result_image
    
    def generate_report(self):
        """Generate recognition report"""
        if not self.face_names:
            return "❌ Tidak ada wajah terdeteksi"
        
        report = f"✅ Terdeteksi {len(self.face_names)} wajah:\n\n"
        for i, (name, confidence) in enumerate(self.face_names, 1):
            if name != "Unknown":
                report += f"  {i}. {name} ({confidence:.2%})\n"
            else:
                report += f"  {i}. Wajah tidak dikenali\n"
        
        return report
    
    def capture_snapshot(self):
        """Placeholder for camera capture - will be handled by Gradio"""
        return None, "Gunakan interface Gradio untuk mengambil snapshot"
    
    def save_snapshot(self, image):
        """Save snapshot to file"""
        if image is None:
            return "❌ Tidak ada gambar untuk disimpan"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.snapshot_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), image)
        
        return f"✅ Snapshot disimpan: {filename}"


# Initialize app
app = FaceRecognitionApp()


def process_and_recognize(image):
    """Main function for face recognition"""
    if image is None:
        return None, "❌ Silakan ambil snapshot terlebih dahulu"
    
    # Convert PIL image to OpenCV format if needed
    if isinstance(image, np.ndarray):
        cv_image = image
    else:
        cv_image = np.array(image)
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    result_image, report = app.process_image(cv_image)
    
    # Save snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = app.snapshot_dir / f"snapshot_{timestamp}.jpg"
    cv2.imwrite(str(filename), cv_image)
    
    save_status = f"✅ Snapshot disimpan ke: snapshots/snapshot_{timestamp}.jpg"
    
    return result_image, report + "\n\n" + save_status


# Create Gradio Interface
with gr.Blocks(title="Face Recognition System") as demo:
    gr.Markdown(
        """
        # 🎭 Sistem Pengenalan Wajah (Face Recognition)
        
        Ambil snapshot dari kamera, dan sistem akan mendeteksi dan mengenali wajah secara otomatis.
        
        **Fitur:**
        - 📷 Live camera input
        - 🔍 Face detection menggunakan OpenCV
        - 🧠 Face recognition menggunakan deep learning
        - 💾 Auto-save snapshot
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
        3. Hasil akan ditampilkan dengan kotak hijau (dikenali) atau merah (tidak dikenali)
        4. Snapshot otomatis disimpan di folder `snapshots/`
        
        **Setup Data Referensi:**
        - Buat folder `reference_faces/`
        - Dalam folder tersebut, buat subfolder untuk setiap orang (contoh: `reference_faces/John_Doe/`)
        - Masukkan foto wajah ke dalam subfolder tersebut
        """
    )
    
    # Button callbacks
    process_btn.click(
        fn=process_and_recognize,
        inputs=camera_input,
        outputs=[result_image, recognition_report]
    )
    
    reset_btn.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[camera_input, result_image, recognition_report]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
