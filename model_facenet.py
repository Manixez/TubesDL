import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image

class FaceNetModel(nn.Module):
    """
    FaceNet model dengan 2 komponen:
    1. MTCNN - untuk deteksi dan alignment wajah
    2. InceptionResnetV1 - untuk face recognition/embedding
    """
    def __init__(self, num_classes, pretrained=True, device=None):
        super(FaceNetModel, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 1. MTCNN untuk face detection
        # Params: image_size, margin, min_face_size, thresholds, factor, post_process
        self.mtcnn = MTCNN(
            image_size=160,  # FaceNet expects 160x160
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False  # Hanya ambil wajah yang paling jelas
        )
        
        # 2. InceptionResnetV1 untuk face recognition
        # Pretrained options: 'vggface2' atau 'casia-webface'
        if pretrained:
            self.facenet = InceptionResnetV1(
                pretrained='vggface2',  # Trained on VGGFace2 (larger dataset)
                classify=False,  # We want embeddings, not classification
                num_classes=None
            ).to(self.device)
        else:
            self.facenet = InceptionResnetV1(
                pretrained=None,
                classify=True,
                num_classes=num_classes
            ).to(self.device)
        
        self.facenet.eval()  # Set to eval mode for feature extraction
        
        # 3. Classification head (untuk fine-tuning pada dataset kita)
        # FaceNet menghasilkan embedding 512-dimensional
        self.embedding_size = 512
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ).to(self.device)
        
        self.num_classes = num_classes
        
    def detect_face(self, image):
        """
        Deteksi wajah dari image menggunakan MTCNN
        
        Args:
            image: PIL Image atau tensor [C, H, W]
        
        Returns:
            face_tensor: Tensor wajah yang sudah di-crop dan aligned [1, C, 160, 160]
            boxes: Bounding box deteksi
            probs: Probability deteksi
        """
        # Jika input adalah tensor, convert ke PIL Image
        if isinstance(image, torch.Tensor):
            # Denormalize jika perlu
            if image.min() < 0:  # Jika sudah dinormalisasi
                image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Deteksi wajah dengan MTCNN
        face_tensor, prob = self.mtcnn(image, return_prob=True)
        
        return face_tensor, prob
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W] - bisa 224x224 (dari dataloader)
            return_embedding: Jika True, return embedding instead of classification
        
        Returns:
            output: Classification logits [B, num_classes] atau embeddings [B, 512]
        """
        batch_size = x.size(0)
        
        # Jika input bukan 160x160, resize dulu
        if x.size(-1) != 160 or x.size(-2) != 160:
            x = torch.nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        # Get embeddings dari FaceNet
        embeddings = self.facenet(x)  # [B, 512]
        
        if return_embedding:
            return embeddings
        
        # Classification
        output = self.classifier(embeddings)  # [B, num_classes]
        
        return output
    
    def get_embedding(self, x):
        """
        Get face embedding untuk face verification/identification
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            embeddings: Face embeddings [B, 512]
        """
        return self.forward(x, return_embedding=True)
    
    def freeze_backbone(self):
        """Freeze FaceNet backbone untuk fine-tuning hanya classifier"""
        for param in self.facenet.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze FaceNet backbone untuk full fine-tuning"""
        for param in self.facenet.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    
    # Test model dengan 70 kelas (sesuai dataset face recognition)
    num_classes = 70
    print(f"\nCreating FaceNet model for {num_classes} classes...")
    model = FaceNetModel(num_classes=num_classes, pretrained=True, device=device)
    
    print("\nModel components:")
    print(f"1. MTCNN (Face Detection): {type(model.mtcnn).__name__}")
    print(f"2. FaceNet (Embedding): {type(model.facenet).__name__}")
    print(f"3. Classifier: {model.classifier}")
    
    # Test dengan input 224x224 (dari dataloader)
    print("\n" + "="*60)
    print("Testing with 224x224 input (from dataloader)...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (logits): {output.shape}")  # Should be [4, 70]
    
    # Get embeddings
    embeddings = model.get_embedding(x)
    print(f"Embedding shape: {embeddings.shape}")  # Should be [4, 512]
    
    # Test face detection dengan PIL Image
    print("\n" + "="*60)
    print("Testing face detection with MTCNN...")
    from PIL import Image
    import numpy as np
    
    # Create dummy RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    
    try:
        face_tensor, prob = model.detect_face(dummy_img)
        if face_tensor is not None:
            print(f"✓ Face detected!")
            print(f"  Face tensor shape: {face_tensor.shape}")
            print(f"  Detection probability: {prob:.4f}")
        else:
            print("✗ No face detected (expected for random image)")
    except Exception as e:
        print(f"Detection test: {str(e)}")
    
    # Test freeze/unfreeze
    print("\n" + "="*60)
    print("Testing freeze/unfreeze backbone...")
    
    # Count trainable params before freeze
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (before freeze): {trainable_before:,}")
    
    # Freeze backbone
    model.freeze_backbone()
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (after freeze): {trainable_after_freeze:,}")
    
    # Unfreeze backbone
    model.unfreeze_backbone()
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (after unfreeze): {trainable_after_unfreeze:,}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
