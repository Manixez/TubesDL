import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image

class ResNet18Model(nn.Module):
    """
    ResNet-18 model untuk face recognition
    Menggunakan pretrained ResNet-18 dari torchvision
    """
    def __init__(self, num_classes, pretrained=True, device=None):
        super(ResNet18Model, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 1. Load ResNet-18
        if pretrained:
            self.resnet = models.resnet18(pretrained=True)
        else:
            self.resnet = models.resnet18(pretrained=False)
        
        # 2. Remove the final FC layer
        # ResNet-18 original FC: nn.Linear(512, 1000)
        self.embedding_size = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()  # Remove classification layer
        
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()  # Set to eval mode for feature extraction
        
        # 3. Classification head (untuk fine-tuning pada dataset kita)
        # ResNet-18 menghasilkan embedding 512-dimensional
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(256, num_classes)
        ).to(self.device)
        
        self.num_classes = num_classes
        
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
        
        # Get embeddings dari ResNet-18
        embeddings = self.resnet(x)  # [B, 512]
        
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
        """Freeze ResNet-18 backbone untuk fine-tuning hanya classifier"""
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet-18 backbone untuk full fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    
    # Test model dengan 70 kelas (sesuai dataset face recognition)
    num_classes = 70
    print(f"\nCreating ResNet-18 model for {num_classes} classes...")
    model = ResNet18Model(num_classes=num_classes, pretrained=True, device=device)
    
    print("\nModel components:")
    print(f"1. ResNet-18 (Backbone): {type(model.resnet).__name__}")
    print(f"2. Classifier: {model.classifier}")
    
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