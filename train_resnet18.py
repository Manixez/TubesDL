import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# Import custom modules
from datareader import FaceRecognition
from model_resnet18 import ResNet18Model

# Set random seed for reproducibility
RANDOM_SEED = 2025
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) if torch.cuda.is_available() else None
np.random.seed(RANDOM_SEED)

class EarlyStopping:
    """Menghentikan training jika validation loss tidak membaik setelah sekian epoch."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 save_dir='checkpoints',
                 model_name='resnet18'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1} - Training")
        print("-" * 60)
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            # Skip if batch is None (all invalid samples)
            if images is None:
                continue
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * correct / total
                print(f"Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        print(f"\nEpoch {epoch+1} - Validation")
        print("-" * 60)
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(self.val_loader):
                # Skip if batch is None (all invalid samples)
                if images is None:
                    continue
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        print(f"Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}%")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save last checkpoint
        last_path = os.path.join(self.save_dir, f'{self.model_name}_last.pth')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot 1: Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss', 
                        marker='o', linewidth=2, markersize=6, color='#2E86AB')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss', 
                        marker='s', linewidth=2, markersize=6, color='#A23B72')
        axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Add min loss annotation
        min_val_loss_idx = np.argmin(self.history['val_loss'])
        min_val_loss = self.history['val_loss'][min_val_loss_idx]
        axes[0, 0].annotate(f'Min Val Loss\n{min_val_loss:.4f}',
                           xy=(min_val_loss_idx + 1, min_val_loss),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc', 
                        marker='o', linewidth=2, markersize=6, color='#2E86AB')
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc', 
                        marker='s', linewidth=2, markersize=6, color='#A23B72')
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Add max accuracy annotation
        max_val_acc_idx = np.argmax(self.history['val_acc'])
        max_val_acc = self.history['val_acc'][max_val_acc_idx]
        axes[0, 1].annotate(f'Best Val Acc\n{max_val_acc:.2f}%',
                           xy=(max_val_acc_idx + 1, max_val_acc),
                           xytext=(10, -30), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 3: Learning Rate
        axes[1, 0].plot(epochs, self.history['lr'], label='Learning Rate', 
                        marker='D', linewidth=2, markersize=6, color='#F18F01')
        axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Train vs Val Gap
        gap = np.array(self.history['train_acc']) - np.array(self.history['val_acc'])
        axes[1, 1].plot(epochs, gap, label='Train-Val Gap', 
                        marker='o', linewidth=2, markersize=6, color='#C73E1D')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].fill_between(epochs, gap, 0, where=(gap > 0), 
                                alpha=0.3, color='red', label='Overfitting')
        axes[1, 1].fill_between(epochs, gap, 0, where=(gap < 0), 
                                alpha=0.3, color='blue', label='Underfitting')
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy Gap (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Monitor)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.model_name}_history.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved!")
    
    def plot_confusion_matrix(self, all_labels, all_preds, num_classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.model_name}_confusion_matrix.png'), dpi=150)
        plt.close()
        
        print(f"✓ Confusion matrix saved!")
    
    def train(self, num_epochs, freeze_backbone_epochs=0):
        """Main training loop"""
        print("="*60)
        print(f"Starting Training - {self.model_name}")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Num Epochs: {num_epochs}")
        print(f"Freeze Backbone: {freeze_backbone_epochs} epochs")
        print(f"Train Batches: {len(self.train_loader)}")
        print(f"Val Batches: {len(self.val_loader)}")
        print("="*60)
        
        start_time = time.time()
        
        # Inisialisasi Early Stopping (Stop jika loss naik 5x berturut-turut)
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)

        for epoch in range(num_epochs):
            # Unfreeze backbone after specified epochs
            if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
                print("\n" + "="*60)
                print(f"Unfreezing backbone at epoch {epoch+1}")
                self.model.unfreeze_backbone()
                print("="*60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, all_preds, all_labels = self.validate(epoch)
            
            # Early Stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("\n" + "!"*60)
                print("Early Stopping triggered! Training stopped to prevent overfitting.")
                print("!"*60)
                break
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            
            self.save_checkpoint(epoch + 1, val_acc, is_best)
            
            # Print epoch summary
            print("\n" + "="*60)
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            print("="*60)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print("="*60)
        
        # Plot final results
        self.plot_history()
        self.plot_confusion_matrix(all_labels, all_preds, self.model.num_classes)


def collate_fn_filter_invalid(batch):
    """Custom collate function to filter out invalid samples (label = -1)"""
    # Filter out invalid samples
    batch = [(img, label, path) for img, label, path in batch if label != -1]
    
    if len(batch) == 0:
        # Return empty batch if all samples are invalid
        return None, None, None
    
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    
    return images, labels, paths


def main():
    # ========== Configuration ==========
    CONFIG = {
        'data_dir': 'Dataset/Train',
        'num_classes': 70,
        'img_size': (224, 224),
        'batch_size': 32,
        'num_epochs': 50,  
        'freeze_backbone_epochs': 50, 
        'learning_rate': 0.001,
        'weight_decay': 1e-4, 
        'step_size': 10,
        'gamma': 0.5,
        'num_workers': 0,
        'save_dir': 'checkpoints',
        'model_name': 'resnet18_frozen',
        'pretrained': True
    }
    
    print("="*60)
    print("ResNet-18 Face Recognition Training")
    print("="*60)
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    print("="*60)
    
    # ========== Device Setup ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ========== Dataset & DataLoader ==========
    print("\n" + "="*60)
    print("Loading Datasets...")
    print("="*60)
    
    train_dataset = FaceRecognition(
        data_dir=CONFIG['data_dir'],
        img_size=CONFIG['img_size'],
        split='train'
    )
    
    val_dataset = FaceRecognition(
        data_dir=CONFIG['data_dir'],
        img_size=CONFIG['img_size'],
        split='val'
    )
    
    print(f"Train samples (total): {len(train_dataset)}")
    print(f"Val samples (total): {len(val_dataset)}")
    
    # Validate dataset integrity
    print("\n" + "="*60)
    print("Validating dataset integrity...")
    print("="*60)
    
    # Check train dataset
    train_valid_count = 0
    train_invalid_indices = []
    for idx in range(len(train_dataset)):
        img, label, path = train_dataset[idx]
        if label == -1:
            train_invalid_indices.append(idx)
        else:
            train_valid_count += 1
    
    # Check val dataset
    val_valid_count = 0
    val_invalid_indices = []
    for idx in range(len(val_dataset)):
        img, label, path = val_dataset[idx]
        if label == -1:
            val_invalid_indices.append(idx)
        else:
            val_valid_count += 1
    
    print(f"Train: {train_valid_count} valid, {len(train_invalid_indices)} invalid")
    print(f"Val: {val_valid_count} valid, {len(val_invalid_indices)} invalid")
    
    if len(train_invalid_indices) > 0:
        print(f"\n⚠️ Warning: {len(train_invalid_indices)} invalid images in train set will be skipped during training")
    if len(val_invalid_indices) > 0:
        print(f"⚠️ Warning: {len(val_invalid_indices)} invalid images in val set will be skipped during validation")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn_filter_invalid  # Custom collate to filter invalid samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn_filter_invalid  # Custom collate to filter invalid samples
    )
    
    # ========== Model Setup ==========
    print("\n" + "="*60)
    print("Creating Model...")
    print("="*60)
    
    model = ResNet18Model(
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained'],
        device=device
    )
    
    # Freeze backbone initially if specified
    if CONFIG['freeze_backbone_epochs'] > 0:
        model.freeze_backbone()
        print(f"✓ Backbone frozen for first {CONFIG['freeze_backbone_epochs']} epochs")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========== Loss & Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['step_size'],
        gamma=CONFIG['gamma']
    )
    
    # ========== Training ==========
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=CONFIG['save_dir'],
        model_name=CONFIG['model_name']
    )
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        freeze_backbone_epochs=CONFIG['freeze_backbone_epochs']
    )
    
    print("\n✓ Training script completed successfully!")


if __name__ == "__main__":
    main()