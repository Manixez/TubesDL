import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import (
    ToTensor, Normalize, Compose, RandomResizedCrop, 
    RandomHorizontalFlip, ColorJitter, Resize
)
from sklearn.model_selection import StratifiedShuffleSplit

# Set random seed for reproducibility
RANDOM_SEED = 2025
random.seed(RANDOM_SEED)      # random seed Python
np.random.seed(RANDOM_SEED)   # random seed Numpy
torch.manual_seed(RANDOM_SEED) # random seed PyTorch
torch.cuda.manual_seed(RANDOM_SEED) if torch.cuda.is_available() else None  # GPU seed



class FaceRecognition(Dataset):
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 data_dir='Dataset/Train_Cropped',  # Changed to Train_Cropped
                 img_size=(224, 224),  # Update default size for ViT
                 transform=None,
                 split='train'
                 ):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
        # Default transforms with data augmentation for training
        if transform is None:
            if split == 'train':
                self.transform = Compose([
                    RandomResizedCrop(img_size, scale=(0.9, 1.0)),  # Less aggressive crop
                    RandomHorizontalFlip(p=0.5),  # Horizontal flip
                    ColorJitter(
                        brightness=0.1,  # Reduced brightness variation
                        contrast=0.1,    # Reduced contrast variation
                        saturation=0.1   # Reduced saturation variation
                    ),
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
            else:  # validation/test transforms
                self.transform = Compose([
                    Resize(img_size),  # Simple resize to target size
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
        else:
            self.transform = transform

        # Baca CSV terlebih dahulu
        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        
        # Verifikasi integritas data dan filter hanya gambar yang valid
        print(f"Verifying {len(df)} images...")
        valid_data = []
        invalid_count = 0
        
        for idx, row in df.iterrows():
            img_file = row['filename']
            label = row['label']
            img_path = os.path.join(data_dir, img_file)
            
            # Check if file exists
            if not os.path.exists(img_path):
                print(f'Warning: Image not found, skipping: {img_file}')
                invalid_count += 1
                continue
                
            # Try to open and verify image
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        print(f'Info: Will convert {img_file} from {img.mode} to RGB')
                # Only add to valid_data if successful
                valid_data.append((img_file, label))
            except Exception as e:
                print(f'Error reading {img_path}, skipping: {str(e)}')
                invalid_count += 1
        
        print(f"Valid images: {len(valid_data)}/{len(df)} (Skipped: {invalid_count})")
        
        # Remap labels to be continuous (0, 1, 2, ..., num_valid_classes-1)
        # This is necessary because some people might have no valid images
        original_labels = [label for _, label in valid_data]
        unique_original_labels = sorted(list(set(original_labels)))
        label_remap = {old_label: new_label for new_label, old_label in enumerate(unique_original_labels)}
        
        # Apply remapping
        remapped_data = [(img_file, label_remap[label]) for img_file, label in valid_data]
        
        # Use remapped data
        self.image_files = [img_file for img_file, _ in remapped_data]
        self.labels = [label for _, label in remapped_data]
        
        # Pasangkan setiap image file dengan labelnya dan simpan dalam list of tuples
        all_data = remapped_data
        
        # Menggunakan stratified split untuk memastikan distribusi kelas seimbang
        # Dengan dataset kecil (4-5 gambar per kelas), gunakan test_size yang lebih kecil
        X = list(range(len(all_data)))
        y = [label for _, label in all_data]
        
        # Hitung test_size minimal (minimal 1 gambar per kelas untuk validation)
        num_classes = len(set(y))
        min_test_size = max(num_classes, int(len(all_data) * 0.15))  # minimal 15% atau num_classes
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=min_test_size, random_state=RANDOM_SEED)
        train_indices, val_indices = next(sss.split(X, y))
        
        if split == 'train':
            self.data = [all_data[i] for i in train_indices]
        elif split == 'val':
            self.data = [all_data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'")
        
        # Define default transforms
        self.default_transform = Compose([
            ToTensor(),  # Converts PIL/numpy to tensor and scales to [0,1]
            Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Load Data dan Label
            img_name = self.data[idx][0]
            img_path = os.path.join(self.data_dir, img_name)
            
            # Load image using PIL directly (more reliable than cv2 for this case)
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms (including resize and normalization)
            try:
                if self.transform:
                    image = self.transform(image)
                else:
                    image = self.default_transform(image)
            except Exception as e:
                raise ValueError(f'Error applying transforms to {img_name}: {str(e)}')
            
            label = self.data[idx][1]
            
            # return data gambar, label, dan nama file (bukan path penuh)
            return image, label, img_name
            
        except Exception as e:
            print(f'Error processing image at index {idx}: {str(e)}')
            # Return default/placeholder data in case of error
            empty_tensor = torch.zeros((3, *self.img_size))
            return empty_tensor, -1, 'error'

if __name__ == "__main__":
    # Test both splits
    train_dataset = FaceRecognition(split='train')
    val_dataset = FaceRecognition(split='val')
    
    print(f"Train data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    
    # Sample 5 random images from each dataset
    train_indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    
    print("\nTrain Dataset Samples:")
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        print(f"Train data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)
    
    print("\nValidation Dataset Samples:")
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        print(f"Val data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)

    # Create a figure with subplots for both train and val
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot train images
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * train_dataset.IMAGENET_STD[j] + train_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Train: {label}")
        axes[0, i].axis('off')
    
    # Plot val images
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * val_dataset.IMAGENET_STD[j] + val_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[1, i].imshow(img_display)
        axes[1, i].set_title(f"Val: {label}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('cek_augmentasi.png')
    plt.close()
    
    # Visualisasi distribusi dataset
    print("\n" + "="*60)
    print("DATASET DISTRIBUTION VISUALIZATION")
    print("="*60)
    
    # Baca CSV dan label mapping
    csv_path = 'Dataset/train.csv'
    label_map_path = 'Dataset/label_mapping.csv'
    
    df = pd.read_csv(csv_path)
    label_df = pd.read_csv(label_map_path)
    
    # Hitung jumlah foto per orang
    photo_counts = df.groupby('person_name').size().reset_index(name='count')
    
    # Sort berdasarkan nama untuk konsistensi
    photo_counts = photo_counts.sort_values('person_name')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    bars = ax.bar(range(len(photo_counts)), photo_counts['count'], 
                   color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Color bars differently based on count
    colors = []
    for count in photo_counts['count']:
        if count >= 5:
            colors.append('green')
        elif count >= 3:
            colors.append('steelblue')
        else:
            colors.append('orange')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Set labels
    ax.set_xlabel('Nama Orang', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jumlah Foto', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Jumlah Foto per Orang dalam Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks dengan nama orang (rotasi 90 derajat)
    ax.set_xticks(range(len(photo_counts)))
    ax.set_xticklabels(photo_counts['person_name'], rotation=90, ha='right', fontsize=8)
    
    # Add grid untuk memudahkan membaca
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics text
    stats_text = f"Total: {len(df)} foto | {len(photo_counts)} orang | "
    stats_text += f"Rata-rata: {photo_counts['count'].mean():.1f} foto/orang | "
    stats_text += f"Min: {photo_counts['count'].min()} | Max: {photo_counts['count'].max()}"
    
    ax.text(0.5, 1.05, stats_text, transform=ax.transAxes, 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='≥5 foto'),
        Patch(facecolor='steelblue', edgecolor='black', label='3-4 foto'),
        Patch(facecolor='orange', edgecolor='black', label='≤2 foto')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualisasi disimpan ke: dataset_distribution.png")
    plt.close()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Total foto: {len(df)}")
    print(f"  - Total orang: {len(photo_counts)}")
    print(f"  - Rata-rata foto/orang: {photo_counts['count'].mean():.2f}")
    print(f"  - Std deviasi: {photo_counts['count'].std():.2f}")
    print(f"  - Min foto: {photo_counts['count'].min()}")
    print(f"  - Max foto: {photo_counts['count'].max()}")
    print(f"\nOrang dengan foto paling sedikit:")
    print(photo_counts.nsmallest(5, 'count').to_string(index=False))
    print(f"\nOrang dengan foto paling banyak:")
    print(photo_counts.nlargest(5, 'count').to_string(index=False))