"""
Extract Embeddings dari FaceNet Model
=====================================

Script ini mengekstrak embedding 512-dimensional dari semua gambar training dan validation
menggunakan model FaceNet yang sudah dilatih. Embedding ini akan digunakan untuk
training classifier KNN/SVM.

Output:
- embeddings_train.npy: Embedding untuk training set
- labels_train.npy: Label untuk training set
- embeddings_val.npy: Embedding untuk validation set
- labels_val.npy: Label untuk validation set
- person_names.json: Mapping dari label ID ke nama orang
"""

import sys
from pathlib import Path

# Add parent directory untuk import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

from model_facenet import FaceNetModel
from datareader import FaceRecognition


def load_facenet_model(model_path, num_classes, device):
    """
    Load FaceNet model dari checkpoint
    
    Args:
        model_path: Path ke model checkpoint (.pth)
        num_classes: Jumlah kelas (70)
        device: torch.device
    
    Returns:
        model: FaceNet model yang sudah di-load
    """
    print(f"📦 Loading FaceNet model from: {model_path}")
    
    # Load checkpoint untuk auto-detect num_classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Auto-detect num_classes dari checkpoint
    detected_classes = None
    for key in checkpoint.keys():
        if 'classifier.5.weight' in key:
            detected_classes = checkpoint[key].shape[0]
            break
    
    if detected_classes is not None:
        num_classes = detected_classes
        print(f"   Auto-detected num_classes: {num_classes}")
    
    # Create model
    model = FaceNetModel(
        num_classes=num_classes,
        pretrained=True,
        device=device
    )
    
    # Load state dict (skip facenet.logits)
    state_dict = {}
    for key, value in checkpoint.items():
        if 'facenet.logits' not in key:
            state_dict[key] = value
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded successfully")
    print(f"   Num classes: {num_classes}")
    print(f"   Embedding size: 512")
    
    return model, num_classes


def extract_embeddings(model, dataloader, device, desc="Extracting"):
    """
    Ekstrak embeddings dari dataset
    
    Args:
        model: FaceNet model
        dataloader: DataLoader untuk dataset
        device: torch.device
        desc: Deskripsi untuk progress bar
    
    Returns:
        embeddings: numpy array [N, 512]
        labels: numpy array [N]
        filenames: list of filenames
    """
    all_embeddings = []
    all_labels = []
    all_filenames = []
    
    model.eval()
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc=desc):
            # Move to device
            images = images.to(device)
            
            # Extract embeddings (512-dimensional)
            embeddings = model.get_embedding(images)  # [batch_size, 512]
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            
            # Append
            all_embeddings.append(embeddings_np)
            all_labels.append(labels_np)
            
            # Handle filenames (bisa tuple atau list)
            if isinstance(filenames, (list, tuple)):
                all_filenames.extend(filenames)
            else:
                all_filenames.append(filenames)
    
    # Concatenate
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels, all_filenames


def save_embeddings(embeddings, labels, filenames, output_dir, prefix):
    """
    Save embeddings, labels, dan filenames ke file
    
    Args:
        embeddings: numpy array [N, 512]
        labels: numpy array [N]
        filenames: list of filenames
        output_dir: Directory untuk menyimpan
        prefix: 'train' atau 'val'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    emb_path = output_dir / f"embeddings_{prefix}.npy"
    np.save(emb_path, embeddings)
    print(f"✅ Saved embeddings to: {emb_path}")
    print(f"   Shape: {embeddings.shape}")
    
    # Save labels
    labels_path = output_dir / f"labels_{prefix}.npy"
    np.save(labels_path, labels)
    print(f"✅ Saved labels to: {labels_path}")
    print(f"   Shape: {labels.shape}")
    
    # Save filenames
    filenames_path = output_dir / f"filenames_{prefix}.json"
    with open(filenames_path, 'w') as f:
        json.dump(filenames, f, indent=2)
    print(f"✅ Saved filenames to: {filenames_path}")
    print(f"   Count: {len(filenames)}")


def create_person_mapping(dataset, output_dir):
    """
    Buat mapping dari label ID ke person name
    
    Args:
        dataset: FaceRecognition dataset
        output_dir: Directory untuk save mapping
    """
    # Get all unique labels and their corresponding names
    label_to_name = {}
    
    # Read from label_mapping.csv
    csv_file = Path(parent_dir) / "Dataset" / "label_mapping.csv"
    
    if csv_file.exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        # Check which column name is used
        if 'label_id' in df.columns:
            label_to_name = dict(zip(df['label_id'], df['person_name']))
        elif 'label' in df.columns:
            label_to_name = dict(zip(df['label'], df['person_name']))
        else:
            # Fallback: use index as label
            label_to_name = dict(enumerate(df['person_name']))
    else:
        # Fallback: extract from dataset
        all_labels = set()
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            all_labels.add(label)
        
        label_to_name = {i: f"Person_{i}" for i in sorted(all_labels)}
    
    # Save to JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_path = output_dir / "person_names.json"
    with open(mapping_path, 'w') as f:
        json.dump(label_to_name, f, indent=2)
    
    print(f"✅ Saved person mapping to: {mapping_path}")
    print(f"   Total persons: {len(label_to_name)}")
    
    return label_to_name


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from FaceNet model')
    parser.add_argument('--model', type=str, default='best_facenet_model.pth',
                       help='Path to FaceNet model checkpoint')
    parser.add_argument('--data-dir', type=str, default='Dataset/Train',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='embedding_classifier/data',
                       help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for extraction')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths (relative to parent_dir)
    if not Path(args.data_dir).is_absolute():
        args.data_dir = str(parent_dir / args.data_dir)
    if not Path(args.output_dir).is_absolute():
        args.output_dir = str(parent_dir / args.output_dir)
    
    print("="*70)
    print("FACENET EMBEDDING EXTRACTION")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Model: {args.model}")
    print(f"   Data dir: {args.data_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Batch size: {args.batch_size}")
    
    # Load model
    model_path = Path(parent_dir) / args.model
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found: {model_path}")
        print(f"   Available models:")
        for p in Path(parent_dir).glob("*.pth"):
            print(f"   - {p.name}")
        return
    
    model, num_classes = load_facenet_model(model_path, num_classes=70, device=device)
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_dataset = FaceRecognition(
        data_dir=args.data_dir,
        img_size=(224, 224),
        split='train'
    )
    val_dataset = FaceRecognition(
        data_dir=args.data_dir,
        img_size=(224, 224),
        split='val'
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} images")
    print(f"✅ Val dataset: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Don't shuffle untuk maintain order
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Extract train embeddings
    print(f"\n🔄 Extracting training embeddings...")
    train_embeddings, train_labels, train_filenames = extract_embeddings(
        model, train_loader, device, desc="Train"
    )
    
    # Extract val embeddings
    print(f"\n🔄 Extracting validation embeddings...")
    val_embeddings, val_labels, val_filenames = extract_embeddings(
        model, val_loader, device, desc="Val"
    )
    
    # Save embeddings
    print(f"\n💾 Saving embeddings...")
    output_dir = Path(args.output_dir)
    save_embeddings(train_embeddings, train_labels, train_filenames, output_dir, 'train')
    save_embeddings(val_embeddings, val_labels, val_filenames, output_dir, 'val')
    
    # Create person mapping
    print(f"\n📖 Creating person mapping...")
    person_mapping = create_person_mapping(train_dataset, output_dir)
    
    # Summary
    print(f"\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Train labels: {train_labels.shape}")
    print(f"Val embeddings: {val_embeddings.shape}")
    print(f"Val labels: {val_labels.shape}")
    print(f"Embedding dimension: 512")
    print(f"Number of classes: {num_classes}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n✅ Embedding extraction completed successfully!")
    print("="*70)
    
    # Print stats
    print(f"\n📊 Statistics:")
    print(f"   Train - Mean: {train_embeddings.mean():.4f}, Std: {train_embeddings.std():.4f}")
    print(f"   Train - Min: {train_embeddings.min():.4f}, Max: {train_embeddings.max():.4f}")
    print(f"   Val - Mean: {val_embeddings.mean():.4f}, Std: {val_embeddings.std():.4f}")
    print(f"   Val - Min: {val_embeddings.min():.4f}, Max: {val_embeddings.max():.4f}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Train KNN classifier: python embedding_classifier/train_knn.py")
    print(f"   2. Train SVM classifier: python embedding_classifier/train_svm.py")
    print(f"   3. Compare results: python embedding_classifier/compare_classifiers.py")


if __name__ == "__main__":
    main()
