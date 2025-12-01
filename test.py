"""
Test Script untuk FaceNet Model
Ujicoba model dengan input gambar dan evaluasi performa
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import json

from model_facenet import FaceNetModel
from datareader import FaceRecognition


def load_model(model_path, num_classes=None, device='cuda'):
    """Load trained FaceNet model"""
    print(f"📦 Loading model from {model_path}")
    
    # Load checkpoint first to detect num_classes if not provided
    checkpoint = torch.load(model_path, map_location=device)
    
    # Auto-detect num_classes from checkpoint if not provided
    if num_classes is None:
        for key in checkpoint.keys():
            if 'classifier.5.weight' in key:
                num_classes = checkpoint[key].shape[0]
                print(f"🔍 Auto-detected num_classes: {num_classes}")
                break
        if num_classes is None:
            num_classes = 70  # fallback
            print(f"⚠️  Could not detect num_classes, using default: {num_classes}")
    
    model = FaceNetModel(
        num_classes=num_classes,
        pretrained=True,
        device=device
    )
    
    # Filter out facenet.logits
    state_dict = {}
    for key, value in checkpoint.items():
        if 'facenet.logits' not in key:
            state_dict[key] = value
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded successfully with {num_classes} classes")
    return model


def test_single_image(model, image_path, person_mapping, device='cuda'):
    """
    Test model pada satu gambar
    
    Returns:
        dict: {
            'predicted_label': int,
            'predicted_name': str,
            'confidence': float,
            'top5_predictions': list of (label, name, confidence)
        }
    """
    from torchvision import transforms
    
    # Preprocessing (sama seperti training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dan preprocess image
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        embeddings = model.get_embedding(tensor)
        outputs = model.classifier(embeddings)
        probs = torch.softmax(outputs, dim=1)
        
        # Top 1
        confidence, predicted = torch.max(probs, 1)
        pred_label = predicted.item()
        pred_conf = confidence.item()
        
        # Top 5
        top5_conf, top5_idx = torch.topk(probs, min(5, probs.size(1)), dim=1)
        top5_predictions = []
        for i in range(min(5, top5_idx.size(1))):
            idx = top5_idx[0, i].item()
            conf = top5_conf[0, i].item()
            name = person_mapping.get(idx, f"Unknown_{idx}")
            top5_predictions.append((idx, name, conf))
    
    pred_name = person_mapping.get(pred_label, f"Unknown_{pred_label}")
    
    return {
        'predicted_label': pred_label,
        'predicted_name': pred_name,
        'confidence': pred_conf,
        'top5_predictions': top5_predictions
    }


def test_dataset(model, dataset, person_mapping, device='cuda'):
    """
    Test model pada seluruh dataset
    
    Returns:
        dict: {
            'y_true': list,
            'y_pred': list,
            'confidences': list,
            'predictions': list of dicts
        }
    """
    print(f"\n🧪 Testing on {len(dataset)} images...")
    
    model.eval()
    y_true = []
    y_pred = []
    confidences = []
    predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Testing"):
            image, label, _ = dataset[idx]  # dataset returns (image, label, img_name)
            image = image.unsqueeze(0).to(device)
            
            # Predict
            embeddings = model.get_embedding(image)
            outputs = model.classifier(embeddings)
            probs = torch.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probs, 1)
            pred_label = predicted.item()
            pred_conf = confidence.item()
            
            y_true.append(label)
            y_pred.append(pred_label)
            confidences.append(pred_conf)
            
            predictions.append({
                'true_label': label,
                'predicted_label': pred_label,
                'true_name': person_mapping.get(label, f"Unknown_{label}"),
                'predicted_name': person_mapping.get(pred_label, f"Unknown_{pred_label}"),
                'confidence': pred_conf
            })
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences,
        'predictions': predictions
    }


def calculate_metrics(y_true, y_pred, confidences):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    avg_confidence = np.mean(confidences)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': avg_confidence
    }


def plot_confusion_matrix(y_true, y_pred, person_mapping, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Untuk dataset besar, hanya plot top 20 classes
    num_classes = len(np.unique(y_true))
    if num_classes > 20:
        print(f"⚠️  Too many classes ({num_classes}), plotting top 20 most frequent")
        # Get top 20 most frequent classes
        unique, counts = np.unique(y_true, return_counts=True)
        top20_idx = np.argsort(counts)[-20:]
        top20_labels = unique[top20_idx]
        
        # Filter confusion matrix
        mask = np.isin(y_true, top20_labels) & np.isin(y_pred, top20_labels)
        y_true_filtered = np.array(y_true)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top20_labels)
        
        class_names = [person_mapping.get(i, f"Class_{i}")[:15] for i in top20_labels]
    else:
        class_names = [person_mapping.get(i, f"Class_{i}")[:15] for i in range(num_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()


def plot_confidence_distribution(confidences, save_path='confidence_distribution.png'):
    """Plot confidence distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confidence distribution saved to {save_path}")
    plt.close()


def save_results(results, metrics, save_path='test_results.json'):
    """Save test results to JSON"""
    output = {
        'metrics': metrics,
        'num_samples': len(results['y_true']),
        'predictions': results['predictions'][:100]  # Save first 100 predictions
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved to {save_path}")


def print_classification_report(y_true, y_pred, person_mapping):
    """Print detailed classification report"""
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    # Get class names
    unique_labels = sorted(set(y_true + y_pred))
    target_names = [person_mapping.get(i, f"Class_{i}") for i in unique_labels]
    
    # Limit to top 20 for readability
    if len(unique_labels) > 20:
        print("\n⚠️  Showing report for top 20 most frequent classes\n")
        unique, counts = np.unique(y_true, return_counts=True)
        top20_idx = np.argsort(counts)[-20:]
        top20_labels = unique[top20_idx]
        
        # Filter
        mask = np.isin(y_true, top20_labels) & np.isin(y_pred, top20_labels)
        y_true_filtered = np.array(y_true)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        
        # Get actual unique labels in filtered data
        actual_labels = sorted(set(y_true_filtered) | set(y_pred_filtered))
        target_names_filtered = [person_mapping.get(i, f"Class_{i}") for i in actual_labels]
        
        print(classification_report(y_true_filtered, y_pred_filtered, 
                                   labels=actual_labels,
                                   target_names=target_names_filtered,
                                   zero_division=0))
    else:
        print(classification_report(y_true, y_pred, 
                                   target_names=target_names,
                                   zero_division=0))


def main():
    print("="*70)
    print("FACENET MODEL TESTING")
    print("="*70)
    
    # Configuration
    model_path = Path("best_facenet_model_v2.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Device: {device}")
    
    # Load model (auto-detect num_classes)
    model = load_model(model_path, num_classes=None, device=device)
    
    # Get num_classes from model
    num_classes = model.classifier[-1].out_features
    print(f"   Num Classes: {num_classes}")
    
    # Load person mapping
    print(f"\n📖 Loading person mapping...")
    mapping_file = Path("APP/config/person_mapping.json")
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            person_mapping = {int(k): v for k, v in data.items()}
        print(f"✅ Loaded {len(person_mapping)} person names")
    else:
        person_mapping = {i: f"Person_{i}" for i in range(num_classes)}
        print(f"⚠️  Using default mapping")
    
    # Load validation dataset
    print(f"\n📂 Loading validation dataset...")
    val_dataset = FaceRecognition(
        data_dir="Dataset/Train",
        img_size=(224, 224),
        split='val'
    )
    print(f"✅ Loaded {len(val_dataset)} validation images")
    
    # Test on validation set
    results = test_dataset(model, val_dataset, person_mapping, device)
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics...")
    metrics = calculate_metrics(results['y_true'], results['y_pred'], results['confidences'])
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(f"Avg Confidence: {metrics['avg_confidence']:.4f} ({metrics['avg_confidence']*100:.2f}%)")
    print("="*70)
    
    # Print classification report
    print_classification_report(results['y_true'], results['y_pred'], person_mapping)
    
    # Plot confusion matrix
    print(f"\n📈 Generating visualizations...")
    plot_confusion_matrix(results['y_true'], results['y_pred'], person_mapping, 
                         'test_confusion_matrix.png')
    plot_confidence_distribution(results['confidences'], 
                                'test_confidence_distribution.png')
    
    # Save results
    save_results(results, metrics, 'test_results.json')
    
    # Test beberapa sample images
    print(f"\n🖼️  Testing on sample images...")
    print("-"*70)
    
    sample_indices = np.random.choice(len(val_dataset), min(5, len(val_dataset)), replace=False)
    for idx in sample_indices:
        # Get data from dataset
        img_name, true_label = val_dataset.data[idx]  # data is (img_name, label)
        image_path = Path(val_dataset.data_dir) / img_name
        true_name = person_mapping.get(true_label, f"Unknown_{true_label}")
        
        result = test_single_image(model, image_path, person_mapping, device)
        
        print(f"\nImage: {Path(image_path).name}")
        print(f"True:      {true_name} (label {true_label})")
        print(f"Predicted: {result['predicted_name']} (label {result['predicted_label']})")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Correct: {'✅' if result['predicted_label'] == true_label else '❌'}")
        print(f"\nTop 5 Predictions:")
        for i, (label, name, conf) in enumerate(result['top5_predictions'], 1):
            marker = "👉" if label == true_label else "  "
            print(f"  {marker} {i}. {name}: {conf:.4f} ({conf*100:.2f}%)")
    
    print("\n" + "="*70)
    print("✅ Testing completed!")
    print("="*70)
    print(f"\n📁 Generated files:")
    print(f"   - test_confusion_matrix.png")
    print(f"   - test_confidence_distribution.png")
    print(f"   - test_results.json")
    print()


if __name__ == "__main__":
    main()
