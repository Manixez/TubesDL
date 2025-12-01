"""
Train K-Nearest Neighbors (KNN) Classifier pada FaceNet Embeddings
====================================================================

Script ini melatih KNN classifier menggunakan embeddings yang sudah diekstrak
dari FaceNet model. KNN sangat cocok untuk dataset kecil karena:
- Tidak memerlukan training yang lama
- Tidak ada parameter yang perlu dilatih
- Bekerja dengan baik pada high-dimensional embeddings

Output:
- knn_classifier.pkl: Model KNN yang sudah dilatih
- knn_results.json: Hasil evaluasi pada validation set
"""

import sys
from pathlib import Path

# Add parent directory untuk import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import argparse
from datetime import datetime


def load_embeddings(data_dir):
    """
    Load embeddings dan labels dari file .npy
    
    Args:
        data_dir: Directory containing embeddings files
    
    Returns:
        train_embeddings, train_labels, val_embeddings, val_labels
    """
    data_dir = Path(data_dir)
    
    print(f"📂 Loading embeddings from: {data_dir}")
    
    # Load training data
    train_emb = np.load(data_dir / "embeddings_train.npy")
    train_labels = np.load(data_dir / "labels_train.npy")
    
    # Load validation data
    val_emb = np.load(data_dir / "embeddings_val.npy")
    val_labels = np.load(data_dir / "labels_val.npy")
    
    print(f"✅ Train embeddings: {train_emb.shape}")
    print(f"✅ Train labels: {train_labels.shape}")
    print(f"✅ Val embeddings: {val_emb.shape}")
    print(f"✅ Val labels: {val_labels.shape}")
    
    return train_emb, train_labels, val_emb, val_labels


def load_person_mapping(data_dir):
    """Load person name mapping"""
    mapping_file = Path(data_dir) / "person_names.json"
    
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            # Convert string keys to int
            person_mapping = {int(k): v for k, v in data.items()}
        print(f"✅ Loaded person mapping: {len(person_mapping)} persons")
        return person_mapping
    else:
        print(f"⚠️  Person mapping not found, using default")
        return {}


def train_knn(train_embeddings, train_labels, n_neighbors=5, metric='euclidean', weights='distance'):
    """
    Train KNN classifier
    
    Args:
        train_embeddings: Training embeddings [N, 512]
        train_labels: Training labels [N]
        n_neighbors: Number of neighbors (default: 5)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        weights: 'uniform' atau 'distance'
    
    Returns:
        knn: Trained KNN classifier
    """
    print(f"\n🔨 Training KNN Classifier...")
    print(f"   n_neighbors: {n_neighbors}")
    print(f"   metric: {metric}")
    print(f"   weights: {weights}")
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        algorithm='auto',
        n_jobs=-1  # Use all CPU cores
    )
    
    knn.fit(train_embeddings, train_labels)
    
    print(f"✅ KNN training completed!")
    
    return knn


def evaluate_knn(knn, val_embeddings, val_labels, person_mapping):
    """
    Evaluate KNN classifier pada validation set
    
    Args:
        knn: Trained KNN classifier
        val_embeddings: Validation embeddings
        val_labels: True labels
        person_mapping: Label to name mapping
    
    Returns:
        results: Dictionary dengan metrics dan predictions
    """
    print(f"\n📊 Evaluating on validation set...")
    
    # Predict
    y_pred = knn.predict(val_embeddings)
    
    # Predict probabilities (berdasarkan voting neighbors)
    y_proba = knn.predict_proba(val_embeddings)
    confidences = np.max(y_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, y_pred)
    precision = precision_score(val_labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(val_labels, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(val_labels, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*70}")
    print(f"KNN VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"Avg Confidence: {confidences.mean():.4f} ({confidences.mean()*100:.2f}%)")
    print(f"{'='*70}")
    
    # Detailed predictions
    predictions = []
    for i, (true_label, pred_label, conf) in enumerate(zip(val_labels, y_pred, confidences)):
        predictions.append({
            'sample_idx': int(i),
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'true_name': person_mapping.get(int(true_label), f"Person_{true_label}"),
            'predicted_name': person_mapping.get(int(pred_label), f"Person_{pred_label}"),
            'confidence': float(conf),
            'correct': bool(true_label == pred_label)
        })
    
    results = {
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_confidence': float(confidences.mean())
        },
        'num_samples': len(val_labels),
        'num_correct': int((val_labels == y_pred).sum()),
        'predictions': predictions
    }
    
    return results, y_pred, confidences


def plot_confusion_matrix(y_true, y_pred, person_mapping, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    num_classes = len(np.unique(y_true))
    
    # Jika terlalu banyak kelas, plot top 20
    if num_classes > 20:
        print(f"\n⚠️  {num_classes} classes detected, plotting top 20 most frequent")
        unique, counts = np.unique(y_true, return_counts=True)
        top20_idx = np.argsort(counts)[-20:]
        top20_labels = unique[top20_idx]
        
        # Filter
        mask = np.isin(y_true, top20_labels) & np.isin(y_pred, top20_labels)
        y_true_filtered = np.array(y_true)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top20_labels)
        
        class_names = [person_mapping.get(int(i), f"Class_{i}")[:15] for i in top20_labels]
    else:
        class_names = [person_mapping.get(int(i), f"Class_{i}")[:15] for i in range(num_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('KNN Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {save_path}")
    plt.close()


def plot_confidence_distribution(confidences, save_path):
    """Plot confidence distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(confidences.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {confidences.mean():.3f}')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('KNN Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confidence distribution saved to: {save_path}")
    plt.close()


def save_model(knn, save_path):
    """Save KNN model to pickle file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(knn, f)
    
    print(f"✅ KNN model saved to: {save_path}")


def save_results(results, save_path):
    """Save results to JSON"""
    save_path = Path(save_path)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train KNN classifier on FaceNet embeddings')
    parser.add_argument('--data-dir', type=str, default='embedding_classifier/data',
                       help='Directory containing embeddings')
    parser.add_argument('--output-dir', type=str, default='embedding_classifier/models',
                       help='Output directory for models')
    parser.add_argument('--n-neighbors', type=int, default=5,
                       help='Number of neighbors for KNN')
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'manhattan'],
                       help='Distance metric')
    parser.add_argument('--weights', type=str, default='distance',
                       choices=['uniform', 'distance'],
                       help='Weight function')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    if not Path(args.data_dir).is_absolute():
        args.data_dir = str(parent_dir / args.data_dir)
    if not Path(args.output_dir).is_absolute():
        args.output_dir = str(parent_dir / args.output_dir)
    
    print("="*70)
    print("KNN CLASSIFIER TRAINING")
    print("="*70)
    
    print(f"\n📋 Configuration:")
    print(f"   Data dir: {args.data_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   n_neighbors: {args.n_neighbors}")
    print(f"   metric: {args.metric}")
    print(f"   weights: {args.weights}")
    
    # Load embeddings
    train_emb, train_labels, val_emb, val_labels = load_embeddings(args.data_dir)
    
    # Load person mapping
    person_mapping = load_person_mapping(args.data_dir)
    
    # Train KNN
    knn = train_knn(train_emb, train_labels, 
                    n_neighbors=args.n_neighbors,
                    metric=args.metric,
                    weights=args.weights)
    
    # Evaluate
    results, y_pred, confidences = evaluate_knn(knn, val_emb, val_labels, person_mapping)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "knn_classifier.pkl"
    save_model(knn, model_path)
    
    # Save results
    results_path = output_dir / "knn_results.json"
    save_results(results, results_path)
    
    # Plot confusion matrix
    cm_path = output_dir / "knn_confusion_matrix.png"
    plot_confusion_matrix(val_labels, y_pred, person_mapping, cm_path)
    
    # Plot confidence distribution
    conf_path = output_dir / "knn_confidence_distribution.png"
    plot_confidence_distribution(confidences, conf_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Best Accuracy: {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {results['num_correct']}/{results['num_samples']}")
    print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"\n📁 Output files:")
    print(f"   - {model_path}")
    print(f"   - {results_path}")
    print(f"   - {cm_path}")
    print(f"   - {conf_path}")
    print(f"{'='*70}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Try different k values: --n-neighbors 3 or --n-neighbors 7")
    print(f"   2. Try cosine metric: --metric cosine")
    print(f"   3. Train SVM: python embedding_classifier/train_svm.py")
    print(f"   4. Compare classifiers: python embedding_classifier/compare_classifiers.py")


if __name__ == "__main__":
    main()
