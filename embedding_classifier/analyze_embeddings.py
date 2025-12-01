"""
Analisis Embeddings dari FaceNet Model
========================================

Script ini menganalisis kualitas embeddings untuk memahami kenapa
accuracy rendah. Analisis meliputi:
1. Visualisasi embeddings dengan t-SNE/PCA
2. Intra-class vs inter-class distance
3. Cluster separation quality
4. Per-class embedding variance

Output:
- Visualisasi t-SNE dan PCA
- Statistik distance metrics
- Identifikasi kelas yang sulit dibedakan
"""

import sys
from pathlib import Path

# Add parent directory untuk import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from collections import defaultdict


def load_data(data_dir):
    """Load embeddings, labels, dan person mapping"""
    data_dir = Path(data_dir)
    
    print(f"📂 Loading data from: {data_dir}")
    
    # Load embeddings
    train_emb = np.load(data_dir / "embeddings_train.npy")
    train_labels = np.load(data_dir / "labels_train.npy")
    val_emb = np.load(data_dir / "embeddings_val.npy")
    val_labels = np.load(data_dir / "labels_val.npy")
    
    # Load person mapping
    mapping_file = data_dir / "person_names.json"
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            person_mapping = {int(k): v for k, v in data.items()}
    else:
        person_mapping = {}
    
    print(f"✅ Train: {train_emb.shape}")
    print(f"✅ Val: {val_emb.shape}")
    print(f"✅ Person mapping: {len(person_mapping)} persons")
    
    return train_emb, train_labels, val_emb, val_labels, person_mapping


def analyze_embedding_statistics(embeddings, labels, split_name="Train"):
    """Analisis statistik dasar embeddings"""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} EMBEDDING STATISTICS")
    print(f"{'='*70}")
    
    print(f"Shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of samples: {embeddings.shape[0]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Basic statistics
    print(f"\nValue statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # L2 norm statistics (penting untuk embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 Norm statistics:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std: {norms.std():.6f}")
    print(f"  Min: {norms.min():.6f}")
    print(f"  Max: {norms.max():.6f}")
    
    return norms


def calculate_intra_inter_class_distances(embeddings, labels):
    """
    Calculate intra-class dan inter-class distances
    
    Intra-class: jarak antar embedding dalam kelas yang sama (harusnya kecil)
    Inter-class: jarak antar embedding dari kelas berbeda (harusnya besar)
    """
    print(f"\n{'='*70}")
    print("INTRA-CLASS vs INTER-CLASS DISTANCE ANALYSIS")
    print(f"{'='*70}")
    
    unique_labels = np.unique(labels)
    
    intra_class_distances = []
    inter_class_distances = []
    
    # Calculate pairwise distances
    dist_matrix = euclidean_distances(embeddings)
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = dist_matrix[i, j]
            
            if labels[i] == labels[j]:
                # Same class - intra-class
                intra_class_distances.append(dist)
            else:
                # Different class - inter-class
                inter_class_distances.append(dist)
    
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)
    
    print(f"\nIntra-class distances (same person):")
    print(f"  Count: {len(intra_class_distances)}")
    print(f"  Mean: {intra_class_distances.mean():.6f}")
    print(f"  Std: {intra_class_distances.std():.6f}")
    print(f"  Min: {intra_class_distances.min():.6f}")
    print(f"  Max: {intra_class_distances.max():.6f}")
    
    print(f"\nInter-class distances (different persons):")
    print(f"  Count: {len(inter_class_distances)}")
    print(f"  Mean: {inter_class_distances.mean():.6f}")
    print(f"  Std: {inter_class_distances.std():.6f}")
    print(f"  Min: {inter_class_distances.min():.6f}")
    print(f"  Max: {inter_class_distances.max():.6f}")
    
    # Separation ratio (higher is better)
    separation_ratio = inter_class_distances.mean() / intra_class_distances.mean()
    print(f"\n📊 Separation Ratio (inter/intra): {separation_ratio:.4f}")
    
    if separation_ratio < 1.2:
        print(f"   ⚠️  POOR - Classes overlap significantly!")
        print(f"   💡 Embeddings are NOT well-separated")
    elif separation_ratio < 1.5:
        print(f"   ⚠️  FAIR - Some class separation")
    elif separation_ratio < 2.0:
        print(f"   ✅ GOOD - Classes are reasonably separated")
    else:
        print(f"   ✅ EXCELLENT - Classes are well-separated")
    
    return intra_class_distances, inter_class_distances


def plot_distance_distributions(intra_dists, inter_dists, save_path):
    """Plot intra-class vs inter-class distance distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overlapping histograms
    ax1 = axes[0]
    ax1.hist(intra_dists, bins=50, alpha=0.6, label='Intra-class (same person)', 
             color='blue', edgecolor='black')
    ax1.hist(inter_dists, bins=50, alpha=0.6, label='Inter-class (diff person)', 
             color='red', edgecolor='black')
    ax1.axvline(intra_dists.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Intra mean: {intra_dists.mean():.3f}')
    ax1.axvline(inter_dists.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Inter mean: {inter_dists.mean():.3f}')
    ax1.set_xlabel('Euclidean Distance', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distance Distributions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    ax2 = axes[1]
    box_data = [intra_dists, inter_dists]
    bp = ax2.boxplot(box_data, labels=['Intra-class', 'Inter-class'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Euclidean Distance', fontsize=12)
    ax2.set_title('Distance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def analyze_per_class_variance(embeddings, labels, person_mapping):
    """Analisis variance per class"""
    print(f"\n{'='*70}")
    print("PER-CLASS EMBEDDING VARIANCE ANALYSIS")
    print(f"{'='*70}")
    
    unique_labels = np.unique(labels)
    
    class_stats = []
    
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) < 2:
            continue
        
        # Calculate variance
        variance = np.var(class_embeddings, axis=0).mean()
        std = np.std(class_embeddings, axis=0).mean()
        
        # Calculate pairwise distances within class
        if len(class_embeddings) > 1:
            pairwise_dists = pdist(class_embeddings, metric='euclidean')
            mean_dist = pairwise_dists.mean()
        else:
            mean_dist = 0
        
        class_stats.append({
            'label': int(label),
            'name': person_mapping.get(int(label), f"Person_{label}"),
            'num_samples': len(class_embeddings),
            'variance': float(variance),
            'std': float(std),
            'mean_intra_dist': float(mean_dist)
        })
    
    # Sort by variance (descending)
    class_stats_sorted = sorted(class_stats, key=lambda x: x['variance'], reverse=True)
    
    print(f"\nTop 10 classes with HIGHEST variance (most scattered):")
    print(f"{'Rank':<6} {'Label':<7} {'Samples':<9} {'Variance':<12} {'Intra-dist':<12} {'Name':<30}")
    print("-" * 80)
    for i, stat in enumerate(class_stats_sorted[:10], 1):
        print(f"{i:<6} {stat['label']:<7} {stat['num_samples']:<9} "
              f"{stat['variance']:<12.6f} {stat['mean_intra_dist']:<12.6f} "
              f"{stat['name'][:28]:<30}")
    
    print(f"\nTop 10 classes with LOWEST variance (most compact):")
    print(f"{'Rank':<6} {'Label':<7} {'Samples':<9} {'Variance':<12} {'Intra-dist':<12} {'Name':<30}")
    print("-" * 80)
    for i, stat in enumerate(reversed(class_stats_sorted[-10:]), 1):
        print(f"{i:<6} {stat['label']:<7} {stat['num_samples']:<9} "
              f"{stat['variance']:<12.6f} {stat['mean_intra_dist']:<12.6f} "
              f"{stat['name'][:28]:<30}")
    
    return class_stats


def visualize_embeddings_tsne(embeddings, labels, person_mapping, save_path, n_samples=20):
    """Visualize embeddings using t-SNE (sample top N classes)"""
    print(f"\n🎨 Creating t-SNE visualization...")
    
    # Get top N most frequent classes for clearer visualization
    unique, counts = np.unique(labels, return_counts=True)
    top_n_idx = np.argsort(counts)[-n_samples:]
    top_n_labels = unique[top_n_idx]
    
    # Filter to top N classes
    mask = np.isin(labels, top_n_labels)
    embeddings_subset = embeddings[mask]
    labels_subset = labels[mask]
    
    print(f"   Using {len(embeddings_subset)} samples from {n_samples} most frequent classes")
    
    # Apply t-SNE
    print(f"   Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_subset)-1))
    embeddings_2d = tsne.fit_transform(embeddings_subset)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create colormap
    unique_labels_subset = np.unique(labels_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_subset)))
    
    for i, label in enumerate(unique_labels_subset):
        mask = labels_subset == label
        name = person_mapping.get(int(label), f"Person_{label}")[:20]
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], label=f"{name}", alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f't-SNE Visualization of Embeddings (Top {n_samples} Classes)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def visualize_embeddings_pca(embeddings, labels, person_mapping, save_path, n_samples=20):
    """Visualize embeddings using PCA"""
    print(f"\n🎨 Creating PCA visualization...")
    
    # Get top N most frequent classes
    unique, counts = np.unique(labels, return_counts=True)
    top_n_idx = np.argsort(counts)[-n_samples:]
    top_n_labels = unique[top_n_idx]
    
    # Filter
    mask = np.isin(labels, top_n_labels)
    embeddings_subset = embeddings[mask]
    labels_subset = labels[mask]
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_subset)
    
    print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_labels_subset = np.unique(labels_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_subset)))
    
    for i, label in enumerate(unique_labels_subset):
        mask = labels_subset == label
        name = person_mapping.get(int(label), f"Person_{label}")[:20]
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], label=f"{name}", alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title(f'PCA Visualization of Embeddings (Top {n_samples} Classes)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def save_analysis_report(stats, save_path):
    """Save analysis report to JSON"""
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved analysis report: {save_path}")


def main():
    data_dir = parent_dir / "embedding_classifier/data"
    output_dir = parent_dir / "embedding_classifier/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FACENET EMBEDDING ANALYSIS")
    print("="*70)
    
    # Load data
    train_emb, train_labels, val_emb, val_labels, person_mapping = load_data(data_dir)
    
    # Combine for overall analysis
    all_emb = np.vstack([train_emb, val_emb])
    all_labels = np.concatenate([train_labels, val_labels])
    
    # 1. Basic statistics
    print(f"\n{'='*70}")
    print("STEP 1: BASIC STATISTICS")
    print(f"{'='*70}")
    train_norms = analyze_embedding_statistics(train_emb, train_labels, "Train")
    val_norms = analyze_embedding_statistics(val_emb, val_labels, "Val")
    
    # 2. Intra vs Inter class distances
    print(f"\n{'='*70}")
    print("STEP 2: DISTANCE ANALYSIS")
    print(f"{'='*70}")
    intra_dists, inter_dists = calculate_intra_inter_class_distances(all_emb, all_labels)
    
    # Plot distance distributions
    plot_distance_distributions(intra_dists, inter_dists, 
                               output_dir / "distance_distributions.png")
    
    # 3. Per-class variance
    print(f"\n{'='*70}")
    print("STEP 3: PER-CLASS ANALYSIS")
    print(f"{'='*70}")
    class_stats = analyze_per_class_variance(all_emb, all_labels, person_mapping)
    
    # 4. Visualizations
    print(f"\n{'='*70}")
    print("STEP 4: VISUALIZATIONS")
    print(f"{'='*70}")
    
    visualize_embeddings_tsne(all_emb, all_labels, person_mapping,
                             output_dir / "embeddings_tsne.png", n_samples=20)
    
    visualize_embeddings_pca(all_emb, all_labels, person_mapping,
                            output_dir / "embeddings_pca.png", n_samples=20)
    
    # 5. Save report
    print(f"\n{'='*70}")
    print("STEP 5: GENERATING REPORT")
    print(f"{'='*70}")
    
    analysis_report = {
        'embedding_stats': {
            'dimension': int(all_emb.shape[1]),
            'num_samples': int(len(all_emb)),
            'num_classes': int(len(np.unique(all_labels))),
            'mean': float(all_emb.mean()),
            'std': float(all_emb.std()),
        },
        'distance_stats': {
            'intra_class_mean': float(intra_dists.mean()),
            'intra_class_std': float(intra_dists.std()),
            'inter_class_mean': float(inter_dists.mean()),
            'inter_class_std': float(inter_dists.std()),
            'separation_ratio': float(inter_dists.mean() / intra_dists.mean()),
        },
        'per_class_stats': class_stats
    }
    
    save_analysis_report(analysis_report, output_dir / "embedding_analysis_report.json")
    
    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    sep_ratio = inter_dists.mean() / intra_dists.mean()
    print(f"\n📊 Key Metrics:")
    print(f"   Separation Ratio: {sep_ratio:.4f}")
    print(f"   Intra-class distance: {intra_dists.mean():.6f} ± {intra_dists.std():.6f}")
    print(f"   Inter-class distance: {inter_dists.mean():.6f} ± {inter_dists.std():.6f}")
    
    print(f"\n💡 Diagnosis:")
    if sep_ratio < 1.2:
        print(f"   ❌ PROBLEM: Embeddings are poorly separated!")
        print(f"   💡 Recommendation: Re-train FaceNet with:")
        print(f"      - Triplet loss instead of softmax")
        print(f"      - More data augmentation")
        print(f"      - Longer training")
    elif sep_ratio < 1.5:
        print(f"   ⚠️  FAIR: Embeddings have some separation")
        print(f"   💡 Recommendation: Fine-tune more or use better loss function")
    else:
        print(f"   ✅ GOOD: Embeddings are reasonably separated")
    
    print(f"\n📁 Output files in: {output_dir}")
    print(f"   - embedding_analysis_report.json")
    print(f"   - distance_distributions.png")
    print(f"   - embeddings_tsne.png")
    print(f"   - embeddings_pca.png")
    
    print(f"\n{'='*70}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Train SVM: python embedding_classifier/train_svm.py")
    print(f"   2. If separation ratio < 1.5, consider re-training FaceNet")


if __name__ == "__main__":
    main()
