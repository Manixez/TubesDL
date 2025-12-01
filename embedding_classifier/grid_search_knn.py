"""
Grid Search untuk KNN - Mencari hyperparameter terbaik
========================================================

Script ini akan mencoba berbagai kombinasi hyperparameter KNN:
- n_neighbors: [1, 3, 5, 7, 9]
- metric: ['euclidean', 'cosine', 'manhattan']
- weights: ['uniform', 'distance']

Dan memilih kombinasi terbaik berdasarkan validation accuracy.
"""

import sys
from pathlib import Path

# Add parent directory untuk import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
import pickle
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_embeddings(data_dir):
    """Load embeddings dan labels"""
    data_dir = Path(data_dir)
    
    train_emb = np.load(data_dir / "embeddings_train.npy")
    train_labels = np.load(data_dir / "labels_train.npy")
    val_emb = np.load(data_dir / "embeddings_val.npy")
    val_labels = np.load(data_dir / "labels_val.npy")
    
    return train_emb, train_labels, val_emb, val_labels


def load_person_mapping(data_dir):
    """Load person mapping"""
    mapping_file = Path(data_dir) / "person_names.json"
    
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    return {}


def grid_search_knn(train_emb, train_labels, val_emb, val_labels):
    """
    Grid search untuk KNN hyperparameters
    
    Returns:
        results: List of dictionaries dengan results untuk setiap kombinasi
        best_params: Best hyperparameters
    """
    # Define parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'metric': ['euclidean', 'cosine', 'manhattan'],
        'weights': ['uniform', 'distance']
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"\n🔍 Grid Search Configuration:")
    print(f"   n_neighbors: {param_grid['n_neighbors']}")
    print(f"   metric: {param_grid['metric']}")
    print(f"   weights: {param_grid['weights']}")
    print(f"   Total combinations: {len(combinations)}")
    
    results = []
    best_accuracy = 0
    best_params = None
    
    print(f"\n🚀 Starting grid search...")
    
    for params in tqdm(combinations, desc="Grid Search"):
        # Train KNN
        knn = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            metric=params['metric'],
            weights=params['weights'],
            algorithm='auto',
            n_jobs=-1
        )
        
        knn.fit(train_emb, train_labels)
        
        # Predict
        y_pred = knn.predict(val_emb)
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, y_pred)
        f1 = f1_score(val_labels, y_pred, average='weighted', zero_division=0)
        
        # Store results
        result = {
            'params': params.copy(),
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
        results.append(result)
        
        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params.copy()
    
    return results, best_params, best_accuracy


def plot_grid_search_results(results, save_dir):
    """Plot grid search results"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to structured format
    k_values = sorted(set(r['params']['n_neighbors'] for r in results))
    metrics = sorted(set(r['params']['metric'] for r in results))
    weights = sorted(set(r['params']['weights'] for r in results))
    
    # Plot 1: Accuracy vs K for different metrics (distance weights)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for weight_type in weights:
        ax = axes[0] if weight_type == 'distance' else axes[1]
        
        for metric in metrics:
            accs = []
            for k in k_values:
                matching = [r for r in results 
                           if r['params']['n_neighbors'] == k 
                           and r['params']['metric'] == metric
                           and r['params']['weights'] == weight_type]
                if matching:
                    accs.append(matching[0]['accuracy'])
                else:
                    accs.append(0)
            
            ax.plot(k_values, accs, marker='o', label=metric, linewidth=2)
        
        ax.set_xlabel('Number of Neighbors (K)', fontsize=12)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title(f'KNN Accuracy vs K (weights={weight_type})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'knn_grid_search_k_vs_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: knn_grid_search_k_vs_accuracy.png")
    plt.close()
    
    # Plot 2: Heatmap for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        # Create matrix: rows = k values, cols = weights
        matrix = np.zeros((len(k_values), len(weights)))
        
        for i, k in enumerate(k_values):
            for j, weight in enumerate(weights):
                matching = [r for r in results 
                           if r['params']['n_neighbors'] == k 
                           and r['params']['metric'] == metric
                           and r['params']['weights'] == weight]
                if matching:
                    matrix[i, j] = matching[0]['accuracy']
        
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=weights, yticklabels=k_values,
                   ax=axes[idx], cbar_kws={'label': 'Accuracy'})
        axes[idx].set_title(f'Metric: {metric}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Weights')
        axes[idx].set_ylabel('K (neighbors)')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'knn_grid_search_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: knn_grid_search_heatmap.png")
    plt.close()
    
    # Plot 3: Top 10 configurations
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"K={r['params']['n_neighbors']}, {r['params']['metric'][:3]}, {r['params']['weights'][:3]}" 
              for r in sorted_results]
    accuracies = [r['accuracy'] for r in sorted_results]
    
    bars = ax.barh(range(len(labels)), accuracies, color='steelblue')
    
    # Color best bar
    bars[0].set_color('green')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Validation Accuracy', fontsize=12)
    ax.set_title('Top 10 KNN Configurations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'knn_top10_configs.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: knn_top10_configs.png")
    plt.close()


def train_best_knn(train_emb, train_labels, val_emb, val_labels, best_params):
    """Train KNN dengan best parameters"""
    print(f"\n🏆 Training KNN with best parameters...")
    print(f"   n_neighbors: {best_params['n_neighbors']}")
    print(f"   metric: {best_params['metric']}")
    print(f"   weights: {best_params['weights']}")
    
    knn = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        metric=best_params['metric'],
        weights=best_params['weights'],
        algorithm='auto',
        n_jobs=-1
    )
    
    knn.fit(train_emb, train_labels)
    
    # Evaluate
    y_pred = knn.predict(val_emb)
    y_proba = knn.predict_proba(val_emb)
    confidences = np.max(y_proba, axis=1)
    
    accuracy = accuracy_score(val_labels, y_pred)
    f1 = f1_score(val_labels, y_pred, average='weighted', zero_division=0)
    
    print(f"\n✅ Best KNN Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Avg Confidence: {confidences.mean():.4f}")
    
    return knn, accuracy, f1


def main():
    data_dir = parent_dir / "embedding_classifier/data"
    output_dir = parent_dir / "embedding_classifier/models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("KNN GRID SEARCH - AUTOMATIC HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading embeddings from: {data_dir}")
    train_emb, train_labels, val_emb, val_labels = load_embeddings(data_dir)
    
    print(f"✅ Train: {train_emb.shape}")
    print(f"✅ Val: {val_emb.shape}")
    
    # Load person mapping
    person_mapping = load_person_mapping(data_dir)
    
    # Grid search
    results, best_params, best_accuracy = grid_search_knn(
        train_emb, train_labels, val_emb, val_labels
    )
    
    # Print top 10 results
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]
    
    for i, r in enumerate(sorted_results, 1):
        params = r['params']
        marker = "🏆" if i == 1 else f"{i:2d}."
        print(f"{marker} K={params['n_neighbors']:2d}, {params['metric']:10s}, {params['weights']:8s} "
              f"→ Acc: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%), F1: {r['f1_score']:.4f}")
    
    print(f"\n{'='*70}")
    print("BEST PARAMETERS")
    print(f"{'='*70}")
    print(f"n_neighbors: {best_params['n_neighbors']}")
    print(f"metric: {best_params['metric']}")
    print(f"weights: {best_params['weights']}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Plot results
    print(f"\n📊 Generating visualizations...")
    plot_grid_search_results(results, output_dir)
    
    # Train final model with best params
    best_knn, final_acc, final_f1 = train_best_knn(
        train_emb, train_labels, val_emb, val_labels, best_params
    )
    
    # Save best model
    model_path = output_dir / "knn_best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_knn, f)
    print(f"\n💾 Best model saved to: {model_path}")
    
    # Save grid search results
    results_path = output_dir / "knn_grid_search_results.json"
    output_data = {
        'best_params': best_params,
        'best_accuracy': float(best_accuracy),
        'all_results': results,
        'top_10': sorted_results
    }
    
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"💾 Grid search results saved to: {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*70}")
    print(f"Total configurations tested: {len(results)}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"\n📁 Output files:")
    print(f"   - {model_path}")
    print(f"   - {results_path}")
    print(f"   - {output_dir}/knn_grid_search_*.png")
    print(f"{'='*70}")
    
    print(f"\n💡 Next step:")
    print(f"   Train SVM: python embedding_classifier/train_svm.py")


if __name__ == "__main__":
    main()
