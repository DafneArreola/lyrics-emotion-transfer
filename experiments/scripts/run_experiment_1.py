#!/usr/bin/env python3
"""
Experiment 1: Logistic Regression Baseline with BOW Features

This script trains interpretable emotion classifiers on each genre
and evaluates cross-genre transfer performance.

Research questions:
1. What words/phrases signal each emotion in each genre?
2. How well do emotion patterns transfer across genres?

Usage:
    python experiments/scripts/run_experiment_1.py

Outputs:
    - results/models/exp1_logreg_{genre}.joblib
    - results/metrics/exp1_results.json
    - results/metrics/exp1_transfer_matrix.csv
    - results/figures/exp1_*.png
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

# Dynamically add src directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_splits import load_splits
from baseline_model import (
    BOWEmotionClassifier,
    evaluate_classifier,
    format_results_table
)


# Configuration
CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42,
    'top_features_per_emotion': 30
}

EMOTION_LABELS = ['angry', 'happy', 'relaxed', 'sad']


def train_genre_models(
    splits: dict,
    config: dict
) -> dict:
    """
    Train a logistic regression classifier for each genre.
    
    Args:
        splits: Dictionary of genre splits from load_splits()
        config: Model configuration parameters
    
    Returns:
        Dictionary mapping genre names to fitted classifiers
    """
    models = {}
    
    for genre, genre_splits in splits.items():
        print(f"\nTraining model for {genre}...")
        
        train_df = genre_splits['train']
        val_df = genre_splits['val']
        
        # Initialize and train
        model = BOWEmotionClassifier(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_iter=config['max_iter'],
            class_weight=config['class_weight'],
            random_state=config['random_state']
        )
        
        model.fit(
            train_df['lyrics'].tolist(),
            train_df['emotion_label'].tolist()
        )
        
        # Validate
        val_pred = model.predict(val_df['lyrics'].tolist())
        val_results = evaluate_classifier(
            val_df['emotion_label'].tolist(),
            val_pred,
            labels=EMOTION_LABELS
        )
        
        print(f"  Vocabulary size: {model.get_vocabulary_size()}")
        print(f"  Validation macro F1: {val_results['macro_f1']:.4f}")
        
        models[genre] = model
    
    return models


def evaluate_cross_genre(
    models: dict,
    splits: dict
) -> pd.DataFrame:
    """
    Evaluate all train-test genre combinations.
    
    Creates a 4x4 transfer matrix showing F1 scores for each
    (train_genre, test_genre) pair.
    
    Args:
        models: Dictionary of trained models per genre
        splits: Dictionary of genre splits
    
    Returns:
        DataFrame with transfer matrix (rows=train, cols=test)
    """
    genres = list(models.keys())
    results_matrix = np.zeros((len(genres), len(genres)))
    detailed_results = {}
    
    for i, train_genre in enumerate(genres):
        model = models[train_genre]
        detailed_results[train_genre] = {}
        
        for j, test_genre in enumerate(genres):
            test_df = splits[test_genre]['test']
            
            # Predict
            y_true = test_df['emotion_label'].tolist()
            y_pred = model.predict(test_df['lyrics'].tolist())
            
            # Evaluate
            results = evaluate_classifier(y_true, y_pred, labels=EMOTION_LABELS)
            results_matrix[i, j] = results['macro_f1']
            detailed_results[train_genre][test_genre] = results
    
    # Create DataFrame
    transfer_df = pd.DataFrame(
        results_matrix,
        index=genres,
        columns=genres
    )
    transfer_df.index.name = 'train_genre'
    transfer_df.columns.name = 'test_genre'
    
    return transfer_df, detailed_results


def extract_all_features(
    models: dict,
    n_features: int = 30
) -> dict:
    """
    Extract top predictive features from all models.
    
    Args:
        models: Dictionary of trained models per genre
        n_features: Number of top features per emotion
    
    Returns:
        Nested dictionary: {genre: {emotion: [(word, coef), ...]}}
    """
    all_features = {}
    
    for genre, model in models.items():
        all_features[genre] = model.get_top_features(n_features)
    
    return all_features


def plot_transfer_matrix(
    transfer_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate heatmap visualization of transfer performance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        transfer_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.2,
        vmax=0.6,
        center=0.4,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Macro F1 Score'}
    )
    
    ax.set_title('Cross-Genre Emotion Classification Transfer Matrix\n(Logistic Regression + BOW)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Genre', fontsize=11)
    ax.set_ylabel('Train Genre', fontsize=11)
    
    # Highlight diagonal (within-genre performance)
    for i in range(len(transfer_df)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                    edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    detailed_results: dict,
    output_dir: Path
) -> None:
    """Generate confusion matrices for selected genre pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot within-genre confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    genres = list(detailed_results.keys())
    
    for idx, genre in enumerate(genres):
        ax = axes[idx // 2, idx % 2]
        
        cm = np.array(detailed_results[genre][genre]['confusion_matrix'])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=EMOTION_LABELS,
            yticklabels=EMOTION_LABELS,
            ax=ax
        )
        
        ax.set_title(f'{genre.upper()} (within-genre)', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.suptitle('Within-Genre Confusion Matrices (Logistic Regression)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'exp1_confusion_within_genre.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_features(
    all_features: dict,
    output_dir: Path,
    n_display: int = 15
) -> None:
    """Generate bar plots of top features per genre and emotion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion in EMOTION_LABELS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        genres = list(all_features.keys())
        
        for idx, genre in enumerate(genres):
            ax = axes[idx // 2, idx % 2]
            
            features = all_features[genre][emotion][:n_display]
            words = [f[0] for f in features]
            coeffs = [f[1] for f in features]
            
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coeffs]
            
            ax.barh(range(len(words)), coeffs, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Coefficient')
            ax.set_title(f'{genre.upper()}', fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Top Features for "{emotion.upper()}" Emotion', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'exp1_features_{emotion}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()


def analyze_asymmetric_transfers(
    transfer_df: pd.DataFrame
) -> dict:
    """
    Identify and quantify asymmetric transfer patterns.
    
    Returns pairs where A→B performance differs substantially from B→A.
    """
    genres = transfer_df.index.tolist()
    asymmetries = []
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i < j:  # Only check each pair once
                f1_g1_to_g2 = transfer_df.loc[g1, g2]
                f1_g2_to_g1 = transfer_df.loc[g2, g1]
                diff = f1_g1_to_g2 - f1_g2_to_g1
                
                asymmetries.append({
                    'pair': f'{g1} ↔ {g2}',
                    'forward': f'{g1} → {g2}',
                    'backward': f'{g2} → {g1}',
                    'f1_forward': f1_g1_to_g2,
                    'f1_backward': f1_g2_to_g1,
                    'difference': diff,
                    'abs_difference': abs(diff)
                })
    
    # Sort by absolute difference
    asymmetries.sort(key=lambda x: x['abs_difference'], reverse=True)
    
    return asymmetries


def save_results(
    transfer_df: pd.DataFrame,
    detailed_results: dict,
    all_features: dict,
    asymmetries: list,
    output_dir: Path,
    config: dict
) -> None:
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transfer matrix
    transfer_df.to_csv(output_dir / 'exp1_transfer_matrix.csv')
    
    # Save detailed results
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for train_g, test_results in detailed_results.items():
        serializable_results[train_g] = {}
        for test_g, metrics in test_results.items():
            serializable_results[train_g][test_g] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in metrics.items()
            }
    
    with open(output_dir / 'exp1_detailed_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save top features
    with open(output_dir / 'exp1_top_features.json', 'w') as f:
        json.dump(all_features, f, indent=2)
    
    # Save asymmetry analysis
    with open(output_dir / 'exp1_asymmetries.json', 'w') as f:
        json.dump(asymmetries, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        'experiment': 'Experiment 1: Logistic Regression Baseline',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'summary': {
            'mean_within_genre_f1': float(np.diag(transfer_df.values).mean()),
            'mean_cross_genre_f1': float(transfer_df.values[~np.eye(len(transfer_df), dtype=bool)].mean()),
            'best_transfer_pair': None,
            'worst_transfer_pair': None
        }
    }
    
    # Find best/worst cross-genre transfers
    mask = ~np.eye(len(transfer_df), dtype=bool)
    cross_genre_vals = transfer_df.values.copy()
    cross_genre_vals[~mask] = np.nan
    
    best_idx = np.unravel_index(np.nanargmax(cross_genre_vals), cross_genre_vals.shape)
    worst_idx = np.unravel_index(np.nanargmin(cross_genre_vals), cross_genre_vals.shape)
    
    genres = transfer_df.index.tolist()
    metadata['summary']['best_transfer_pair'] = {
        'train': genres[best_idx[0]],
        'test': genres[best_idx[1]],
        'f1': float(cross_genre_vals[best_idx])
    }
    metadata['summary']['worst_transfer_pair'] = {
        'train': genres[worst_idx[0]],
        'test': genres[worst_idx[1]],
        'f1': float(cross_genre_vals[worst_idx])
    }
    
    with open(output_dir / 'exp1_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Run Experiment 1."""
    print("=" * 70)
    print("EXPERIMENT 1: LOGISTIC REGRESSION BASELINE WITH BOW FEATURES")
    print("=" * 70)
    
    # Setup paths
    metrics_dir = project_root / "results" / "metrics"
    models_dir = project_root / "results" / "models"
    figures_dir = project_root / "results" / "figures"
    
    # Load data splits
    print("\n[1/6] Loading data splits...")
    splits = load_splits(input_dir=project_root / "data" / "splits")
    
    for genre, genre_splits in splits.items():
        n_train = len(genre_splits['train'])
        n_test = len(genre_splits['test'])
        print(f"  {genre}: train={n_train}, test={n_test}")
    
    # Train models
    print("\n[2/6] Training genre-specific models...")
    models = train_genre_models(splits, CONFIG)
    
    # Save models
    print("\n[3/6] Saving trained models...")
    models_dir.mkdir(parents=True, exist_ok=True)
    for genre, model in models.items():
        model_path = models_dir / f"exp1_logreg_{genre}.joblib"
        model.save(model_path)
        print(f"  Saved: {model_path.name}")
    
    # Cross-genre evaluation
    print("\n[4/6] Evaluating cross-genre transfer...")
    transfer_df, detailed_results = evaluate_cross_genre(models, splits)
    
    print("\nTransfer Matrix (Macro F1):")
    print(transfer_df.round(4).to_string())
    
    # Analyze asymmetries
    asymmetries = analyze_asymmetric_transfers(transfer_df)
    
    print("\nAsymmetric Transfer Patterns:")
    for asym in asymmetries[:3]:  # Top 3
        print(f"  {asym['pair']}: {asym['f1_forward']:.4f} vs {asym['f1_backward']:.4f} "
              f"(Δ = {asym['difference']:+.4f})")
    
    # Extract features
    print("\n[5/6] Extracting top predictive features...")
    all_features = extract_all_features(models, CONFIG['top_features_per_emotion'])
    
    # Print sample features
    print("\nSample top features for 'angry' emotion:")
    for genre in list(models.keys())[:2]:
        features = all_features[genre]['angry'][:5]
        feature_str = ", ".join([f[0] for f in features])
        print(f"  {genre}: {feature_str}")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_transfer_matrix(transfer_df, figures_dir / 'exp1_transfer_matrix.png')
    print("  Saved: exp1_transfer_matrix.png")
    
    plot_confusion_matrices(detailed_results, figures_dir)
    print("  Saved: exp1_confusion_within_genre.png")
    
    plot_top_features(all_features, figures_dir)
    print("  Saved: exp1_features_{emotion}.png (4 files)")
    
    # Save all results
    save_results(
        transfer_df, detailed_results, all_features, asymmetries,
        metrics_dir, CONFIG
    )
    print(f"\n  Results saved to: {metrics_dir}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)
    
    within_genre_mean = np.diag(transfer_df.values).mean()
    cross_genre_mean = transfer_df.values[~np.eye(len(transfer_df), dtype=bool)].mean()
    transfer_drop = within_genre_mean - cross_genre_mean
    
    print(f"""
Summary:
  Mean within-genre F1:  {within_genre_mean:.4f}
  Mean cross-genre F1:   {cross_genre_mean:.4f}
  Transfer performance drop: {transfer_drop:.4f} ({transfer_drop/within_genre_mean*100:.1f}%)

Key findings to investigate:
  1. Review transfer matrix for asymmetric patterns
  2. Compare top features across genres for same emotion
  3. Examine confusion matrices for systematic misclassifications

Next step: Run Experiment 2 (DistilBERT) for comparison
    """)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()


