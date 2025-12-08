#!/usr/bin/env python3
"""
Experiment 2 (Extended): DistilBERT Neural Transfer Learning

This script trains DistilBERT-based emotion classifiers with extended training,
early stopping, and optimized hyperparameters to give the neural model a 
stronger test against the BOW baseline.

Changes from initial run:
- Increased epochs: 3 → 10 (with early stopping)
- Lower learning rate: 2e-5 → 1e-5 (more stable fine-tuning)
- More warmup steps: 100 → 200 (gentler start)
- Early stopping patience: 3 epochs
- Gradient accumulation effective batch size consideration

Usage:
    python experiments/scripts/run_experiment_2_extended.py

Outputs:
    - results/models/exp2_distilbert_{genre}/
    - results/metrics/exp2_results.json
    - results/metrics/exp2_transfer_matrix.csv
    - results/figures/exp2_*.png
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
import torch

from data_splits import load_splits
from distilbert_model import (
    DistilBERTEmotionClassifier,
    evaluate_classifier
)


# EXTENDED TRAINING Configuration
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'num_labels': 4,
    'max_length': 512,
    
    # Extended training parameters
    'epochs': 10,                    # Increased from 3
    'batch_size': 16,
    'learning_rate': 1e-5,           # Reduced from 2e-5 for stability
    'warmup_steps': 200,             # Increased from 100
    'weight_decay': 0.01,
    'use_class_weights': True,
    'early_stopping_patience': 3,    # NEW: stop if no improvement for 3 epochs
    
    'random_seed': 42
}

EMOTION_LABELS = ['angry', 'happy', 'relaxed', 'sad']


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_genre_models(
    splits: dict,
    config: dict,
    models_dir: Path
) -> dict:
    """
    Train a DistilBERT classifier for each genre with extended training.
    """
    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for genre, genre_splits in splits.items():
        print(f"\n{'='*60}")
        print(f"Training model for {genre.upper()}")
        print(f"{'='*60}")
        
        train_df = genre_splits['train']
        val_df = genre_splits['val']
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")
        print(f"  Class distribution (train):")
        for emotion in EMOTION_LABELS:
            count = (train_df['emotion_label'] == emotion).sum()
            pct = count / len(train_df) * 100
            print(f"    {emotion}: {count} ({pct:.1f}%)")
        
        # Initialize classifier
        model = DistilBERTEmotionClassifier(
            num_labels=config['num_labels'],
            model_name=config['model_name'],
            max_length=config['max_length'],
            device=device
        )
        
        # Train with extended settings
        model.fit(
            train_texts=train_df['lyrics'].tolist(),
            train_labels=train_df['emotion_label'].tolist(),
            val_texts=val_df['lyrics'].tolist(),
            val_labels=val_df['emotion_label'].tolist(),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            warmup_steps=config['warmup_steps'],
            weight_decay=config['weight_decay'],
            use_class_weights=config['use_class_weights'],
            early_stopping_patience=config['early_stopping_patience'],
            verbose=True
        )
        
        # Report training history
        best_epoch = np.argmax(model.history['val_f1']) + 1
        best_f1 = max(model.history['val_f1'])
        print(f"\n  Best validation F1: {best_f1:.4f} (epoch {best_epoch})")
        print(f"  Training completed after {len(model.history['train_loss'])} epochs")
        
        # Save model
        model_path = models_dir / f"exp2_distilbert_{genre}"
        model.save(model_path)
        print(f"  Saved model to: {model_path}")
        
        models[genre] = model
        
        # Clear CUDA cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return models


def evaluate_cross_genre(
    models: dict,
    splits: dict,
    batch_size: int = 16
) -> tuple:
    """
    Evaluate all train-test genre combinations.
    """
    genres = list(models.keys())
    results_matrix = np.zeros((len(genres), len(genres)))
    detailed_results = {}
    
    print(f"\n{'='*60}")
    print("CROSS-GENRE EVALUATION")
    print(f"{'='*60}")
    
    for i, train_genre in enumerate(genres):
        model = models[train_genre]
        detailed_results[train_genre] = {}
        
        for j, test_genre in enumerate(genres):
            test_df = splits[test_genre]['test']
            
            # Predict
            y_true = test_df['emotion_label'].tolist()
            y_pred = model.predict(
                test_df['lyrics'].tolist(),
                batch_size=batch_size
            )
            
            # Evaluate
            results = evaluate_classifier(y_true, y_pred.tolist(), labels=EMOTION_LABELS)
            results_matrix[i, j] = results['macro_f1']
            detailed_results[train_genre][test_genre] = results
            
            marker = "**" if train_genre == test_genre else "  "
            print(f"{marker}{train_genre:12s} -> {test_genre:12s}: F1 = {results['macro_f1']:.4f}")
    
    # Create DataFrame
    transfer_df = pd.DataFrame(
        results_matrix,
        index=genres,
        columns=genres
    )
    transfer_df.index.name = 'train_genre'
    transfer_df.columns.name = 'test_genre'
    
    return transfer_df, detailed_results


def load_experiment1_results(metrics_dir: Path) -> pd.DataFrame:
    """Load Experiment 1 transfer matrix for comparison."""
    exp1_path = metrics_dir / 'exp1_transfer_matrix.csv'
    
    if exp1_path.exists():
        exp1_df = pd.read_csv(exp1_path, index_col=0)
        return exp1_df
    else:
        print(f"Warning: Experiment 1 results not found at {exp1_path}")
        return None


def analyze_asymmetric_transfers(transfer_df: pd.DataFrame) -> list:
    """
    Identify and quantify asymmetric transfer patterns.
    """
    genres = transfer_df.index.tolist()
    asymmetries = []
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i < j:
                f1_g1_to_g2 = transfer_df.loc[g1, g2]
                f1_g2_to_g1 = transfer_df.loc[g2, g1]
                diff = f1_g1_to_g2 - f1_g2_to_g1
                
                asymmetries.append({
                    'pair': f'{g1} ↔ {g2}',
                    'forward': f'{g1} → {g2}',
                    'backward': f'{g2} → {g1}',
                    'f1_forward': float(f1_g1_to_g2),
                    'f1_backward': float(f1_g2_to_g1),
                    'difference': float(diff),
                    'abs_difference': float(abs(diff))
                })
    
    asymmetries.sort(key=lambda x: x['abs_difference'], reverse=True)
    return asymmetries


def plot_transfer_matrix(
    transfer_df: pd.DataFrame,
    output_path: Path,
    title: str = "DistilBERT (Extended)"
) -> None:
    """Generate heatmap visualization of transfer performance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    ax.set_title(f'Cross-Genre Emotion Classification Transfer Matrix\n({title})', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Genre', fontsize=11)
    ax.set_ylabel('Train Genre', fontsize=11)
    
    # Highlight diagonal
    for i in range(len(transfer_df)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                    edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_heatmap(
    exp1_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate side-by-side comparison of Exp 1 and Exp 2."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Exp 1
    sns.heatmap(
        exp1_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.2,
        vmax=0.6,
        center=0.4,
        square=True,
        ax=axes[0],
        cbar=False
    )
    axes[0].set_title('Experiment 1: Logistic Regression + BOW', fontweight='bold')
    axes[0].set_xlabel('Test Genre')
    axes[0].set_ylabel('Train Genre')
    
    # Exp 2
    sns.heatmap(
        exp2_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.2,
        vmax=0.6,
        center=0.4,
        square=True,
        ax=axes[1],
        cbar=False
    )
    axes[1].set_title('Experiment 2: DistilBERT (Extended Training)', fontweight='bold')
    axes[1].set_xlabel('Test Genre')
    axes[1].set_ylabel('Train Genre')
    
    # Difference (Exp 2 - Exp 1)
    diff_df = exp2_df - exp1_df
    sns.heatmap(
        diff_df,
        annot=True,
        fmt='+.3f',
        cmap='RdBu',
        vmin=-0.15,
        vmax=0.15,
        center=0,
        square=True,
        ax=axes[2],
        cbar_kws={'label': 'F1 Difference'}
    )
    axes[2].set_title('Improvement (Exp 2 - Exp 1)', fontweight='bold')
    axes[2].set_xlabel('Test Genre')
    axes[2].set_ylabel('Train Genre')
    
    plt.suptitle('Cross-Genre Transfer: BOW vs. DistilBERT (Extended)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    models: dict,
    output_path: Path
) -> None:
    """Plot training and validation curves for all genres."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (genre, model) in enumerate(models.items()):
        ax = axes[idx // 2, idx % 2]
        
        epochs = range(1, len(model.history['train_loss']) + 1)
        
        # Plot losses
        ax2 = ax.twinx()
        
        line1, = ax.plot(epochs, model.history['val_f1'], 'b-o', 
                         label='Val F1', linewidth=2, markersize=6)
        line2, = ax2.plot(epochs, model.history['train_loss'], 'r--', 
                          label='Train Loss', linewidth=1.5, alpha=0.7)
        line3, = ax2.plot(epochs, model.history['val_loss'], 'r-', 
                          label='Val Loss', linewidth=1.5, alpha=0.7)
        
        # Mark best epoch
        best_epoch = np.argmax(model.history['val_f1']) + 1
        best_f1 = max(model.history['val_f1'])
        ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7)
        ax.scatter([best_epoch], [best_f1], color='green', s=100, zorder=5, 
                   marker='*', label=f'Best (F1={best_f1:.3f})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation F1', color='blue')
        ax2.set_ylabel('Loss', color='red')
        ax.set_title(f'{genre.upper()}', fontweight='bold')
        ax.set_ylim(0, 0.7)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves (Extended Training with Early Stopping)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    detailed_results: dict,
    output_dir: Path
) -> None:
    """Generate confusion matrices for within-genre evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    plt.suptitle('Within-Genre Confusion Matrices (DistilBERT Extended)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'exp2_confusion_within_genre.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_emotion_comparison(
    exp1_results: dict,
    exp2_results: dict,
    output_path: Path
) -> None:
    """Compare per-emotion F1 scores between experiments."""
    
    # Extract within-genre per-emotion F1 scores
    data = []
    for genre in exp1_results.keys():
        for emotion in EMOTION_LABELS:
            if genre in exp1_results and genre in exp1_results[genre]:
                exp1_f1 = exp1_results[genre][genre]['per_emotion_f1'].get(emotion, 0)
            else:
                exp1_f1 = 0
            
            exp2_f1 = exp2_results[genre][genre]['per_emotion_f1'].get(emotion, 0)
            
            data.append({
                'Genre': genre,
                'Emotion': emotion,
                'Experiment 1 (BOW)': exp1_f1,
                'Experiment 2 (DistilBERT)': exp2_f1
            })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, emotion in enumerate(EMOTION_LABELS):
        ax = axes[idx // 2, idx % 2]
        emotion_df = df[df['Emotion'] == emotion]
        
        x = np.arange(len(emotion_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, emotion_df['Experiment 1 (BOW)'], 
                       width, label='Exp 1: BOW', color='#3498db')
        bars2 = ax.bar(x + width/2, emotion_df['Experiment 2 (DistilBERT)'], 
                       width, label='Exp 2: DistilBERT', color='#e74c3c')
        
        ax.set_xlabel('Genre')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{emotion.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_df['Genre'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Per-Emotion F1 Comparison: BOW vs. DistilBERT Extended (Within-Genre)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(
    transfer_df: pd.DataFrame,
    detailed_results: dict,
    asymmetries: list,
    exp1_df: pd.DataFrame,
    models: dict,
    output_dir: Path,
    config: dict
) -> None:
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transfer matrix
    transfer_df.to_csv(output_dir / 'exp2_transfer_matrix.csv')
    
    # Save detailed results
    with open(output_dir / 'exp2_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save asymmetry analysis
    with open(output_dir / 'exp2_asymmetries.json', 'w') as f:
        json.dump(asymmetries, f, indent=2)
    
    # Save training histories
    training_histories = {}
    for genre, model in models.items():
        training_histories[genre] = {
            'train_loss': model.history['train_loss'],
            'val_loss': model.history['val_loss'],
            'val_f1': model.history['val_f1'],
            'best_epoch': int(np.argmax(model.history['val_f1']) + 1),
            'best_val_f1': float(max(model.history['val_f1'])),
            'total_epochs': len(model.history['train_loss'])
        }
    
    with open(output_dir / 'exp2_training_histories.json', 'w') as f:
        json.dump(training_histories, f, indent=2)
    
    # Compute comparison metrics
    within_genre_mean = float(np.diag(transfer_df.values).mean())
    cross_genre_mean = float(transfer_df.values[~np.eye(len(transfer_df), dtype=bool)].mean())
    
    comparison = {}
    if exp1_df is not None:
        exp1_within = float(np.diag(exp1_df.values).mean())
        exp1_cross = float(exp1_df.values[~np.eye(len(exp1_df), dtype=bool)].mean())
        
        comparison = {
            'exp1_within_genre_f1': exp1_within,
            'exp1_cross_genre_f1': exp1_cross,
            'exp2_within_genre_f1': within_genre_mean,
            'exp2_cross_genre_f1': cross_genre_mean,
            'within_genre_improvement': within_genre_mean - exp1_within,
            'cross_genre_improvement': cross_genre_mean - exp1_cross,
            'exp1_transfer_drop': (exp1_within - exp1_cross) / exp1_within,
            'exp2_transfer_drop': (within_genre_mean - cross_genre_mean) / within_genre_mean
        }
    
    # Save experiment metadata
    metadata = {
        'experiment': 'Experiment 2: DistilBERT Neural Transfer (EXTENDED)',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'training_info': {
            'epochs_configured': config['epochs'],
            'early_stopping_patience': config['early_stopping_patience'],
            'learning_rate': config['learning_rate'],
            'warmup_steps': config['warmup_steps']
        },
        'summary': {
            'mean_within_genre_f1': within_genre_mean,
            'mean_cross_genre_f1': cross_genre_mean,
            'transfer_drop_pct': (within_genre_mean - cross_genre_mean) / within_genre_mean * 100
        },
        'comparison_with_exp1': comparison
    }
    
    with open(output_dir / 'exp2_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Run Experiment 2 with extended training."""
    print("=" * 70)
    print("EXPERIMENT 2 (EXTENDED): DISTILBERT NEURAL TRANSFER LEARNING")
    print("=" * 70)
    
    print("\nExtended training configuration:")
    print(f"  Epochs: {CONFIG['epochs']} (with early stopping)")
    print(f"  Learning rate: {CONFIG['learning_rate']} (reduced for stability)")
    print(f"  Warmup steps: {CONFIG['warmup_steps']}")
    print(f"  Early stopping patience: {CONFIG['early_stopping_patience']} epochs")
    
    # Set seed
    set_seed(CONFIG['random_seed'])
    
    # Setup paths
    metrics_dir = project_root / "results" / "metrics"
    models_dir = project_root / "results" / "models"
    figures_dir = project_root / "results" / "figures"
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("\nNo GPU available. Training on CPU.")
        print("Note: Extended training will take longer (~10-15 min per genre)")
    
    # Load data splits
    print("\n[1/6] Loading data splits...")
    splits = load_splits(input_dir=project_root / "data" / "splits")
    
    for genre, genre_splits in splits.items():
        n_train = len(genre_splits['train'])
        n_test = len(genre_splits['test'])
        print(f"  {genre}: train={n_train}, test={n_test}")
    
    # Train models
    print("\n[2/6] Training genre-specific DistilBERT models (extended)...")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    models = train_genre_models(splits, CONFIG, models_dir)
    
    # Cross-genre evaluation
    print("\n[3/6] Evaluating cross-genre transfer...")
    transfer_df, detailed_results = evaluate_cross_genre(
        models, splits, batch_size=CONFIG['batch_size']
    )
    
    print("\nTransfer Matrix (Macro F1):")
    print(transfer_df.round(4).to_string())
    
    # Analyze asymmetries
    asymmetries = analyze_asymmetric_transfers(transfer_df)
    
    print("\nAsymmetric Transfer Patterns:")
    for asym in asymmetries[:3]:
        print(f"  {asym['pair']}: {asym['f1_forward']:.4f} vs {asym['f1_backward']:.4f} "
              f"(Δ = {asym['difference']:+.4f})")
    
    # Load Experiment 1 results for comparison
    print("\n[4/6] Comparing with Experiment 1 (BOW baseline)...")
    exp1_df = load_experiment1_results(metrics_dir)
    
    if exp1_df is not None:
        print("\nComparison Summary:")
        
        exp1_within = np.diag(exp1_df.values).mean()
        exp1_cross = exp1_df.values[~np.eye(len(exp1_df), dtype=bool)].mean()
        exp2_within = np.diag(transfer_df.values).mean()
        exp2_cross = transfer_df.values[~np.eye(len(transfer_df), dtype=bool)].mean()
        
        print(f"  {'Metric':<25} {'Exp 1 (BOW)':<15} {'Exp 2 (BERT)':<15} {'Δ':<10}")
        print(f"  {'-'*65}")
        print(f"  {'Within-genre F1':<25} {exp1_within:<15.4f} {exp2_within:<15.4f} {exp2_within-exp1_within:+.4f}")
        print(f"  {'Cross-genre F1':<25} {exp1_cross:<15.4f} {exp2_cross:<15.4f} {exp2_cross-exp1_cross:+.4f}")
        print(f"  {'Transfer drop':<25} {(exp1_within-exp1_cross)/exp1_within*100:<14.1f}% {(exp2_within-exp2_cross)/exp2_within*100:<14.1f}%")
    
    # Load Exp 1 detailed results for per-emotion comparison
    exp1_detailed_path = metrics_dir / 'exp1_detailed_results.json'
    exp1_detailed = None
    if exp1_detailed_path.exists():
        with open(exp1_detailed_path, 'r') as f:
            exp1_detailed = json.load(f)
    
    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_transfer_matrix(transfer_df, figures_dir / 'exp2_transfer_matrix.png', 
                         'DistilBERT (Extended)')
    print("  Saved: exp2_transfer_matrix.png")
    
    if exp1_df is not None:
        plot_comparison_heatmap(exp1_df, transfer_df, figures_dir / 'exp2_comparison_heatmap.png')
        print("  Saved: exp2_comparison_heatmap.png")
    
    plot_training_curves(models, figures_dir / 'exp2_training_curves.png')
    print("  Saved: exp2_training_curves.png")
    
    plot_confusion_matrices(detailed_results, figures_dir)
    print("  Saved: exp2_confusion_within_genre.png")
    
    if exp1_detailed is not None:
        plot_per_emotion_comparison(
            exp1_detailed, detailed_results,
            figures_dir / 'exp2_per_emotion_comparison.png'
        )
        print("  Saved: exp2_per_emotion_comparison.png")
    
    # Save all results
    print("\n[6/6] Saving results...")
    save_results(
        transfer_df, detailed_results, asymmetries,
        exp1_df, models, metrics_dir, CONFIG
    )
    print(f"  Results saved to: {metrics_dir}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 (EXTENDED) COMPLETE")
    print("=" * 70)
    
    within_genre_mean = np.diag(transfer_df.values).mean()
    cross_genre_mean = transfer_df.values[~np.eye(len(transfer_df), dtype=bool)].mean()
    transfer_drop = (within_genre_mean - cross_genre_mean) / within_genre_mean * 100
    
    print(f"""
Summary:
  Mean within-genre F1:  {within_genre_mean:.4f}
  Mean cross-genre F1:   {cross_genre_mean:.4f}
  Transfer performance drop: {transfer_drop:.1f}%
""")
    
    # Report training efficiency
    print("Training Summary:")
    for genre, model in models.items():
        best_epoch = np.argmax(model.history['val_f1']) + 1
        best_f1 = max(model.history['val_f1'])
        total_epochs = len(model.history['train_loss'])
        print(f"  {genre}: best F1={best_f1:.4f} at epoch {best_epoch}/{total_epochs}")
    
    if exp1_df is not None:
        exp1_drop = (exp1_within - exp1_cross) / exp1_within * 100
        improvement = exp2_cross - exp1_cross
        
        print(f"""
Comparison with Experiment 1:
  Cross-genre F1 change: {improvement:+.4f}
  Transfer drop: {exp1_drop:.1f}% (BOW) vs {transfer_drop:.1f}% (BERT)
""")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()


