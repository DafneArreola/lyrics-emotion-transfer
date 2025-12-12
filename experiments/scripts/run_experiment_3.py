#!/usr/bin/env python3
"""
Experiment 3: Asymmetric Transfer Pattern Analysis

This experiment investigates WHY certain cross-genre transfer directions
outperform others by analyzing:

1. Vocabulary overlap between genres
2. Emotion-specific vocabulary distribution
3. Genre similarity based on multiple metrics
4. Prediction error patterns in failed transfers
5. Relationship between linguistic features and transfer success

Research questions:
- What explains the asymmetry in transfer performance?
- Does vocabulary size/overlap predict transfer success?
- Do genres with similar emotion distributions transfer better?
- Which emotion categories drive transfer failures?

Usage:
    python experiments/scripts/run_experiment_3.py

Outputs:
    - results/metrics/exp3_asymmetry_analysis.json
    - results/metrics/exp3_vocabulary_analysis.json
    - results/figures/exp3_*.png
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import combinations

# Dynamically add src directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

from data_splits import load_splits

# Configuration
EMOTION_LABELS = ['angry', 'happy', 'relaxed', 'sad']
GENRES = ['classic-pop', 'hip-hop', 'hard-rock', 'country']


def load_experiment_results(metrics_dir: Path) -> dict:
    """Load results from Experiments 1 and 2."""
    results = {}
    
    # Experiment 1 (BOW)
    exp1_matrix_path = metrics_dir / 'exp1_transfer_matrix.csv'
    exp1_details_path = metrics_dir / 'exp1_detailed_results.json'
    exp1_features_path = metrics_dir / 'exp1_top_features.json'
    
    if exp1_matrix_path.exists():
        results['exp1_matrix'] = pd.read_csv(exp1_matrix_path, index_col=0)
    if exp1_details_path.exists():
        with open(exp1_details_path, 'r') as f:
            results['exp1_details'] = json.load(f)
    if exp1_features_path.exists():
        with open(exp1_features_path, 'r') as f:
            results['exp1_features'] = json.load(f)
    
    # Experiment 2 (DistilBERT)
    exp2_matrix_path = metrics_dir / 'exp2_transfer_matrix.csv'
    exp2_details_path = metrics_dir / 'exp2_detailed_results.json'
    
    if exp2_matrix_path.exists():
        results['exp2_matrix'] = pd.read_csv(exp2_matrix_path, index_col=0)
    if exp2_details_path.exists():
        with open(exp2_details_path, 'r') as f:
            results['exp2_details'] = json.load(f)
    
    return results


def compute_asymmetry_metrics(transfer_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed asymmetry metrics for all genre pairs.
    
    Returns DataFrame with columns:
    - genre_a, genre_b: The two genres
    - a_to_b, b_to_a: F1 scores in each direction
    - asymmetry: Signed difference (a_to_b - b_to_a)
    - abs_asymmetry: Absolute difference
    - ratio: Ratio of larger to smaller
    - dominant_direction: Which direction is stronger
    """
    genres = transfer_matrix.index.tolist()
    records = []
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i < j:
                a_to_b = transfer_matrix.loc[g1, g2]
                b_to_a = transfer_matrix.loc[g2, g1]
                diff = a_to_b - b_to_a
                
                records.append({
                    'genre_a': g1,
                    'genre_b': g2,
                    'a_to_b': float(a_to_b),
                    'b_to_a': float(b_to_a),
                    'asymmetry': float(diff),
                    'abs_asymmetry': float(abs(diff)),
                    'ratio': float(max(a_to_b, b_to_a) / min(a_to_b, b_to_a)) if min(a_to_b, b_to_a) > 0 else float('inf'),
                    'dominant_direction': f'{g1}→{g2}' if diff > 0 else f'{g2}→{g1}'
                })
    
    df = pd.DataFrame(records)
    df = df.sort_values('abs_asymmetry', ascending=False)
    return df


def compute_vocabulary_metrics(splits: dict, max_features: int = 10000) -> dict:
    """
    Compute vocabulary statistics for each genre.
    
    Returns dict with:
    - vocab_size: Number of unique tokens per genre
    - total_tokens: Total token count per genre
    - type_token_ratio: Vocabulary diversity measure
    - vocab_sets: Actual vocabulary sets for overlap computation
    """
    metrics = {}
    
    for genre, genre_splits in splits.items():
        # Combine train + val + test for full vocabulary
        all_lyrics = pd.concat([
            genre_splits['train'],
            genre_splits['val'],
            genre_splits['test']
        ])['lyrics'].tolist()
        
        # Tokenize using CountVectorizer for vocabulary
        vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        vectorizer.fit(all_lyrics)
        vocab = set(vectorizer.get_feature_names_out())
        
        # Count tokens using simple split (preprocess_lyrics returns list)
        all_tokens = []
        for lyric in all_lyrics:
            # Simple tokenization - lowercase and split
            tokens = lyric.lower().split()
            all_tokens.extend(tokens)
        
        metrics[genre] = {
            'vocab_size': len(vocab),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'type_token_ratio': len(set(all_tokens)) / len(all_tokens) if all_tokens else 0,
            'vocab_set': vocab,
            'avg_song_length': np.mean([len(l.split()) for l in all_lyrics])
        }
    
    return metrics


def compute_vocabulary_overlap(vocab_metrics: dict) -> pd.DataFrame:
    """
    Compute pairwise vocabulary overlap between genres.
    
    Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
    """
    genres = list(vocab_metrics.keys())
    overlap_matrix = np.zeros((len(genres), len(genres)))
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            v1 = vocab_metrics[g1]['vocab_set']
            v2 = vocab_metrics[g2]['vocab_set']
            
            intersection = len(v1 & v2)
            union = len(v1 | v2)
            
            overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    return pd.DataFrame(overlap_matrix, index=genres, columns=genres)


def compute_emotion_distribution_similarity(splits: dict) -> pd.DataFrame:
    """
    Compute pairwise similarity of emotion distributions between genres.
    
    Uses Jensen-Shannon divergence (lower = more similar).
    """
    from scipy.spatial.distance import jensenshannon
    
    # Get emotion distributions
    distributions = {}
    for genre, genre_splits in splits.items():
        all_data = pd.concat([genre_splits['train'], genre_splits['val'], genre_splits['test']])
        counts = all_data['emotion_label'].value_counts()
        # Ensure consistent ordering
        dist = np.array([counts.get(e, 0) for e in EMOTION_LABELS])
        dist = dist / dist.sum()  # Normalize
        distributions[genre] = dist
    
    genres = list(distributions.keys())
    similarity_matrix = np.zeros((len(genres), len(genres)))
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            # Convert JS divergence to similarity (1 - divergence)
            js_div = jensenshannon(distributions[g1], distributions[g2])
            similarity_matrix[i, j] = 1 - js_div
    
    return pd.DataFrame(similarity_matrix, index=genres, columns=genres)


def analyze_emotion_specific_transfer(exp_details: dict) -> dict:
    """
    Analyze which emotions drive transfer success/failure for each genre pair.
    
    Returns per-emotion transfer matrices.
    """
    genres = list(exp_details.keys())
    emotion_matrices = {}
    
    for emotion in EMOTION_LABELS:
        matrix = np.zeros((len(genres), len(genres)))
        
        for i, train_genre in enumerate(genres):
            for j, test_genre in enumerate(genres):
                if train_genre in exp_details and test_genre in exp_details[train_genre]:
                    per_emotion_f1 = exp_details[train_genre][test_genre].get('per_emotion_f1', {})
                    matrix[i, j] = per_emotion_f1.get(emotion, 0)
        
        emotion_matrices[emotion] = pd.DataFrame(matrix, index=genres, columns=genres)
    
    return emotion_matrices


def compute_transfer_predictors(
    transfer_matrix: pd.DataFrame,
    vocab_overlap: pd.DataFrame,
    emotion_similarity: pd.DataFrame,
    vocab_metrics: dict
) -> pd.DataFrame:
    """
    Create a dataset relating transfer performance to potential predictors.
    
    For each (train_genre, test_genre) pair, compute:
    - Transfer F1
    - Vocabulary overlap
    - Emotion distribution similarity
    - Train genre vocab size
    - Test genre vocab size
    - Vocab size ratio
    """
    records = []
    genres = transfer_matrix.index.tolist()
    
    for train_genre in genres:
        for test_genre in genres:
            if train_genre != test_genre:
                records.append({
                    'train_genre': train_genre,
                    'test_genre': test_genre,
                    'transfer_f1': transfer_matrix.loc[train_genre, test_genre],
                    'vocab_overlap': vocab_overlap.loc[train_genre, test_genre],
                    'emotion_similarity': emotion_similarity.loc[train_genre, test_genre],
                    'train_vocab_size': vocab_metrics[train_genre]['vocab_size'],
                    'test_vocab_size': vocab_metrics[test_genre]['vocab_size'],
                    'vocab_ratio': vocab_metrics[train_genre]['vocab_size'] / vocab_metrics[test_genre]['vocab_size'],
                    'train_avg_length': vocab_metrics[train_genre]['avg_song_length'],
                    'test_avg_length': vocab_metrics[test_genre]['avg_song_length']
                })
    
    return pd.DataFrame(records)


def compute_feature_overlap_by_emotion(exp1_features: dict) -> dict:
    """
    Analyze overlap of top emotional features across genres.
    
    For each emotion, compute how many top features are shared between genres.
    """
    overlap_analysis = {}
    
    for emotion in EMOTION_LABELS:
        genre_features = {}
        
        for genre in GENRES:
            if genre in exp1_features and emotion in exp1_features[genre]:
                # Extract just the feature names (not coefficients)
                features = [f[0] for f in exp1_features[genre][emotion][:30]]
                genre_features[genre] = set(features)
        
        # Compute pairwise overlap
        overlap_matrix = {}
        for g1, g2 in combinations(genre_features.keys(), 2):
            shared = genre_features[g1] & genre_features[g2]
            overlap_matrix[f'{g1}↔{g2}'] = {
                'shared_count': len(shared),
                'shared_features': list(shared)[:10],  # Top 10 for display
                'g1_unique': list(genre_features[g1] - genre_features[g2])[:5],
                'g2_unique': list(genre_features[g2] - genre_features[g1])[:5]
            }
        
        overlap_analysis[emotion] = overlap_matrix
    
    return overlap_analysis


def plot_asymmetry_comparison(
    exp1_asymmetry: pd.DataFrame,
    exp2_asymmetry: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot asymmetry comparison between BOW and DistilBERT."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data
    pairs = [f"{row['genre_a']}↔{row['genre_b']}" for _, row in exp1_asymmetry.iterrows()]
    
    # BOW asymmetries
    ax1 = axes[0]
    colors1 = ['#e74c3c' if x < 0 else '#27ae60' for x in exp1_asymmetry['asymmetry']]
    bars1 = ax1.barh(pairs, exp1_asymmetry['asymmetry'], color=colors1)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Asymmetry (A→B minus B→A)')
    ax1.set_title('Experiment 1: BOW', fontweight='bold')
    ax1.set_xlim(-0.25, 0.25)
    
    # Add value labels
    for bar, val in zip(bars1, exp1_asymmetry['asymmetry']):
        ax1.text(val + 0.01 if val >= 0 else val - 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    # DistilBERT asymmetries
    ax2 = axes[1]
    # Reorder exp2 to match exp1 pairs
    exp2_ordered = exp2_asymmetry.set_index(
        exp2_asymmetry['genre_a'] + '↔' + exp2_asymmetry['genre_b']
    ).loc[pairs]
    
    colors2 = ['#e74c3c' if x < 0 else '#27ae60' for x in exp2_ordered['asymmetry']]
    bars2 = ax2.barh(pairs, exp2_ordered['asymmetry'], color=colors2)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Asymmetry (A→B minus B→A)')
    ax2.set_title('Experiment 2: DistilBERT', fontweight='bold')
    ax2.set_xlim(-0.25, 0.25)
    
    for bar, val in zip(bars2, exp2_ordered['asymmetry']):
        ax2.text(val + 0.01 if val >= 0 else val - 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    plt.suptitle('Transfer Asymmetry Comparison: BOW vs. DistilBERT', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_vocabulary_overlap_heatmap(
    vocab_overlap: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot vocabulary overlap heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        vocab_overlap,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0.3,
        vmax=0.7,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Jaccard Similarity'}
    )
    
    ax.set_title('Vocabulary Overlap Between Genres\n(Jaccard Similarity)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Genre')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_emotion_distribution_comparison(
    splits: dict,
    output_path: Path
) -> None:
    """Plot emotion distributions across genres."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'angry': '#e74c3c', 'happy': '#f1c40f', 'relaxed': '#3498db', 'sad': '#9b59b6'}
    
    for idx, genre in enumerate(GENRES):
        ax = axes[idx // 2, idx % 2]
        
        all_data = pd.concat([
            splits[genre]['train'],
            splits[genre]['val'],
            splits[genre]['test']
        ])
        
        counts = all_data['emotion_label'].value_counts()
        emotions = EMOTION_LABELS
        values = [counts.get(e, 0) for e in emotions]
        percentages = [v / sum(values) * 100 for v in values]
        
        bars = ax.bar(emotions, percentages, color=[colors[e] for e in emotions])
        
        ax.set_ylabel('Percentage')
        ax.set_title(f'{genre.upper()}', fontweight='bold')
        ax.set_ylim(0, 60)
        
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Emotion Distribution by Genre', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_transfer_predictors(
    predictors_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot relationships between transfer performance and predictors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Vocab overlap vs Transfer F1
    ax1 = axes[0, 0]
    ax1.scatter(predictors_df['vocab_overlap'], predictors_df['transfer_f1'], 
               c='#3498db', alpha=0.7, s=100)
    
    # Add correlation
    r, p = stats.pearsonr(predictors_df['vocab_overlap'], predictors_df['transfer_f1'])
    ax1.set_xlabel('Vocabulary Overlap (Jaccard)')
    ax1.set_ylabel('Transfer F1')
    ax1.set_title(f'Vocabulary Overlap vs. Transfer\nr={r:.3f}, p={p:.3f}', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(predictors_df['vocab_overlap'], predictors_df['transfer_f1'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(predictors_df['vocab_overlap'].min(), predictors_df['vocab_overlap'].max(), 100)
    ax1.plot(x_line, p_line(x_line), 'r--', alpha=0.7)
    
    # 2. Emotion similarity vs Transfer F1
    ax2 = axes[0, 1]
    ax2.scatter(predictors_df['emotion_similarity'], predictors_df['transfer_f1'],
               c='#e74c3c', alpha=0.7, s=100)
    
    r, p = stats.pearsonr(predictors_df['emotion_similarity'], predictors_df['transfer_f1'])
    ax2.set_xlabel('Emotion Distribution Similarity')
    ax2.set_ylabel('Transfer F1')
    ax2.set_title(f'Emotion Similarity vs. Transfer\nr={r:.3f}, p={p:.3f}', fontweight='bold')
    
    z = np.polyfit(predictors_df['emotion_similarity'], predictors_df['transfer_f1'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(predictors_df['emotion_similarity'].min(), predictors_df['emotion_similarity'].max(), 100)
    ax2.plot(x_line, p_line(x_line), 'r--', alpha=0.7)
    
    # 3. Train vocab size vs Transfer F1
    ax3 = axes[1, 0]
    ax3.scatter(predictors_df['train_vocab_size'], predictors_df['transfer_f1'],
               c='#27ae60', alpha=0.7, s=100)
    
    r, p = stats.pearsonr(predictors_df['train_vocab_size'], predictors_df['transfer_f1'])
    ax3.set_xlabel('Training Genre Vocabulary Size')
    ax3.set_ylabel('Transfer F1')
    ax3.set_title(f'Train Vocab Size vs. Transfer\nr={r:.3f}, p={p:.3f}', fontweight='bold')
    
    # 4. Vocab ratio vs Transfer F1
    ax4 = axes[1, 1]
    ax4.scatter(predictors_df['vocab_ratio'], predictors_df['transfer_f1'],
               c='#9b59b6', alpha=0.7, s=100)
    
    r, p = stats.pearsonr(predictors_df['vocab_ratio'], predictors_df['transfer_f1'])
    ax4.set_xlabel('Vocabulary Ratio (Train/Test)')
    ax4.set_ylabel('Transfer F1')
    ax4.set_title(f'Vocab Ratio vs. Transfer\nr={r:.3f}, p={p:.3f}', fontweight='bold')
    
    plt.suptitle('Transfer Performance Predictors (BOW Model)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_emotion_transfer_heatmaps(
    emotion_matrices: dict,
    output_path: Path
) -> None:
    """Plot per-emotion transfer heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, emotion in enumerate(EMOTION_LABELS):
        ax = axes[idx // 2, idx % 2]
        
        sns.heatmap(
            emotion_matrices[emotion],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=0.8,
            center=0.4,
            square=True,
            ax=ax,
            cbar_kws={'label': 'F1 Score'}
        )
        
        ax.set_title(f'{emotion.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Genre')
        ax.set_ylabel('Train Genre')
    
    plt.suptitle('Per-Emotion Cross-Genre Transfer (BOW)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_genre_similarity_summary(
    vocab_overlap: pd.DataFrame,
    emotion_similarity: pd.DataFrame,
    transfer_matrix: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot combined genre similarity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Vocab overlap
    sns.heatmap(vocab_overlap, annot=True, fmt='.2f', cmap='Blues',
               square=True, ax=axes[0], cbar=False)
    axes[0].set_title('Vocabulary Overlap', fontweight='bold')
    
    # Emotion similarity
    sns.heatmap(emotion_similarity, annot=True, fmt='.2f', cmap='Greens',
               square=True, ax=axes[1], cbar=False)
    axes[1].set_title('Emotion Distribution Similarity', fontweight='bold')
    
    # Transfer performance (average of both directions)
    transfer_avg = (transfer_matrix + transfer_matrix.T) / 2
    sns.heatmap(transfer_avg, annot=True, fmt='.2f', cmap='RdYlGn',
               square=True, ax=axes[2], vmin=0.2, vmax=0.5, center=0.35)
    axes[2].set_title('Avg. Transfer F1 (BOW)', fontweight='bold')
    
    plt.suptitle('Genre Similarity Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(
    exp1_asymmetry: pd.DataFrame,
    exp2_asymmetry: pd.DataFrame,
    vocab_metrics: dict,
    vocab_overlap: pd.DataFrame,
    emotion_similarity: pd.DataFrame,
    predictors_df: pd.DataFrame,
    feature_overlap: dict,
    output_dir: Path
) -> None:
    """Save all analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Asymmetry analysis
    asymmetry_results = {
        'exp1_asymmetries': exp1_asymmetry.to_dict('records'),
        'exp2_asymmetries': exp2_asymmetry.to_dict('records'),
        'comparison': {
            'exp1_max_asymmetry': float(exp1_asymmetry['abs_asymmetry'].max()),
            'exp2_max_asymmetry': float(exp2_asymmetry['abs_asymmetry'].max()),
            'exp1_mean_asymmetry': float(exp1_asymmetry['abs_asymmetry'].mean()),
            'exp2_mean_asymmetry': float(exp2_asymmetry['abs_asymmetry'].mean())
        }
    }
    
    with open(output_dir / 'exp3_asymmetry_analysis.json', 'w') as f:
        json.dump(asymmetry_results, f, indent=2)
    
    # Vocabulary analysis
    vocab_results = {
        'genre_vocab_stats': {
            genre: {k: v for k, v in genre_stats.items() if k != 'vocab_set'}
            for genre, genre_stats in vocab_metrics.items()
        },
        'vocab_overlap_matrix': vocab_overlap.to_dict(),
        'emotion_similarity_matrix': emotion_similarity.to_dict()
    }
    
    with open(output_dir / 'exp3_vocabulary_analysis.json', 'w') as f:
        json.dump(vocab_results, f, indent=2)
    
    # Predictor correlations
    correlations = {}
    for col in ['vocab_overlap', 'emotion_similarity', 'train_vocab_size', 'vocab_ratio']:
        r, p = stats.pearsonr(predictors_df[col], predictors_df['transfer_f1'])
        correlations[col] = {'pearson_r': float(r), 'p_value': float(p)}
    
    predictor_results = {
        'correlations': correlations,
        'data': predictors_df.to_dict('records')
    }
    
    with open(output_dir / 'exp3_predictor_analysis.json', 'w') as f:
        json.dump(predictor_results, f, indent=2)
    
    # Feature overlap
    with open(output_dir / 'exp3_feature_overlap.json', 'w') as f:
        json.dump(feature_overlap, f, indent=2)
    
    # Metadata
    metadata = {
        'experiment': 'Experiment 3: Asymmetric Transfer Analysis',
        'timestamp': datetime.now().isoformat(),
        'genres': GENRES,
        'emotions': EMOTION_LABELS
    }
    
    with open(output_dir / 'exp3_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Run Experiment 3: Asymmetric Transfer Analysis."""
    print("=" * 70)
    print("EXPERIMENT 3: ASYMMETRIC TRANSFER PATTERN ANALYSIS")
    print("=" * 70)
    
    # Setup paths
    metrics_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load previous experiment results
    print("\n[1/8] Loading experiment results...")
    exp_results = load_experiment_results(metrics_dir)
    
    if 'exp1_matrix' not in exp_results:
        print("ERROR: Experiment 1 results not found. Run Experiment 1 first.")
        return
    
    print(f"  Loaded: Exp 1 transfer matrix, detailed results, features")
    if 'exp2_matrix' in exp_results:
        print(f"  Loaded: Exp 2 transfer matrix, detailed results")
    
    # Load data splits
    print("\n[2/8] Loading data splits...")
    splits = load_splits(input_dir=project_root / "data" / "splits")
    
    # Compute asymmetry metrics
    print("\n[3/8] Computing asymmetry metrics...")
    exp1_asymmetry = compute_asymmetry_metrics(exp_results['exp1_matrix'])
    
    print("\n  BOW Asymmetries (sorted by magnitude):")
    for _, row in exp1_asymmetry.iterrows():
        print(f"    {row['genre_a']}↔{row['genre_b']}: {row['asymmetry']:+.3f} "
              f"(dominant: {row['dominant_direction']})")
    
    if 'exp2_matrix' in exp_results:
        exp2_asymmetry = compute_asymmetry_metrics(exp_results['exp2_matrix'])
        print("\n  DistilBERT Asymmetries:")
        for _, row in exp2_asymmetry.iterrows():
            print(f"    {row['genre_a']}↔{row['genre_b']}: {row['asymmetry']:+.3f} "
                  f"(dominant: {row['dominant_direction']})")
    else:
        exp2_asymmetry = exp1_asymmetry.copy()  # Placeholder
    
    # Compute vocabulary metrics
    print("\n[4/8] Analyzing vocabulary characteristics...")
    vocab_metrics = compute_vocabulary_metrics(splits)
    
    print("\n  Genre Vocabulary Statistics:")
    for genre, genre_stats in vocab_metrics.items():
        print(f"    {genre}: {genre_stats['vocab_size']} unique terms, "
              f"avg {genre_stats['avg_song_length']:.0f} tokens/song")
    
    # Compute vocabulary overlap
    print("\n[5/8] Computing vocabulary overlap...")
    vocab_overlap = compute_vocabulary_overlap(vocab_metrics)
    
    print("\n  Vocabulary Overlap (Jaccard):")
    print(vocab_overlap.round(3).to_string())
    
    # Compute emotion distribution similarity
    print("\n[6/8] Computing emotion distribution similarity...")
    emotion_similarity = compute_emotion_distribution_similarity(splits)
    
    print("\n  Emotion Distribution Similarity:")
    print(emotion_similarity.round(3).to_string())
    
    # Compute transfer predictors
    print("\n[7/8] Analyzing transfer predictors...")
    predictors_df = compute_transfer_predictors(
        exp_results['exp1_matrix'],
        vocab_overlap,
        emotion_similarity,
        vocab_metrics
    )
    
    print("\n  Correlations with Transfer F1:")
    for col in ['vocab_overlap', 'emotion_similarity', 'train_vocab_size', 'vocab_ratio']:
        r, p = stats.pearsonr(predictors_df[col], predictors_df['transfer_f1'])
        sig = "**" if p < 0.05 else ""
        print(f"    {col}: r={r:+.3f}, p={p:.3f} {sig}")
    
    # Analyze per-emotion transfer
    emotion_matrices = analyze_emotion_specific_transfer(exp_results['exp1_details'])
    
    # Analyze feature overlap
    feature_overlap = {}
    if 'exp1_features' in exp_results:
        feature_overlap = compute_feature_overlap_by_emotion(exp_results['exp1_features'])
    
    # Generate visualizations
    print("\n[8/8] Generating visualizations...")
    
    plot_asymmetry_comparison(exp1_asymmetry, exp2_asymmetry,
                              figures_dir / 'exp3_asymmetry_comparison.png')
    print("  Saved: exp3_asymmetry_comparison.png")
    
    plot_vocabulary_overlap_heatmap(vocab_overlap,
                                    figures_dir / 'exp3_vocabulary_overlap.png')
    print("  Saved: exp3_vocabulary_overlap.png")
    
    plot_emotion_distribution_comparison(splits,
                                         figures_dir / 'exp3_emotion_distributions.png')
    print("  Saved: exp3_emotion_distributions.png")
    
    plot_transfer_predictors(predictors_df,
                            figures_dir / 'exp3_transfer_predictors.png')
    print("  Saved: exp3_transfer_predictors.png")
    
    plot_emotion_transfer_heatmaps(emotion_matrices,
                                   figures_dir / 'exp3_emotion_transfer.png')
    print("  Saved: exp3_emotion_transfer.png")
    
    plot_genre_similarity_summary(vocab_overlap, emotion_similarity,
                                  exp_results['exp1_matrix'],
                                  figures_dir / 'exp3_genre_similarity.png')
    print("  Saved: exp3_genre_similarity.png")
    
    # Save results
    print("\n  Saving results...")
    save_results(
        exp1_asymmetry, exp2_asymmetry,
        vocab_metrics, vocab_overlap, emotion_similarity,
        predictors_df, feature_overlap,
        metrics_dir
    )
    print(f"  Results saved to: {metrics_dir}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)
    
    print(f"""
Key Findings:

1. ASYMMETRY PATTERNS
   - BOW max asymmetry: {exp1_asymmetry['abs_asymmetry'].max():.3f}
   - BOW mean asymmetry: {exp1_asymmetry['abs_asymmetry'].mean():.3f}
   - Largest: {exp1_asymmetry.iloc[0]['genre_a']}↔{exp1_asymmetry.iloc[0]['genre_b']} 
     ({exp1_asymmetry.iloc[0]['asymmetry']:+.3f})

2. VOCABULARY ANALYSIS
   - Largest vocab: {max(vocab_metrics.items(), key=lambda x: x[1]['vocab_size'])[0]}
   - Smallest vocab: {min(vocab_metrics.items(), key=lambda x: x[1]['vocab_size'])[0]}
   - Highest overlap: {vocab_overlap.values[np.triu_indices_from(vocab_overlap.values, k=1)].max():.3f}

3. TRANSFER PREDICTORS
""")
    
    for col in ['vocab_overlap', 'emotion_similarity']:
        r, p = stats.pearsonr(predictors_df[col], predictors_df['transfer_f1'])
        print(f"   - {col}: r={r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
