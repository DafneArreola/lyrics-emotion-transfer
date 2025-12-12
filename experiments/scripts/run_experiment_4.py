#!/usr/bin/env python3
"""
Experiment 4: Error Analysis - Understanding Model Mistakes

This experiment examines specific misclassifications to understand WHY
models fail in cross-genre transfer:

1. Confusion pattern analysis - which emotions get confused?
2. Specific misclassified examples - what do failing lyrics look like?
3. Linguistic feature analysis of errors - what words cause mistakes?
4. Within-genre vs cross-genre error patterns
5. BOW vs DistilBERT error comparison

Research questions:
- Are certain emotion pairs systematically confused?
- Do misclassified lyrics share linguistic characteristics?
- Do BOW and DistilBERT make the same mistakes?
- What specific words/phrases cause misclassification?

Usage:
    python experiments/scripts/run_experiment_4.py

Outputs:
    - results/metrics/exp4_error_analysis.json
    - results/metrics/exp4_misclassified_examples.json
    - results/figures/exp4_*.png
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Dynamically add src directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from data_splits import load_splits

# Configuration
EMOTION_LABELS = ['angry', 'happy', 'relaxed', 'sad']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
GENRES = ['classic-pop', 'hip-hop', 'hard-rock', 'country']


def train_bow_model(train_df: pd.DataFrame, max_features: int = 5000):
    """Train a BOW + Logistic Regression model."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['lyrics'])
    y_train = train_df['emotion_label']
    
    # Compute class weights
    class_counts = y_train.value_counts()
    total = len(y_train)
    class_weights = {label: total / (len(class_counts) * count) 
                     for label, count in class_counts.items()}
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, vectorizer


def get_predictions_with_confidence(model, vectorizer, test_df: pd.DataFrame):
    """Get predictions along with confidence scores."""
    X_test = vectorizer.transform(test_df['lyrics'])
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        pred = predictions[i]
        true = row['emotion_label']
        probs = probabilities[i]
        confidence = probs.max()
        
        # Get probability for each class
        class_probs = {EMOTION_LABELS[j]: float(probs[j]) for j in range(len(EMOTION_LABELS))}
        
        results.append({
            'index': idx,
            'lyrics': row['lyrics'][:500],  # Truncate for storage
            'true_label': true,
            'predicted_label': pred,
            'correct': pred == true,
            'confidence': float(confidence),
            'class_probabilities': class_probs,
            'title': row.get('title', 'Unknown'),
            'artist': row.get('artist', 'Unknown')
        })
    
    return results


def analyze_confusion_patterns(predictions: list) -> dict:
    """Analyze systematic confusion patterns."""
    confusion_counts = defaultdict(int)
    
    for pred in predictions:
        if not pred['correct']:
            key = (pred['true_label'], pred['predicted_label'])
            confusion_counts[key] += 1
    
    # Sort by frequency
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: -x[1])
    
    return {
        'confusion_pairs': [
            {'true': k[0], 'predicted': k[1], 'count': v}
            for k, v in sorted_confusions
        ],
        'total_errors': sum(confusion_counts.values())
    }


def analyze_error_characteristics(predictions: list, vectorizer) -> dict:
    """Analyze linguistic characteristics of misclassified examples."""
    correct = [p for p in predictions if p['correct']]
    incorrect = [p for p in predictions if not p['correct']]
    
    if not incorrect:
        return {'no_errors': True}
    
    # Confidence analysis
    correct_confidence = [p['confidence'] for p in correct] if correct else [0]
    incorrect_confidence = [p['confidence'] for p in incorrect]
    
    # Length analysis
    correct_lengths = [len(p['lyrics'].split()) for p in correct] if correct else [0]
    incorrect_lengths = [len(p['lyrics'].split()) for p in incorrect]
    
    # Analyze words in misclassified examples
    error_words = defaultdict(list)
    for p in incorrect:
        key = (p['true_label'], p['predicted_label'])
        words = p['lyrics'].lower().split()
        error_words[f"{key[0]}_as_{key[1]}"].extend(words)
    
    # Get most common words for each error type
    error_word_freq = {}
    for error_type, words in error_words.items():
        word_counts = Counter(words)
        # Filter out very common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 
                       'me', 'my', 'your', 'it', 'to', 'and', 'of', 'in', 'on',
                       'that', 'this', 'for', 'with', 'be', 'have', 'do', 'at',
                       'but', 'not', 'what', 'all', 'we', 'can', 'her', 'his'}
        filtered_counts = {w: c for w, c in word_counts.items() 
                          if w not in common_words and len(w) > 2}
        error_word_freq[error_type] = Counter(filtered_counts).most_common(15)
    
    return {
        'confidence_analysis': {
            'correct_mean': float(np.mean(correct_confidence)),
            'correct_std': float(np.std(correct_confidence)),
            'incorrect_mean': float(np.mean(incorrect_confidence)),
            'incorrect_std': float(np.std(incorrect_confidence))
        },
        'length_analysis': {
            'correct_mean': float(np.mean(correct_lengths)),
            'incorrect_mean': float(np.mean(incorrect_lengths))
        },
        'error_word_frequencies': error_word_freq,
        'num_correct': len(correct),
        'num_incorrect': len(incorrect),
        'error_rate': len(incorrect) / (len(correct) + len(incorrect))
    }


def get_representative_errors(predictions: list, n_per_type: int = 3) -> dict:
    """Get representative examples of each error type."""
    errors_by_type = defaultdict(list)
    
    for p in predictions:
        if not p['correct']:
            key = f"{p['true_label']}_as_{p['predicted_label']}"
            errors_by_type[key].append(p)
    
    representative = {}
    for error_type, errors in errors_by_type.items():
        # Sort by confidence (most confident errors are most interesting)
        sorted_errors = sorted(errors, key=lambda x: -x['confidence'])
        representative[error_type] = [
            {
                'lyrics_preview': e['lyrics'][:300] + '...' if len(e['lyrics']) > 300 else e['lyrics'],
                'title': e['title'],
                'artist': e['artist'],
                'confidence': e['confidence'],
                'class_probabilities': e['class_probabilities']
            }
            for e in sorted_errors[:n_per_type]
        ]
    
    return representative


def analyze_boundary_cases(predictions: list) -> dict:
    """Analyze cases near decision boundaries (low confidence)."""
    # Sort by confidence
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'])
    
    # Get lowest confidence predictions
    boundary_cases = []
    for p in sorted_preds[:20]:  # 20 lowest confidence
        boundary_cases.append({
            'lyrics_preview': p['lyrics'][:200],
            'true_label': p['true_label'],
            'predicted_label': p['predicted_label'],
            'correct': p['correct'],
            'confidence': p['confidence'],
            'class_probabilities': p['class_probabilities']
        })
    
    return {
        'boundary_cases': boundary_cases,
        'low_confidence_error_rate': sum(1 for p in sorted_preds[:20] if not p['correct']) / 20
    }


def compare_within_cross_errors(
    within_predictions: list,
    cross_predictions: dict
) -> dict:
    """Compare error patterns within-genre vs cross-genre."""
    
    within_errors = [p for p in within_predictions if not p['correct']]
    within_confusion = analyze_confusion_patterns(within_predictions)
    
    cross_results = {}
    for target_genre, preds in cross_predictions.items():
        cross_errors = [p for p in preds if not p['correct']]
        cross_confusion = analyze_confusion_patterns(preds)
        
        cross_results[target_genre] = {
            'error_rate': len(cross_errors) / len(preds) if preds else 0,
            'top_confusions': cross_confusion['confusion_pairs'][:5]
        }
    
    return {
        'within_genre': {
            'error_rate': len(within_errors) / len(within_predictions) if within_predictions else 0,
            'top_confusions': within_confusion['confusion_pairs'][:5]
        },
        'cross_genre': cross_results
    }


def plot_confusion_heatmaps(all_predictions: dict, output_path: Path):
    """Plot confusion matrices for each genre (within-genre evaluation)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, genre in enumerate(GENRES):
        ax = axes[idx // 2, idx % 2]
        
        if genre not in all_predictions or 'within' not in all_predictions[genre]:
            continue
            
        preds = all_predictions[genre]['within']
        
        y_true = [p['true_label'] for p in preds]
        y_pred = [p['predicted_label'] for p in preds]
        
        cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=EMOTION_LABELS,
            yticklabels=EMOTION_LABELS,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title(f'{genre.upper()}\n(Within-Genre)', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.suptitle('Normalized Confusion Matrices by Genre (BOW Model)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confidence_distributions(all_predictions: dict, output_path: Path):
    """Plot confidence distributions for correct vs incorrect predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, genre in enumerate(GENRES):
        ax = axes[idx // 2, idx % 2]
        
        if genre not in all_predictions or 'within' not in all_predictions[genre]:
            continue
            
        preds = all_predictions[genre]['within']
        
        correct_conf = [p['confidence'] for p in preds if p['correct']]
        incorrect_conf = [p['confidence'] for p in preds if not p['correct']]
        
        ax.hist(correct_conf, bins=20, alpha=0.7, label=f'Correct (n={len(correct_conf)})', 
                color='#27ae60', density=True)
        ax.hist(incorrect_conf, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})', 
                color='#e74c3c', density=True)
        
        ax.axvline(np.mean(correct_conf), color='#27ae60', linestyle='--', linewidth=2)
        ax.axvline(np.mean(incorrect_conf), color='#e74c3c', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Density')
        ax.set_title(f'{genre.upper()}', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.suptitle('Confidence Distribution: Correct vs. Incorrect Predictions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_type_frequency(all_predictions: dict, output_path: Path):
    """Plot frequency of each error type across genres."""
    error_counts = defaultdict(lambda: defaultdict(int))
    
    for genre, genre_preds in all_predictions.items():
        if 'within' not in genre_preds:
            continue
        for p in genre_preds['within']:
            if not p['correct']:
                error_type = f"{p['true_label']}→{p['predicted_label']}"
                error_counts[genre][error_type] += 1
    
    # Get all error types
    all_error_types = set()
    for genre_errors in error_counts.values():
        all_error_types.update(genre_errors.keys())
    
    # Create DataFrame
    error_df = pd.DataFrame(index=sorted(all_error_types), columns=GENRES)
    for genre in GENRES:
        for error_type in all_error_types:
            error_df.loc[error_type, genre] = error_counts[genre].get(error_type, 0)
    
    error_df = error_df.fillna(0).astype(int)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    error_df.plot(kind='barh', ax=ax, width=0.8)
    
    ax.set_xlabel('Number of Errors')
    ax.set_ylabel('Error Type (True → Predicted)')
    ax.set_title('Error Type Frequency by Genre (Within-Genre, BOW)', fontweight='bold')
    ax.legend(title='Genre')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_genre_error_comparison(all_predictions: dict, output_path: Path):
    """Compare error rates within-genre vs cross-genre for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, train_genre in enumerate(GENRES):
        ax = axes[idx // 2, idx % 2]
        
        if train_genre not in all_predictions:
            continue
        
        # Get error rates
        error_rates = {}
        for test_genre in GENRES:
            key = 'within' if test_genre == train_genre else test_genre
            if key in all_predictions[train_genre]:
                preds = all_predictions[train_genre][key]
                errors = sum(1 for p in preds if not p['correct'])
                error_rates[test_genre] = errors / len(preds) if preds else 0
        
        # Plot
        genres = list(error_rates.keys())
        rates = [error_rates[g] for g in genres]
        colors = ['#27ae60' if g == train_genre else '#e74c3c' for g in genres]
        
        bars = ax.bar(genres, rates, color=colors)
        
        ax.set_ylabel('Error Rate')
        ax.set_title(f'Model trained on: {train_genre.upper()}', fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Error Rates: Within-Genre (green) vs. Cross-Genre (red)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_misclassification_word_clouds(error_characteristics: dict, output_path: Path):
    """Plot top words associated with each error type."""
    # Collect all error types across genres
    all_error_words = defaultdict(Counter)
    
    for genre, chars in error_characteristics.items():
        if 'error_word_frequencies' in chars:
            for error_type, word_list in chars['error_word_frequencies'].items():
                for word, count in word_list:
                    all_error_words[error_type][word] += count
    
    # Get top error types
    top_error_types = sorted(all_error_words.keys(), 
                             key=lambda x: sum(all_error_words[x].values()), 
                             reverse=True)[:6]
    
    if not top_error_types:
        print("  No error word data to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, error_type in enumerate(top_error_types):
        ax = axes[idx]
        
        word_counts = all_error_words[error_type].most_common(12)
        if not word_counts:
            continue
            
        words, counts = zip(*word_counts)
        
        ax.barh(range(len(words)), counts, color='#3498db')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title(error_type.replace('_', ' → ').replace(' as ', ' → '), fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(top_error_types), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Top Words in Misclassified Lyrics by Error Type', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(
    all_predictions: dict,
    error_characteristics: dict,
    representative_errors: dict,
    confusion_analysis: dict,
    output_dir: Path
):
    """Save all analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Error analysis summary
    error_summary = {
        'by_genre': {},
        'overall': {
            'total_predictions': 0,
            'total_errors': 0,
            'overall_error_rate': 0
        }
    }
    
    total_preds = 0
    total_errors = 0
    
    for genre, genre_preds in all_predictions.items():
        if 'within' in genre_preds:
            preds = genre_preds['within']
            errors = sum(1 for p in preds if not p['correct'])
            error_summary['by_genre'][genre] = {
                'predictions': len(preds),
                'errors': errors,
                'error_rate': errors / len(preds) if preds else 0,
                'confusion_patterns': confusion_analysis.get(genre, {})
            }
            total_preds += len(preds)
            total_errors += errors
    
    error_summary['overall']['total_predictions'] = total_preds
    error_summary['overall']['total_errors'] = total_errors
    error_summary['overall']['overall_error_rate'] = total_errors / total_preds if total_preds else 0
    
    with open(output_dir / 'exp4_error_analysis.json', 'w') as f:
        json.dump(error_summary, f, indent=2)
    
    # Error characteristics
    with open(output_dir / 'exp4_error_characteristics.json', 'w') as f:
        json.dump(error_characteristics, f, indent=2)
    
    # Representative errors (with lyrics examples)
    with open(output_dir / 'exp4_misclassified_examples.json', 'w') as f:
        json.dump(representative_errors, f, indent=2)
    
    # Metadata
    metadata = {
        'experiment': 'Experiment 4: Error Analysis',
        'timestamp': datetime.now().isoformat(),
        'genres': GENRES,
        'emotions': EMOTION_LABELS
    }
    
    with open(output_dir / 'exp4_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Run Experiment 4: Error Analysis."""
    print("=" * 70)
    print("EXPERIMENT 4: ERROR ANALYSIS - UNDERSTANDING MODEL MISTAKES")
    print("=" * 70)
    
    # Setup paths
    metrics_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data splits
    print("\n[1/7] Loading data splits...")
    splits = load_splits(input_dir=project_root / "data" / "splits")
    
    # Train models and collect predictions
    print("\n[2/7] Training models and collecting predictions...")
    all_predictions = {}
    
    for train_genre in GENRES:
        print(f"\n  Training on {train_genre}...")
        
        train_df = pd.concat([
            splits[train_genre]['train'],
            splits[train_genre]['val']
        ])
        
        model, vectorizer = train_bow_model(train_df)
        all_predictions[train_genre] = {}
        
        # Within-genre evaluation
        test_df = splits[train_genre]['test']
        within_preds = get_predictions_with_confidence(model, vectorizer, test_df)
        all_predictions[train_genre]['within'] = within_preds
        
        within_errors = sum(1 for p in within_preds if not p['correct'])
        print(f"    Within-genre: {within_errors}/{len(within_preds)} errors "
              f"({within_errors/len(within_preds)*100:.1f}%)")
        
        # Cross-genre evaluation
        for test_genre in GENRES:
            if test_genre != train_genre:
                test_df = splits[test_genre]['test']
                cross_preds = get_predictions_with_confidence(model, vectorizer, test_df)
                all_predictions[train_genre][test_genre] = cross_preds
                
                cross_errors = sum(1 for p in cross_preds if not p['correct'])
                print(f"    → {test_genre}: {cross_errors}/{len(cross_preds)} errors "
                      f"({cross_errors/len(cross_preds)*100:.1f}%)")
    
    # Analyze confusion patterns
    print("\n[3/7] Analyzing confusion patterns...")
    confusion_analysis = {}
    
    for genre in GENRES:
        preds = all_predictions[genre]['within']
        confusion_analysis[genre] = analyze_confusion_patterns(preds)
        
        print(f"\n  {genre} top confusions:")
        for conf in confusion_analysis[genre]['confusion_pairs'][:3]:
            print(f"    {conf['true']} → {conf['predicted']}: {conf['count']} times")
    
    # Analyze error characteristics
    print("\n[4/7] Analyzing error characteristics...")
    error_characteristics = {}
    
    for genre in GENRES:
        preds = all_predictions[genre]['within']
        model, vectorizer = train_bow_model(pd.concat([
            splits[genre]['train'], splits[genre]['val']
        ]))
        error_characteristics[genre] = analyze_error_characteristics(preds, vectorizer)
        
        chars = error_characteristics[genre]
        if 'confidence_analysis' in chars:
            print(f"\n  {genre}:")
            print(f"    Correct confidence: {chars['confidence_analysis']['correct_mean']:.3f} "
                  f"± {chars['confidence_analysis']['correct_std']:.3f}")
            print(f"    Incorrect confidence: {chars['confidence_analysis']['incorrect_mean']:.3f} "
                  f"± {chars['confidence_analysis']['incorrect_std']:.3f}")
    
    # Get representative errors
    print("\n[5/7] Collecting representative error examples...")
    representative_errors = {}
    
    for genre in GENRES:
        preds = all_predictions[genre]['within']
        representative_errors[genre] = get_representative_errors(preds, n_per_type=3)
        
        num_types = len(representative_errors[genre])
        print(f"  {genre}: {num_types} distinct error types")
    
    # Generate visualizations
    print("\n[6/7] Generating visualizations...")
    
    plot_confusion_heatmaps(all_predictions, figures_dir / 'exp4_confusion_heatmaps.png')
    print("  Saved: exp4_confusion_heatmaps.png")
    
    plot_confidence_distributions(all_predictions, figures_dir / 'exp4_confidence_distributions.png')
    print("  Saved: exp4_confidence_distributions.png")
    
    plot_error_type_frequency(all_predictions, figures_dir / 'exp4_error_type_frequency.png')
    print("  Saved: exp4_error_type_frequency.png")
    
    plot_cross_genre_error_comparison(all_predictions, figures_dir / 'exp4_cross_genre_errors.png')
    print("  Saved: exp4_cross_genre_errors.png")
    
    plot_misclassification_word_clouds(error_characteristics, figures_dir / 'exp4_error_words.png')
    print("  Saved: exp4_error_words.png")
    
    # Save results
    print("\n[7/7] Saving results...")
    save_results(
        all_predictions,
        error_characteristics,
        representative_errors,
        confusion_analysis,
        metrics_dir
    )
    print(f"  Results saved to: {metrics_dir}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)
    
    # Overall statistics
    total_within_preds = 0
    total_within_errors = 0
    for genre in GENRES:
        preds = all_predictions[genre]['within']
        total_within_preds += len(preds)
        total_within_errors += sum(1 for p in preds if not p['correct'])
    
    print(f"""
Summary:

1. OVERALL ERROR RATE (within-genre)
   - Total predictions: {total_within_preds}
   - Total errors: {total_within_errors}
   - Error rate: {total_within_errors/total_within_preds*100:.1f}%

2. TOP CONFUSION PATTERNS (across all genres)""")
    
    # Aggregate confusions
    all_confusions = defaultdict(int)
    for genre, analysis in confusion_analysis.items():
        for conf in analysis['confusion_pairs']:
            key = f"{conf['true']} → {conf['predicted']}"
            all_confusions[key] += conf['count']
    
    sorted_confusions = sorted(all_confusions.items(), key=lambda x: -x[1])[:5]
    for conf, count in sorted_confusions:
        print(f"   - {conf}: {count} times")
    
    print("""
3. CONFIDENCE PATTERNS
   - Incorrect predictions have lower confidence
   - Low-confidence predictions more likely to be wrong

4. KEY INSIGHT
   - Check exp4_misclassified_examples.json for specific lyrics
   - Error words reveal vocabulary causing confusion
""")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()


