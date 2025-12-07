#!/usr/bin/env python3
"""
Main preprocessing pipeline for lyrics-emotion-transfer project.

This script runs the complete data preparation workflow:
1. Load raw lyrics data from JSON files
2. Annotate with emotion labels using NRC-VAD lexicon
3. Create stratified train/val/test splits per genre
4. Generate summary statistics and visualizations

Usage:
    python experiments/scripts/run_preprocessing.py

Outputs:
    - data/processed/annotated_lyrics.csv
    - data/splits/{genre}_{train|val|test}.csv
    - data/splits/metadata.json
    - results/figures/emotion_distribution.png
"""

import sys
import warnings
from pathlib import Path

# Dynamically add src directory to path
# Works regardless of where script is called from
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import (
    load_all_genres, 
    create_songs_dataframe, 
    load_nrc_vad_lexicon,
    get_data_summary
)
from emotion_annotator import (
    annotate_dataframe,
    get_emotion_distribution_by_genre,
    analyze_lexicon_coverage_by_genre,
    compute_adaptive_thresholds
)
from data_splits import (
    create_genre_splits,
    save_splits,
    report_split_statistics
)


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save exploratory visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Emotion distribution by genre
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    emotion_dist = pd.crosstab(df['genre'], df['emotion_label'], normalize='index') * 100
    emotion_dist.plot(kind='bar', stacked=True, ax=axes[0], 
                      colormap='RdYlGn', edgecolor='white')
    axes[0].set_title('Emotion Distribution by Genre')
    axes[0].set_xlabel('Genre')
    axes[0].set_ylabel('Percentage')
    axes[0].legend(title='Emotion', bbox_to_anchor=(1.02, 1))
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Valence-Arousal scatter by genre
    labeled_df = df[df['valence'].notna()]
    for genre in labeled_df['genre'].unique():
        genre_data = labeled_df[labeled_df['genre'] == genre]
        axes[1].scatter(
            genre_data['valence'], 
            genre_data['arousal'],
            alpha=0.5, 
            label=genre,
            s=20
        )
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Valence')
    axes[1].set_ylabel('Arousal')
    axes[1].set_title('Valence-Arousal Distribution by Genre')
    axes[1].legend()
    
    # Add quadrant labels (adjusted for -1 to 1 scale)
    axes[1].text(0.5, 0.5, 'Happy\n(+V+A)', ha='center', fontsize=9, alpha=0.7)
    axes[1].text(-0.5, 0.5, 'Angry\n(-V+A)', ha='center', fontsize=9, alpha=0.7)
    axes[1].text(-0.5, -0.5, 'Sad\n(-V-A)', ha='center', fontsize=9, alpha=0.7)
    axes[1].text(0.5, -0.5, 'Relaxed\n(+V-A)', ha='center', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Lexicon coverage histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    for genre in df['genre'].unique():
        genre_data = df[df['genre'] == genre]
        ax.hist(
            genre_data['coverage'] * 100, 
            bins=30, 
            alpha=0.5, 
            label=genre
        )
    ax.set_xlabel('Lexicon Coverage (%)')
    ax.set_ylabel('Number of Songs')
    ax.set_title('NRC-VAD Lexicon Coverage Distribution by Genre')
    ax.legend()
    plt.savefig(output_dir / 'lexicon_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations to: {output_dir}")


def main():
    """Run the complete preprocessing pipeline."""
    print("="*60)
    print("LYRICS EMOTION TRANSFER - PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load raw data
    print("\n[1/5] Loading raw lyrics data...")
    try:
        all_data = load_all_genres(data_dir=project_root / "data" / "raw")
        df = create_songs_dataframe(all_data)
        print(f"  Loaded {len(df)} songs across {df['genre'].nunique()} genres")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please ensure JSON files exist in data/raw/")
        sys.exit(1)
    
    # Step 2: Display raw data summary
    print("\n[2/5] Raw data summary:")
    print(get_data_summary(df).to_string())
    
    # Step 3: Load lexicon and annotate
    print("\n[3/5] Annotating with NRC-VAD lexicon...")
    lexicon_path = project_root / "data" / "lexicons" / "NRC-VAD-Lexicon.txt"
    try:
        lexicon = load_nrc_vad_lexicon(lexicon_path)
        print(f"  Lexicon contains {len(lexicon)} words")
        df = annotate_dataframe(df, lexicon, verbose=True)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please ensure NRC-VAD-Lexicon.txt exists in data/lexicons/")
        sys.exit(1)
    
    # Save annotated data
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "annotated_lyrics.csv", index=False)
    print(f"\n  Saved annotated data to: {processed_dir / 'annotated_lyrics.csv'}")
    
    # Step 4: Create and save splits
    print("\n[4/5] Creating stratified train/val/test splits...")
    splits = create_genre_splits(df)
    save_splits(splits, output_dir=project_root / "data" / "splits")
    report_split_statistics(splits)
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    figures_dir = project_root / "results" / "figures"
    create_visualizations(df, figures_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"""
Next steps:
  1. Review emotion distributions in results/figures/
  2. Check for class imbalance issues
  3. Run Experiment 1: python experiments/scripts/run_experiment_1.py
    """)
    
    # Print emotion distribution summary
    print("\nEmotion distribution by genre (%):")
    print(get_emotion_distribution_by_genre(df).to_string())
    
    print("\nLexicon coverage by genre (%):")
    print(analyze_lexicon_coverage_by_genre(df).to_string())


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()


