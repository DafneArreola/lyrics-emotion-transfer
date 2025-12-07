"""
Data splitting utilities for cross-genre emotion classification experiments.

This module provides stratified train/test splitting that supports:
1. Within-genre evaluation (Experiments 1-2)
2. Cross-genre transfer evaluation (Experiment 3)
3. Reproducible splits with fixed random seeds
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from data_loader import get_project_root


DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1  # Proportion of training set for validation


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    stratify_col: str = 'emotion_label',
    random_state: int = DEFAULT_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split maintaining emotion distribution.
    
    Args:
        df: Annotated DataFrame with emotion labels
        test_size: Proportion for test set
        val_size: Proportion of training set for validation
        stratify_col: Column to stratify on
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Remove samples without labels
    df_labeled = df[df[stratify_col].notna()].copy()
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df_labeled,
        test_size=test_size,
        stratify=df_labeled[stratify_col],
        random_state=random_state
    )
    
    # Second split: train vs val
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[stratify_col],
        random_state=random_state
    )
    
    return train, val, test


def create_genre_splits(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_SEED
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create separate train/val/test splits for each genre.
    
    This supports Experiments 1-2 where models are trained on each genre
    separately and evaluated on all genres.
    
    Args:
        df: Annotated DataFrame with 'genre' and 'emotion_label' columns
        test_size: Proportion for test set
        val_size: Proportion of training set for validation
        random_state: Random seed
    
    Returns:
        Nested dictionary: {genre: {'train': df, 'val': df, 'test': df}}
    """
    genres = df['genre'].unique()
    splits = {}
    
    for genre in genres:
        genre_df = df[df['genre'] == genre].copy()
        
        try:
            train, val, test = create_stratified_split(
                genre_df,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state
            )
            
            splits[genre] = {
                'train': train.reset_index(drop=True),
                'val': val.reset_index(drop=True),
                'test': test.reset_index(drop=True)
            }
            
            print(f"{genre}: train={len(train)}, val={len(val)}, test={len(test)}")
            
        except ValueError as e:
            print(f"Warning: Could not stratify {genre} - {e}")
            # Fall back to non-stratified split
            train_val, test = train_test_split(
                genre_df, test_size=test_size, random_state=random_state
            )
            train, val = train_test_split(
                train_val, test_size=val_size, random_state=random_state
            )
            splits[genre] = {
                'train': train.reset_index(drop=True),
                'val': val.reset_index(drop=True),
                'test': test.reset_index(drop=True)
            }
    
    return splits


def save_splits(
    splits: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Optional[Path] = None,
    format: str = 'csv'
) -> None:
    """
    Save split DataFrames to disk.
    
    Args:
        splits: Output from create_genre_splits()
        output_dir: Directory for output files
        format: 'csv' or 'json'
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "splits"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for genre, genre_splits in splits.items():
        for split_name, split_df in genre_splits.items():
            filename = f"{genre}_{split_name}.{format}"
            filepath = output_dir / filename
            
            if format == 'csv':
                split_df.to_csv(filepath, index=False)
            elif format == 'json':
                split_df.to_json(filepath, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    # Save metadata
    metadata = {
        'genres': list(splits.keys()),
        'splits': ['train', 'val', 'test'],
        'format': format,
        'sizes': {
            genre: {
                split_name: len(split_df)
                for split_name, split_df in genre_splits.items()
            }
            for genre, genre_splits in splits.items()
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved splits to: {output_dir}")


def load_splits(
    input_dir: Optional[Path] = None,
    format: str = 'csv'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load previously saved splits from disk.
    
    Args:
        input_dir: Directory containing split files
        format: 'csv' or 'json'
    
    Returns:
        Nested dictionary matching create_genre_splits() output
    """
    if input_dir is None:
        input_dir = get_project_root() / "data" / "splits"
    
    input_dir = Path(input_dir)
    
    # Load metadata
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    splits = {}
    
    for genre in metadata['genres']:
        splits[genre] = {}
        for split_name in metadata['splits']:
            filename = f"{genre}_{split_name}.{format}"
            filepath = input_dir / filename
            
            if format == 'csv':
                splits[genre][split_name] = pd.read_csv(filepath)
            elif format == 'json':
                splits[genre][split_name] = pd.read_json(filepath)
    
    return splits


def get_cross_genre_eval_pairs(genres: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all train-test genre pairs for cross-genre evaluation.
    
    Supports Experiment 3 (Asymmetric Transfer Analysis).
    
    Args:
        genres: List of genre names
    
    Returns:
        List of (train_genre, test_genre) tuples including same-genre pairs
    """
    pairs = []
    for train_genre in genres:
        for test_genre in genres:
            pairs.append((train_genre, test_genre))
    
    return pairs


def report_split_statistics(splits: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """
    Print detailed statistics about the data splits.
    
    Args:
        splits: Output from create_genre_splits()
    """
    print("\n" + "="*60)
    print("DATA SPLIT STATISTICS")
    print("="*60)
    
    for genre, genre_splits in splits.items():
        print(f"\n--- {genre.upper()} ---")
        
        for split_name, df in genre_splits.items():
            print(f"\n  {split_name.capitalize()} set (n={len(df)}):")
            
            if 'emotion_label' in df.columns:
                dist = df['emotion_label'].value_counts(normalize=True) * 100
                for emotion, pct in dist.items():
                    print(f"    {emotion}: {pct:.1f}%")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    total_train = sum(len(s['train']) for s in splits.values())
    total_val = sum(len(s['val']) for s in splits.values())
    total_test = sum(len(s['test']) for s in splits.values())
    
    print(f"\nTotal samples: {total_train + total_val + total_test}")
    print(f"  Training:   {total_train}")
    print(f"  Validation: {total_val}")
    print(f"  Test:       {total_test}")


if __name__ == "__main__":
    from data_loader import load_all_genres, create_songs_dataframe
    from emotion_annotator import annotate_dataframe, load_nrc_vad_lexicon
    
    print("Loading and annotating data...")
    all_data = load_all_genres()
    df = create_songs_dataframe(all_data)
    lexicon = load_nrc_vad_lexicon()
    df_annotated = annotate_dataframe(df, lexicon, verbose=False)
    
    print("\nCreating genre-specific splits...")
    splits = create_genre_splits(df_annotated)
    
    report_split_statistics(splits)
    
    print("\nCross-genre evaluation pairs:")
    genres = list(splits.keys())
    pairs = get_cross_genre_eval_pairs(genres)
    print(f"  Total pairs: {len(pairs)} ({len(genres)}x{len(genres)} grid)")
    
    # Save splits
    save_splits(splits)


