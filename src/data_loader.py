"""
Data loading utilities for lyrics-emotion-transfer project.

This module provides functions for loading raw lyrics data from JSON files
and the NRC-VAD lexicon for emotion annotation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


def get_project_root() -> Path:
    """
    Dynamically detect the project root directory.
    Assumes this file is in src/ subdirectory.
    """
    return Path(__file__).parent.parent


def load_genre_data(genre: str, data_dir: Optional[Path] = None) -> Dict:
    """
    Load raw lyrics data for a single genre.
    
    Args:
        genre: One of 'classic-pop', 'hip-hop', 'hard-rock', 'country'
        data_dir: Optional path to data directory. Defaults to project data/raw/
    
    Returns:
        Dictionary containing genre metadata and list of songs
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"
    
    filepath = data_dir / f"{genre}.json"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_all_genres(data_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """
    Load lyrics data for all four genres.
    
    Args:
        data_dir: Optional path to data directory
    
    Returns:
        Dictionary mapping genre names to lists of song dictionaries
    """
    genres = ['classic-pop', 'hip-hop', 'hard-rock', 'country']
    all_data = {}
    
    for genre in genres:
        try:
            genre_data = load_genre_data(genre, data_dir)
            all_data[genre] = genre_data.get('songs', [])
            print(f"Loaded {len(all_data[genre])} songs from {genre}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            all_data[genre] = []
    
    return all_data


def load_nrc_vad_lexicon(lexicon_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the NRC Valence-Arousal-Dominance lexicon.
    
    The lexicon format is tab-separated with columns:
    term, valence, arousal, dominance
    
    Values are on a scale from approximately -1 to 1.
    
    Args:
        lexicon_path: Optional path to lexicon file
    
    Returns:
        DataFrame with word as index and V, A, D scores as columns
    """
    if lexicon_path is None:
        lexicon_path = get_project_root() / "data" / "lexicons" / "NRC-VAD-Lexicon.txt"
    
    lexicon_path = Path(lexicon_path)
    
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
    
    # NRC-VAD format: term\tvalence\tarousal\tdominance (with header row)
    lexicon = pd.read_csv(
        lexicon_path,
        sep='\t',
        header=0  # Use first row as header
    )
    
    # Rename columns to standardized names if needed
    lexicon.columns = [col.lower().strip() for col in lexicon.columns]
    
    # Handle 'term' vs 'word' column naming
    if 'term' in lexicon.columns:
        lexicon = lexicon.rename(columns={'term': 'word'})
    
    # Lowercase all words for consistent matching
    lexicon['word'] = lexicon['word'].str.lower().str.strip()
    lexicon = lexicon.set_index('word')
    
    return lexicon


def create_songs_dataframe(all_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Convert loaded genre data into a unified DataFrame.
    
    Args:
        all_data: Dictionary from load_all_genres()
    
    Returns:
        DataFrame with columns: genre, artist, title, lyrics, year
    """
    rows = []
    
    for genre, songs in all_data.items():
        for song in songs:
            rows.append({
                'genre': genre,
                'artist': song.get('artist', ''),
                'title': song.get('title', ''),
                'lyrics': song.get('lyrics', ''),
                'year': song.get('year', None)
            })
    
    df = pd.DataFrame(rows)
    
    # Basic cleaning: remove songs with empty lyrics
    df = df[df['lyrics'].str.strip().str.len() > 0]
    
    return df


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Songs DataFrame from create_songs_dataframe()
    
    Returns:
        DataFrame with per-genre statistics
    """
    summary = df.groupby('genre').agg({
        'title': 'count',
        'lyrics': lambda x: x.str.split().str.len().mean(),
        'year': ['min', 'max']
    }).round(2)
    
    summary.columns = ['song_count', 'avg_word_count', 'year_min', 'year_max']
    
    return summary


if __name__ == "__main__":
    # Example usage and data exploration
    print("Loading all genre data...")
    all_data = load_all_genres()
    
    print("\nCreating unified DataFrame...")
    df = create_songs_dataframe(all_data)
    
    print(f"\nTotal songs: {len(df)}")
    print("\nData Summary:")
    print(get_data_summary(df))
    
    print("\nLoading NRC-VAD lexicon...")
    try:
        lexicon = load_nrc_vad_lexicon()
        print(f"Lexicon size: {len(lexicon)} words")
        print(f"Sample entries:\n{lexicon.head()}")
    except FileNotFoundError as e:
        print(f"Note: {e}")

