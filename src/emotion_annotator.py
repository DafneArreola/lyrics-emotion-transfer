"""
Emotion annotation for lyrics using the NRC-VAD lexicon.

This module implements emotion labeling based on Russell's Valence-Arousal model,
mapping continuous V-A scores to four discrete emotion categories:
    - happy:   +V, +A (high valence, high arousal)
    - angry:   -V, +A (low valence, high arousal)  
    - sad:     -V, -A (low valence, low arousal)
    - relaxed: +V, -A (high valence, low arousal)

The approach follows Hu et al. (2009) and aligns with established
music emotion research conventions.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter

from data_loader import load_nrc_vad_lexicon, get_project_root


# Default thresholds (will be overridden by adaptive thresholds)
VALENCE_THRESHOLD = 0.0
AROUSAL_THRESHOLD = 0.0

EMOTION_QUADRANTS = {
    'happy': {'valence': 'high', 'arousal': 'high'},    # +V+A
    'angry': {'valence': 'low', 'arousal': 'high'},     # -V+A
    'sad': {'valence': 'low', 'arousal': 'low'},        # -V-A
    'relaxed': {'valence': 'high', 'arousal': 'low'}    # +V-A
}


def preprocess_lyrics(text: str) -> List[str]:
    """
    Preprocess lyrics text for lexicon matching.
    
    Args:
        text: Raw lyrics string
    
    Returns:
        List of lowercase tokens
    """
    # Remove common lyrics artifacts (e.g., [Verse], [Chorus])
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove punctuation except apostrophes (for contractions)
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Lowercase and tokenize
    tokens = text.lower().split()
    
    return tokens


def compute_lyrics_vad(
    lyrics: str, 
    lexicon: pd.DataFrame,
    return_details: bool = False
) -> Dict:
    """
    Compute aggregate Valence-Arousal-Dominance scores for lyrics.
    
    Uses mean pooling over all matched words, following standard practice
    in lexicon-based sentiment analysis.
    
    Args:
        lyrics: Raw lyrics text
        lexicon: NRC-VAD lexicon DataFrame (word-indexed)
        return_details: If True, include matched words in output
    
    Returns:
        Dictionary with 'valence', 'arousal', 'dominance', 'matched_count', 
        'total_count', and optionally 'matched_words'
    """
    tokens = preprocess_lyrics(lyrics)
    
    matched_scores = []
    matched_words = []
    
    for token in tokens:
        if token in lexicon.index:
            scores = lexicon.loc[token]
            matched_scores.append({
                'valence': scores['valence'],
                'arousal': scores['arousal'],
                'dominance': scores['dominance']
            })
            matched_words.append(token)
    
    result = {
        'matched_count': len(matched_scores),
        'total_count': len(tokens),
        'coverage': len(matched_scores) / len(tokens) if tokens else 0
    }
    
    if matched_scores:
        result['valence'] = np.mean([s['valence'] for s in matched_scores])
        result['arousal'] = np.mean([s['arousal'] for s in matched_scores])
        result['dominance'] = np.mean([s['dominance'] for s in matched_scores])
    else:
        result['valence'] = None
        result['arousal'] = None
        result['dominance'] = None
    
    if return_details:
        result['matched_words'] = matched_words
    
    return result


def assign_emotion_label(
    valence: Optional[float], 
    arousal: Optional[float],
    v_threshold: float = VALENCE_THRESHOLD,
    a_threshold: float = AROUSAL_THRESHOLD
) -> Optional[str]:
    """
    Map V-A scores to discrete emotion quadrant.
    
    Args:
        valence: Valence score (typically -1 to 1 scale)
        arousal: Arousal score (typically -1 to 1 scale)
        v_threshold: Threshold for high/low valence classification
        a_threshold: Threshold for high/low arousal classification
    
    Returns:
        Emotion label ('happy', 'angry', 'sad', 'relaxed') or None if no score
    """
    if valence is None or arousal is None:
        return None
    
    high_valence = valence >= v_threshold
    high_arousal = arousal >= a_threshold
    
    if high_valence and high_arousal:
        return 'happy'
    elif not high_valence and high_arousal:
        return 'angry'
    elif not high_valence and not high_arousal:
        return 'sad'
    else:  # high_valence and not high_arousal
        return 'relaxed'


def compute_adaptive_thresholds(
    df: pd.DataFrame,
    method: str = 'median'
) -> Tuple[float, float]:
    """
    Compute data-driven thresholds for emotion classification.
    
    Using median-based thresholds is methodologically defensible as it:
    1. Adapts to actual data distribution rather than assuming scale centering
    2. Guarantees more balanced class distributions
    3. Is reproducible and documentable
    
    Args:
        df: DataFrame with 'valence' and 'arousal' columns (pre-computed VAD scores)
        method: 'median' (recommended) or 'mean'
    
    Returns:
        Tuple of (valence_threshold, arousal_threshold)
    """
    valid_df = df[df['valence'].notna() & df['arousal'].notna()]
    
    if method == 'median':
        v_threshold = valid_df['valence'].median()
        a_threshold = valid_df['arousal'].median()
    elif method == 'mean':
        v_threshold = valid_df['valence'].mean()
        a_threshold = valid_df['arousal'].mean()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")
    
    return v_threshold, a_threshold


def annotate_dataframe(
    df: pd.DataFrame,
    lexicon: pd.DataFrame,
    lyrics_col: str = 'lyrics',
    adaptive_threshold: bool = True,
    threshold_method: str = 'median',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Annotate a DataFrame of songs with emotion labels.
    
    Args:
        df: DataFrame with lyrics column
        lexicon: NRC-VAD lexicon DataFrame
        lyrics_col: Name of column containing lyrics text
        adaptive_threshold: If True, compute thresholds from data distribution
        threshold_method: Method for adaptive thresholds ('median' or 'mean')
        verbose: Print progress updates
    
    Returns:
        DataFrame with added columns: valence, arousal, dominance, 
        coverage, emotion_label
    """
    df = df.copy()
    
    # Step 1: Compute VAD scores for all songs
    if verbose:
        print("  Computing VAD scores...")
    
    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processing song {idx + 1}/{total}")
        
        vad = compute_lyrics_vad(row[lyrics_col], lexicon)
        results.append(vad)
    
    results_df = pd.DataFrame(results)
    
    for col in ['valence', 'arousal', 'dominance', 'coverage']:
        df[col] = results_df[col].values
    
    # Step 2: Compute thresholds
    if adaptive_threshold:
        v_threshold, a_threshold = compute_adaptive_thresholds(
            df, method=threshold_method
        )
        if verbose:
            print(f"\n  Adaptive thresholds ({threshold_method}):")
            print(f"    Valence threshold: {v_threshold:.4f}")
            print(f"    Arousal threshold: {a_threshold:.4f}")
    else:
        v_threshold = VALENCE_THRESHOLD
        a_threshold = AROUSAL_THRESHOLD
        if verbose:
            print(f"\n  Using fixed thresholds: V={v_threshold}, A={a_threshold}")
    
    # Store thresholds in DataFrame attributes for reference
    df.attrs['valence_threshold'] = v_threshold
    df.attrs['arousal_threshold'] = a_threshold
    df.attrs['threshold_method'] = threshold_method if adaptive_threshold else 'fixed'
    
    # Step 3: Assign emotion labels using computed thresholds
    df['emotion_label'] = df.apply(
        lambda row: assign_emotion_label(
            row['valence'], row['arousal'], v_threshold, a_threshold
        ),
        axis=1
    )
    
    # Report annotation statistics
    if verbose:
        labeled = df['emotion_label'].notna().sum()
        print(f"\nAnnotation complete:")
        print(f"  Successfully labeled: {labeled}/{len(df)} ({labeled/len(df)*100:.1f}%)")
        print(f"  Average lexicon coverage: {df['coverage'].mean()*100:.1f}%")
        print(f"\nEmotion distribution:")
        print(df['emotion_label'].value_counts())
    
    return df


def get_emotion_distribution_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute emotion label distribution per genre.
    
    Args:
        df: Annotated DataFrame with 'genre' and 'emotion_label' columns
    
    Returns:
        Cross-tabulation DataFrame showing counts and percentages
    """
    # Raw counts
    counts = pd.crosstab(df['genre'], df['emotion_label'])
    
    # Normalize to percentages within each genre
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    return percentages.round(2)


def analyze_lexicon_coverage_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze lexicon coverage statistics by genre.
    
    Useful for understanding potential biases in emotion annotation
    across different lyrical styles.
    
    Args:
        df: Annotated DataFrame with 'genre' and 'coverage' columns
    
    Returns:
        DataFrame with coverage statistics per genre
    """
    coverage_stats = df.groupby('genre')['coverage'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]) * 100  # Convert to percentages
    
    return coverage_stats.round(2)


if __name__ == "__main__":
    from data_loader import load_all_genres, create_songs_dataframe
    
    print("Loading data...")
    all_data = load_all_genres()
    df = create_songs_dataframe(all_data)
    
    print("\nLoading NRC-VAD lexicon...")
    lexicon = load_nrc_vad_lexicon()
    
    print(f"\nAnnotating {len(df)} songs with adaptive thresholds...")
    df_annotated = annotate_dataframe(df, lexicon, adaptive_threshold=True)
    
    print("\nEmotion distribution by genre:")
    print(get_emotion_distribution_by_genre(df_annotated))
    
    print("\nLexicon coverage by genre:")
    print(analyze_lexicon_coverage_by_genre(df_annotated))
    
    # Save annotated data
    output_path = get_project_root() / "data" / "processed" / "annotated_lyrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_annotated.to_csv(output_path, index=False)
    print(f"\nSaved annotated data to: {output_path}")


