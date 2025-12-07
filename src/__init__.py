"""
Lyrics Emotion Transfer - Source Module

This package provides utilities for cross-genre emotion classification
in song lyrics, including data loading, emotion annotation using the
NRC-VAD lexicon, and data splitting for experiments.
"""

from .data_loader import (
    load_genre_data,
    load_all_genres,
    load_nrc_vad_lexicon,
    create_songs_dataframe,
    get_data_summary,
    get_project_root
)

from .emotion_annotator import (
    preprocess_lyrics,
    compute_lyrics_vad,
    assign_emotion_label,
    annotate_dataframe,
    get_emotion_distribution_by_genre,
    analyze_lexicon_coverage_by_genre,
    compute_adaptive_thresholds
)

from .data_splits import (
    create_stratified_split,
    create_genre_splits,
    save_splits,
    load_splits,
    get_cross_genre_eval_pairs,
    report_split_statistics
)

__version__ = "0.1.0"


