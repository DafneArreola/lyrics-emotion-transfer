"""
Emotion annotation using Valence-Arousal lexicon
Implements approach similar to MoodyLyrics paper (Çano & Morisio, 2017)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import re
from collections import Counter
import requests
import io


class ValenceArousalAnnotator:
    """
    Annotate lyrics with emotions based on Valence-Arousal model
    Uses NRC VAD Lexicon (Valence, Arousal, Dominance)
    """
    
    def __init__(self, config_path='configs/emotions.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        thresholds = self.config['annotation_thresholds']
        self.valence_threshold = thresholds['valence_threshold']
        self.arousal_threshold = thresholds['arousal_threshold']
        self.min_scored_words = thresholds['min_scored_words']
        
        # Load NRC VAD Lexicon
        self.vad_lexicon = self._load_vad_lexicon()
    
    def _load_vad_lexicon(self):
        """
        Load NRC Valence-Arousal-Dominance Lexicon
        Download from: https://saifmohammad.com/WebPages/nrc-vad.html
        """
        # Try to load local copy first
        lexicon_path = Path('data/raw/NRC-VAD-Lexicon.txt')
        
        if not lexicon_path.exists():
            print("NRC-VAD Lexicon not found locally.")
            print("Please download from: https://saifmohammad.com/WebPages/nrc-vad.html")
            print("Save to: data/raw/NRC-VAD-Lexicon.txt")
            print("\nUsing simplified example lexicon for demonstration...")
            
            # Return example lexicon structure
            return self._create_example_lexicon()
        
        # Load the actual lexicon
        vad_dict = {}
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    word, valence, arousal, dominance = parts
                    vad_dict[word.lower()] = {
                        'valence': float(valence),
                        'arousal': float(arousal),
                        'dominance': float(dominance)
                    }
        
        print(f"Loaded VAD lexicon with {len(vad_dict)} words")
        return vad_dict
    
    def _create_example_lexicon(self):
        """
        Create example VAD lexicon for demonstration
        In practice, download the full NRC-VAD lexicon
        """
        # Example words with VAD scores (0-1 scale)
        example_words = {
            # Happy words (high V, high A)
            'happy': {'valence': 0.9, 'arousal': 0.7, 'dominance': 0.6},
            'joy': {'valence': 0.9, 'arousal': 0.7, 'dominance': 0.6},
            'love': {'valence': 0.9, 'arousal': 0.6, 'dominance': 0.5},
            'excited': {'valence': 0.8, 'arousal': 0.9, 'dominance': 0.7},
            
            # Angry words (low V, high A)
            'angry': {'valence': 0.2, 'arousal': 0.8, 'dominance': 0.7},
            'hate': {'valence': 0.1, 'arousal': 0.8, 'dominance': 0.6},
            'rage': {'valence': 0.1, 'arousal': 0.9, 'dominance': 0.8},
            'furious': {'valence': 0.2, 'arousal': 0.9, 'dominance': 0.7},
            
            # Sad words (low V, low A)
            'sad': {'valence': 0.2, 'arousal': 0.3, 'dominance': 0.2},
            'depressed': {'valence': 0.1, 'arousal': 0.2, 'dominance': 0.1},
            'lonely': {'valence': 0.2, 'arousal': 0.3, 'dominance': 0.2},
            'hurt': {'valence': 0.2, 'arousal': 0.4, 'dominance': 0.3},
            
            # Relaxed words (high V, low A)
            'calm': {'valence': 0.7, 'arousal': 0.3, 'dominance': 0.5},
            'peaceful': {'valence': 0.8, 'arousal': 0.2, 'dominance': 0.5},
            'relaxed': {'valence': 0.7, 'arousal': 0.2, 'dominance': 0.5},
            'serene': {'valence': 0.8, 'arousal': 0.2, 'dominance': 0.5},
        }
        return example_words
    
    def clean_lyrics(self, text):
        """Clean and tokenize lyrics"""
        if pd.isna(text):
            return []
        
        # Remove URLs, special characters, extra whitespace
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and lowercase
        tokens = text.lower().split()
        return tokens
    
    def calculate_vad_scores(self, lyrics):
        """
        Calculate average Valence and Arousal scores for lyrics
        Returns: (valence, arousal, coverage)
        coverage = percentage of words found in lexicon
        """
        tokens = self.clean_lyrics(lyrics)
        
        if not tokens:
            return None, None, 0.0
        
        valences = []
        arousals = []
        
        for word in tokens:
            if word in self.vad_lexicon:
                valences.append(self.vad_lexicon[word]['valence'])
                arousals.append(self.vad_lexicon[word]['arousal'])
        
        if not valences:
            return None, None, 0.0
        
        coverage = len(valences) / len(tokens)
        
        return np.mean(valences), np.mean(arousals), coverage
    
    def assign_emotion(self, valence, arousal):
        """
        Assign emotion category based on Valence-Arousal quadrant
        Q1 (Happy): High V, High A
        Q2 (Angry): Low V, High A
        Q3 (Sad): Low V, Low A
        Q4 (Relaxed): High V, Low A
        """
        if valence is None or arousal is None:
            return None
        
        high_valence = valence >= self.valence_threshold
        high_arousal = arousal >= self.arousal_threshold
        
        if high_valence and high_arousal:
            return 'happy'
        elif not high_valence and high_arousal:
            return 'angry'
        elif not high_valence and not high_arousal:
            return 'sad'
        else:  # high_valence and not high_arousal
            return 'relaxed'
    
    def annotate_dataset(self, input_csv, output_csv):
        """
        Annotate a dataset with emotion labels
        """
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        print(f"Annotating {len(df)} songs with VAD-based emotions...")
        
        # Calculate VAD scores
        vad_results = df['lyrics'].apply(
            lambda x: self.calculate_vad_scores(x)
        )
        
        df['valence'] = vad_results.apply(lambda x: x[0])
        df['arousal'] = vad_results.apply(lambda x: x[1])
        df['vad_coverage'] = vad_results.apply(lambda x: x[2])
        
        # Assign emotions
        df['emotion'] = df.apply(
            lambda row: self.assign_emotion(row['valence'], row['arousal']),
            axis=1
        )
        
        # Filter out songs with low coverage or missing emotions
        min_coverage = 0.1  # At least 10% of words in lexicon
        df_filtered = df[
            (df['vad_coverage'] >= min_coverage) & 
            (df['emotion'].notna())
        ].copy()
        
        print(f"\nAnnotation complete:")
        print(f"  Original songs: {len(df)}")
        print(f"  After filtering (coverage >= {min_coverage}): {len(df_filtered)}")
        print(f"  Removed: {len(df) - len(df_filtered)}")
        
        print(f"\nEmotion distribution:")
        print(df_filtered['emotion'].value_counts())
        
        print(f"\nAverage VAD coverage: {df_filtered['vad_coverage'].mean():.2%}")
        
        # Save annotated data
        df_filtered.to_csv(output_csv, index=False)
        print(f"\nSaved annotated data to: {output_csv}")
        
        return df_filtered
    
    def balance_emotions(self, df, target_per_emotion=None):
        """
        Balance dataset to have equal samples per emotion
        """
        emotion_counts = df['emotion'].value_counts()
        print("\nOriginal emotion distribution:")
        print(emotion_counts)
        
        if target_per_emotion is None:
            target_per_emotion = emotion_counts.min()
        
        print(f"\nBalancing to {target_per_emotion} samples per emotion...")
        
        balanced_dfs = []
        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion]
            sampled = emotion_df.sample(
                n=min(target_per_emotion, len(emotion_df)),
                random_state=42
            )
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print("\nBalanced emotion distribution:")
        print(balanced_df['emotion'].value_counts())
        
        return balanced_df


if __name__ == "__main__":
    annotator = ValenceArousalAnnotator()
    
    # Example: Annotate collected data
    # annotator.annotate_dataset(
    #     'data/raw/genius_pulls/pop_raw.csv',
    #     'data/processed/pop_annotated.csv'
    # )