"""
Logistic Regression baseline model for emotion classification.

This module implements Experiment 1: training interpretable BOW-based
classifiers on each genre and evaluating cross-genre transfer performance.

Research question: What words/phrases signal each emotion in each genre?
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
import joblib


class BOWEmotionClassifier:
    """
    Bag-of-words logistic regression classifier for emotion detection.
    
    Designed for interpretability: extracts top predictive features
    per emotion class for linguistic analysis.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams to extract (1,2) = unigrams + bigrams
            min_df: Minimum document frequency for terms
            max_iter: Maximum iterations for logistic regression
            class_weight: 'balanced' to handle class imbalance
            random_state: Random seed for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.vectorizer = None
        self.classifier = None
        self.classes_ = None
        self.is_fitted = False
    
    def fit(self, texts: List[str], labels: List[str]) -> 'BOWEmotionClassifier':
        """
        Fit the vectorizer and classifier on training data.
        
        Args:
            texts: List of lyrics strings
            labels: List of emotion labels
        
        Returns:
            self
        """
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words='english'
        )
        
        # Fit and transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Initialize and fit classifier
        self.classifier = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        self.classifier.fit(X, labels)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict emotion labels for new texts.
        
        Args:
            texts: List of lyrics strings
        
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for new texts.
        
        Args:
            texts: List of lyrics strings
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def get_top_features(
        self, 
        n_features: int = 30
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top predictive features for each emotion class.
        
        Uses logistic regression coefficients to identify words
        most strongly associated with each emotion.
        
        Args:
            n_features: Number of top features to extract per class
        
        Returns:
            Dictionary mapping emotion labels to list of (word, coefficient) tuples
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = {}
        
        for idx, emotion in enumerate(self.classes_):
            # Get coefficients for this class
            coefficients = self.classifier.coef_[idx]
            
            # Get indices of top positive coefficients
            top_indices = np.argsort(coefficients)[-n_features:][::-1]
            
            # Extract feature names and coefficients
            top_features[emotion] = [
                (feature_names[i], coefficients[i]) 
                for i in top_indices
            ]
        
        return top_features
    
    def get_vocabulary_size(self) -> int:
        """Return the actual vocabulary size after fitting."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
    
    def save(self, path: Path) -> None:
        """Save the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classes_': self.classes_,
            'config': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_iter': self.max_iter,
                'class_weight': self.class_weight,
                'random_state': self.random_state
            }
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'BOWEmotionClassifier':
        """Load a fitted model from disk."""
        data = joblib.load(path)
        
        model = cls(**data['config'])
        model.vectorizer = data['vectorizer']
        model.classifier = data['classifier']
        model.classes_ = data['classes_']
        model.is_fitted = True
        
        return model


def evaluate_classifier(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names (for consistent ordering)
    
    Returns:
        Dictionary with macro F1, per-emotion F1, precision, recall,
        and confusion matrix
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    results = {
        'macro_f1': f1_score(y_true, y_pred, average='macro', labels=labels),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', labels=labels),
        'macro_precision': precision_score(y_true, y_pred, average='macro', labels=labels),
        'macro_recall': recall_score(y_true, y_pred, average='macro', labels=labels),
        'per_emotion_f1': {},
        'per_emotion_precision': {},
        'per_emotion_recall': {},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
        'support': {}
    }
    
    # Per-emotion metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels)
    
    for idx, label in enumerate(labels):
        results['per_emotion_f1'][label] = f1_per_class[idx]
        results['per_emotion_precision'][label] = precision_per_class[idx]
        results['per_emotion_recall'][label] = recall_per_class[idx]
        results['support'][label] = sum(1 for y in y_true if y == label)
    
    return results


def format_results_table(results: Dict, title: str = "") -> str:
    """Format evaluation results as a readable table."""
    lines = []
    
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
    
    lines.append(f"Macro F1:     {results['macro_f1']:.4f}")
    lines.append(f"Weighted F1:  {results['weighted_f1']:.4f}")
    lines.append(f"Precision:    {results['macro_precision']:.4f}")
    lines.append(f"Recall:       {results['macro_recall']:.4f}")
    
    lines.append("\nPer-emotion F1 scores:")
    for emotion in results['labels']:
        f1 = results['per_emotion_f1'][emotion]
        support = results['support'][emotion]
        lines.append(f"  {emotion:10s}: {f1:.4f} (n={support})")
    
    return "\n".join(lines)


