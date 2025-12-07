"""
DistilBERT-based emotion classifier for cross-genre transfer experiments.

This module implements Experiment 2: training neural classifiers on each genre
and evaluating whether contextual embeddings capture cross-genre patterns
better than BOW features.

Research question: Do neural models generalize better across genres than simple models?
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from tqdm import tqdm


class LyricsDataset(Dataset):
    """
    PyTorch Dataset for lyrics emotion classification.
    
    Handles tokenization and label encoding for DistilBERT input.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: DistilBertTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of lyrics strings
            labels: List of emotion label strings
            tokenizer: DistilBERT tokenizer
            label2id: Mapping from label string to integer ID
            max_length: Maximum sequence length (DistilBERT max is 512)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label2id[label], dtype=torch.long)
        }


class DistilBERTEmotionClassifier:
    """
    DistilBERT-based classifier for emotion detection in lyrics.
    
    Wraps the HuggingFace transformers implementation with training,
    evaluation, and prediction methods designed for cross-genre experiments.
    """
    
    def __init__(
        self,
        num_labels: int = 4,
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            num_labels: Number of emotion classes
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.num_labels = num_labels
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Label mappings
        self.label2id = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.is_fitted = False
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def _init_model(self) -> None:
        """Initialize a fresh model for training."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
    
    def _compute_class_weights(self, labels: List[str]) -> torch.Tensor:
        """
        Compute inverse frequency class weights for balanced training.
        
        Args:
            labels: List of training label strings
        
        Returns:
            Tensor of class weights
        """
        label_counts = pd.Series(labels).value_counts()
        total = len(labels)
        
        weights = []
        for i in range(self.num_labels):
            label = self.id2label[i]
            count = label_counts.get(label, 1)
            weights.append(total / (self.num_labels * count))
        
        return torch.tensor(weights, dtype=torch.float).to(self.device)
    
    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        use_class_weights: bool = True,
        verbose: bool = True
    ) -> 'DistilBERTEmotionClassifier':
        """
        Fine-tune DistilBERT on training data.
        
        Args:
            train_texts: Training lyrics
            train_labels: Training emotion labels
            val_texts: Validation lyrics (optional)
            val_labels: Validation emotion labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for AdamW
            warmup_steps: Linear warmup steps
            weight_decay: Weight decay for regularization
            use_class_weights: Whether to use weighted loss
            verbose: Print progress
        
        Returns:
            self
        """
        # Initialize fresh model
        self._init_model()
        
        # Create datasets
        train_dataset = LyricsDataset(
            train_texts, train_labels, self.tokenizer, 
            self.label2id, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_texts is not None and val_labels is not None:
            val_dataset = LyricsDataset(
                val_texts, val_labels, self.tokenizer,
                self.label2id, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        # Compute class weights
        if use_class_weights:
            class_weights = self._compute_class_weights(train_labels)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            else:
                pbar = train_loader
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if verbose:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_f1 = self._evaluate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_f1)
                
                if verbose:
                    print(f'  Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
            else:
                if verbose:
                    print(f'  Train Loss: {avg_train_loss:.4f}')
        
        self.is_fitted = True
        return self
    
    def _evaluate_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        self.model.train()
        
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Predict emotion labels for new texts.
        
        Args:
            texts: List of lyrics strings
            batch_size: Batch size for inference
        
        Returns:
            Array of predicted label strings
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = ['angry'] * len(texts)
        dataset = LyricsDataset(
            texts, dummy_labels, self.tokenizer,
            self.label2id, self.max_length
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        # Convert to label strings
        return np.array([self.id2label[p] for p in all_preds])
    
    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Predict class probabilities for new texts.
        
        Args:
            texts: List of lyrics strings
            batch_size: Batch size for inference
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        self.model.eval()
        
        dummy_labels = ['angry'] * len(texts)
        dataset = LyricsDataset(
            texts, dummy_labels, self.tokenizer,
            self.label2id, self.max_length
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def save(self, path: Path) -> None:
        """Save the fitted model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save metadata
        metadata = {
            'num_labels': self.num_labels,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'history': self.history
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'DistilBERTEmotionClassifier':
        """Load a fitted model from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Initialize classifier
        classifier = cls(
            num_labels=metadata['num_labels'],
            model_name=metadata['model_name'],
            max_length=metadata['max_length'],
            device=device
        )
        
        # Load model and tokenizer
        classifier.model = DistilBertForSequenceClassification.from_pretrained(path)
        classifier.model.to(classifier.device)
        classifier.tokenizer = DistilBertTokenizer.from_pretrained(path)
        
        classifier.label2id = metadata['label2id']
        classifier.id2label = {int(k): v for k, v in metadata['id2label'].items()}
        classifier.history = metadata['history']
        classifier.is_fitted = True
        
        return classifier


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
        labels = ['angry', 'happy', 'relaxed', 'sad']
    
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
        results['per_emotion_f1'][label] = float(f1_per_class[idx])
        results['per_emotion_precision'][label] = float(precision_per_class[idx])
        results['per_emotion_recall'][label] = float(recall_per_class[idx])
        results['support'][label] = sum(1 for y in y_true if y == label)
    
    return results