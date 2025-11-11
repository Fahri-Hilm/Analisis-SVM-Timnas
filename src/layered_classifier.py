"""
Layered Analytics Classifiers
Multi-layer classification untuk analisis sentimen yang lebih mendalam
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    multilabel_confusion_matrix, hamming_loss,
    jaccard_score
)
import joblib
from typing import List, Dict, Tuple, Optional
import yaml


class EmotionClassifier:
    """
    Multi-label emotion classification
    Labels: marah, kecewa, sedih, senang, bangga, takut
    """
    
    def __init__(self, emotion_labels: List[str], 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 threshold: float = 0.5):
        self.emotion_labels = emotion_labels
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.threshold = threshold
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.mlb = MultiLabelBinarizer(classes=emotion_labels)
        self.classifier = OneVsRestClassifier(
            LinearSVC(class_weight='balanced', random_state=42)
        )
        
    def train(self, texts: List[str], emotion_labels: List[List[str]]):
        """
        Train emotion classifier
        
        Args:
            texts: List of text comments
            emotion_labels: List of lists of emotion labels for each text
        """
        # Transform texts to TF-IDF
        X = self.vectorizer.fit_transform(texts)
        
        # Transform multi-labels
        y = self.mlb.fit_transform(emotion_labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        
        return self
    
    def predict(self, texts: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        """
        Predict emotions for texts
        
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        X = self.vectorizer.transform(texts)
        
        # Get decision scores
        decision_scores = self.classifier.decision_function(X)
        
        # Apply threshold
        predictions_binary = (decision_scores > self.threshold).astype(int)
        
        # Transform back to labels
        predicted_labels = self.mlb.inverse_transform(predictions_binary)
        predicted_labels = [list(labels) if labels else [] for labels in predicted_labels]
        
        # Get confidence (normalized decision scores)
        confidence_scores = 1 / (1 + np.exp(-decision_scores))  # sigmoid
        
        return predicted_labels, confidence_scores
    
    def evaluate(self, texts: List[str], true_labels: List[List[str]]) -> Dict:
        """Evaluate emotion classifier"""
        predicted_labels, confidence_scores = self.predict(texts)
        
        X = self.vectorizer.transform(texts)
        y_true = self.mlb.transform(true_labels)
        y_pred = self.mlb.transform(predicted_labels)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        metrics = {
            'macro_precision': float(precision),
            'macro_recall': float(recall),
            'macro_f1': float(f1),
            'hamming_loss': float(hamming_loss(y_true, y_pred)),
            'jaccard_score': float(jaccard_score(y_true, y_pred, average='samples')),
            'per_label_report': classification_report(
                y_true, y_pred, target_names=self.emotion_labels, output_dict=True
            )
        }
        
        return metrics
    
    def save(self, model_path: str, vectorizer_path: str, mlb_path: str):
        """Save model artifacts"""
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.mlb, mlb_path)
    
    def load(self, model_path: str, vectorizer_path: str, mlb_path: str):
        """Load model artifacts"""
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.mlb = joblib.load(mlb_path)
        return self


class AspectClassifier:
    """
    Multi-label aspect classification
    Labels: manajemen, pelatih, pemain, strategi, wasit, PSSI, federasi, fanbase
    """
    
    def __init__(self, aspect_labels: List[str],
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 threshold: float = 0.5):
        self.aspect_labels = aspect_labels
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.threshold = threshold
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.mlb = MultiLabelBinarizer(classes=aspect_labels)
        self.classifier = OneVsRestClassifier(
            LinearSVC(class_weight='balanced', random_state=42)
        )
    
    def train(self, texts: List[str], aspect_labels: List[List[str]]):
        """Train aspect classifier"""
        X = self.vectorizer.fit_transform(texts)
        y = self.mlb.fit_transform(aspect_labels)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, texts: List[str]) -> Tuple[List[List[str]], np.ndarray]:
        """Predict aspects for texts"""
        X = self.vectorizer.transform(texts)
        decision_scores = self.classifier.decision_function(X)
        predictions_binary = (decision_scores > self.threshold).astype(int)
        predicted_labels = self.mlb.inverse_transform(predictions_binary)
        predicted_labels = [list(labels) if labels else [] for labels in predicted_labels]
        confidence_scores = 1 / (1 + np.exp(-decision_scores))
        return predicted_labels, confidence_scores
    
    def evaluate(self, texts: List[str], true_labels: List[List[str]]) -> Dict:
        """Evaluate aspect classifier"""
        predicted_labels, _ = self.predict(texts)
        y_true = self.mlb.transform(true_labels)
        y_pred = self.mlb.transform(predicted_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        return {
            'macro_precision': float(precision),
            'macro_recall': float(recall),
            'macro_f1': float(f1),
            'hamming_loss': float(hamming_loss(y_true, y_pred)),
            'jaccard_score': float(jaccard_score(y_true, y_pred, average='samples')),
            'per_label_report': classification_report(
                y_true, y_pred, target_names=self.aspect_labels, output_dict=True
            )
        }
    
    def save(self, model_path: str, vectorizer_path: str, mlb_path: str):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.mlb, mlb_path)
    
    def load(self, model_path: str, vectorizer_path: str, mlb_path: str):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.mlb = joblib.load(mlb_path)
        return self


class ToxicityClassifier:
    """
    Binary toxicity classification
    Labels: toxic, non-toxic
    """
    
    def __init__(self, max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    
    def train(self, texts: List[str], labels: List[str]):
        """Train toxicity classifier"""
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Predict toxicity for texts
        
        Returns:
            Tuple of (predicted_labels, toxicity_scores)
        """
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        # Get probability of toxic class
        toxic_idx = list(self.label_encoder.classes_).index('toxic') if 'toxic' in self.label_encoder.classes_ else 1
        toxicity_scores = probabilities[:, toxic_idx]
        
        return list(predicted_labels), toxicity_scores
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate toxicity classifier"""
        predicted_labels, _ = self.predict(texts)
        
        y_true = self.label_encoder.transform(true_labels)
        y_pred = self.label_encoder.transform(predicted_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def save(self, model_path: str, vectorizer_path: str, encoder_path: str):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
    
    def load(self, model_path: str, vectorizer_path: str, encoder_path: str):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        return self


class StanceClassifier:
    """
    Stance classification
    Labels: pro, kontra, tidak_jelas
    """
    
    def __init__(self, stance_labels: List[str],
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.stance_labels = stance_labels
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    
    def train(self, texts: List[str], labels: List[str]):
        """Train stance classifier"""
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """Predict stance for texts"""
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1)
        
        return list(predicted_labels), confidence_scores
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate stance classifier"""
        predicted_labels, _ = self.predict(texts)
        
        y_true = self.label_encoder.transform(true_labels)
        y_pred = self.label_encoder.transform(predicted_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def save(self, model_path: str, vectorizer_path: str, encoder_path: str):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
    
    def load(self, model_path: str, vectorizer_path: str, encoder_path: str):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        return self


class IntentClassifier:
    """
    Intent classification
    Labels: pertanyaan, komplain, saran, ajakan, humor, informasi
    """
    
    def __init__(self, intent_labels: List[str],
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.intent_labels = intent_labels
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    
    def train(self, texts: List[str], labels: List[str]):
        """Train intent classifier"""
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        """Predict intent for texts"""
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        confidence_scores = np.max(probabilities, axis=1)
        
        return list(predicted_labels), confidence_scores
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate intent classifier"""
        predicted_labels, _ = self.predict(texts)
        
        y_true = self.label_encoder.transform(true_labels)
        y_pred = self.label_encoder.transform(predicted_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def save(self, model_path: str, vectorizer_path: str, encoder_path: str):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
    
    def load(self, model_path: str, vectorizer_path: str, encoder_path: str):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        return self
