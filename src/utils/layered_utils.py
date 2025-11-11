"""
Utilities untuk layered analytics evaluation dan reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json


def calculate_confidence_stats(confidence_scores: np.ndarray, 
                               threshold: float = 0.60) -> Dict:
    """
    Calculate confidence statistics
    
    Args:
        confidence_scores: Array of confidence scores
        threshold: Low confidence threshold
    
    Returns:
        Dictionary of confidence statistics
    """
    low_conf_count = np.sum(confidence_scores < threshold)
    low_conf_pct = (low_conf_count / len(confidence_scores)) * 100
    
    return {
        'mean_confidence': float(np.mean(confidence_scores)),
        'median_confidence': float(np.median(confidence_scores)),
        'std_confidence': float(np.std(confidence_scores)),
        'min_confidence': float(np.min(confidence_scores)),
        'max_confidence': float(np.max(confidence_scores)),
        'low_confidence_count': int(low_conf_count),
        'low_confidence_percentage': float(low_conf_pct),
        'threshold_used': float(threshold)
    }


def flag_low_confidence(confidence_scores: np.ndarray, 
                       threshold: float = 0.60) -> np.ndarray:
    """
    Flag predictions with low confidence
    
    Args:
        confidence_scores: Array of confidence scores
        threshold: Low confidence threshold
    
    Returns:
        Boolean array indicating low confidence
    """
    return confidence_scores < threshold


def multilabel_to_string(labels_list: List[List[str]]) -> List[str]:
    """
    Convert list of label lists to comma-separated strings
    
    Args:
        labels_list: List of lists of labels
    
    Returns:
        List of comma-separated label strings
    """
    return [','.join(sorted(labels)) if labels else '' for labels in labels_list]


def string_to_multilabel(labels_str: List[str]) -> List[List[str]]:
    """
    Convert comma-separated strings back to list of label lists
    
    Args:
        labels_str: List of comma-separated label strings
    
    Returns:
        List of lists of labels
    """
    return [s.split(',') if s else [] for s in labels_str]


def create_layer_summary(df: pd.DataFrame, 
                         layer_name: str,
                         label_column: str) -> Dict:
    """
    Create summary statistics for a layer
    
    Args:
        df: DataFrame with predictions
        layer_name: Name of the layer
        label_column: Column name containing labels
    
    Returns:
        Summary dictionary
    """
    summary = {
        'layer_name': layer_name,
        'total_predictions': len(df),
        'label_distribution': {}
    }
    
    # For multilabel columns
    if df[label_column].dtype == 'object' and ',' in str(df[label_column].iloc[0]):
        all_labels = []
        for labels_str in df[label_column]:
            if pd.notna(labels_str) and labels_str:
                all_labels.extend(labels_str.split(','))
        
        from collections import Counter
        label_counts = Counter(all_labels)
        summary['label_distribution'] = dict(label_counts)
        summary['avg_labels_per_instance'] = len(all_labels) / len(df) if len(df) > 0 else 0
    else:
        # For single-label columns
        summary['label_distribution'] = df[label_column].value_counts().to_dict()
    
    return summary


def plot_label_distribution(label_counts: Dict[str, int], 
                            title: str,
                            save_path: str = None):
    """
    Plot label distribution
    
    Args:
        label_counts: Dictionary of label counts
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Count')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_layered_output_dataframe(
    texts: List[str],
    sentiment_labels: List[str],
    sentiment_confidence: np.ndarray,
    emotion_labels: List[List[str]] = None,
    aspect_labels: List[List[str]] = None,
    toxicity_labels: List[str] = None,
    toxicity_scores: np.ndarray = None,
    stance_labels: List[str] = None,
    stance_confidence: np.ndarray = None,
    intent_labels: List[str] = None,
    intent_confidence: np.ndarray = None,
    low_confidence_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Create comprehensive output dataframe with all layer predictions
    
    Args:
        texts: Input texts
        sentiment_labels: Sentiment predictions
        sentiment_confidence: Sentiment confidence scores
        emotion_labels: Emotion predictions (optional)
        aspect_labels: Aspect predictions (optional)
        toxicity_labels: Toxicity predictions (optional)
        toxicity_scores: Toxicity scores (optional)
        stance_labels: Stance predictions (optional)
        stance_confidence: Stance confidence scores (optional)
        intent_labels: Intent predictions (optional)
        intent_confidence: Intent confidence scores (optional)
        low_confidence_threshold: Threshold for low confidence flag
    
    Returns:
        DataFrame with all predictions
    """
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiment_labels,
        'sentiment_confidence': sentiment_confidence,
        'low_confidence_flag': sentiment_confidence < low_confidence_threshold
    })
    
    if emotion_labels is not None:
        df['emotions'] = multilabel_to_string(emotion_labels)
    
    if aspect_labels is not None:
        df['aspects'] = multilabel_to_string(aspect_labels)
    
    if toxicity_labels is not None:
        df['toxicity_label'] = toxicity_labels
    
    if toxicity_scores is not None:
        df['toxicity_score'] = toxicity_scores
    
    if stance_labels is not None:
        df['stance'] = stance_labels
    
    if stance_confidence is not None:
        df['stance_confidence'] = stance_confidence
    
    if intent_labels is not None:
        df['intent'] = intent_labels
    
    if intent_confidence is not None:
        df['intent_confidence'] = intent_confidence
    
    return df


def save_metrics_json(metrics: Dict, filepath: str):
    """
    Save metrics dictionary to JSON file
    
    Args:
        metrics: Metrics dictionary
        filepath: Path to save JSON
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics_json(filepath: str) -> Dict:
    """
    Load metrics from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Metrics dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_comprehensive_report(
    sentiment_metrics: Dict,
    emotion_metrics: Dict = None,
    aspect_metrics: Dict = None,
    toxicity_metrics: Dict = None,
    stance_metrics: Dict = None,
    intent_metrics: Dict = None
) -> str:
    """
    Create comprehensive text report of all layer metrics
    
    Args:
        sentiment_metrics: Sentiment layer metrics
        emotion_metrics: Emotion layer metrics (optional)
        aspect_metrics: Aspect layer metrics (optional)
        toxicity_metrics: Toxicity layer metrics (optional)
        stance_metrics: Stance layer metrics (optional)
        intent_metrics: Intent layer metrics (optional)
    
    Returns:
        Formatted text report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LAYERED ANALYTICS COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Sentiment
    report_lines.append("1. SENTIMENT ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append(f"   Accuracy: {sentiment_metrics.get('accuracy', 'N/A'):.4f}")
    report_lines.append(f"   Precision: {sentiment_metrics.get('precision', 'N/A'):.4f}")
    report_lines.append(f"   Recall: {sentiment_metrics.get('recall', 'N/A'):.4f}")
    report_lines.append(f"   F1-Score: {sentiment_metrics.get('f1_score', 'N/A'):.4f}")
    report_lines.append("")
    
    # Emotion
    if emotion_metrics:
        report_lines.append("2. EMOTION ANALYSIS (Multi-label)")
        report_lines.append("-" * 40)
        report_lines.append(f"   Macro Precision: {emotion_metrics.get('macro_precision', 'N/A'):.4f}")
        report_lines.append(f"   Macro Recall: {emotion_metrics.get('macro_recall', 'N/A'):.4f}")
        report_lines.append(f"   Macro F1: {emotion_metrics.get('macro_f1', 'N/A'):.4f}")
        report_lines.append(f"   Hamming Loss: {emotion_metrics.get('hamming_loss', 'N/A'):.4f}")
        report_lines.append(f"   Jaccard Score: {emotion_metrics.get('jaccard_score', 'N/A'):.4f}")
        report_lines.append("")
    
    # Aspect
    if aspect_metrics:
        report_lines.append("3. ASPECT ANALYSIS (Multi-label)")
        report_lines.append("-" * 40)
        report_lines.append(f"   Macro Precision: {aspect_metrics.get('macro_precision', 'N/A'):.4f}")
        report_lines.append(f"   Macro Recall: {aspect_metrics.get('macro_recall', 'N/A'):.4f}")
        report_lines.append(f"   Macro F1: {aspect_metrics.get('macro_f1', 'N/A'):.4f}")
        report_lines.append(f"   Hamming Loss: {aspect_metrics.get('hamming_loss', 'N/A'):.4f}")
        report_lines.append(f"   Jaccard Score: {aspect_metrics.get('jaccard_score', 'N/A'):.4f}")
        report_lines.append("")
    
    # Toxicity
    if toxicity_metrics:
        report_lines.append("4. TOXICITY DETECTION")
        report_lines.append("-" * 40)
        report_lines.append(f"   Accuracy: {toxicity_metrics.get('accuracy', 'N/A'):.4f}")
        report_lines.append(f"   Precision: {toxicity_metrics.get('precision', 'N/A'):.4f}")
        report_lines.append(f"   Recall: {toxicity_metrics.get('recall', 'N/A'):.4f}")
        report_lines.append(f"   F1-Score: {toxicity_metrics.get('f1_score', 'N/A'):.4f}")
        report_lines.append("")
    
    # Stance
    if stance_metrics:
        report_lines.append("5. STANCE CLASSIFICATION")
        report_lines.append("-" * 40)
        report_lines.append(f"   Accuracy: {stance_metrics.get('accuracy', 'N/A'):.4f}")
        report_lines.append(f"   Precision: {stance_metrics.get('precision', 'N/A'):.4f}")
        report_lines.append(f"   Recall: {stance_metrics.get('recall', 'N/A'):.4f}")
        report_lines.append(f"   F1-Score: {stance_metrics.get('f1_score', 'N/A'):.4f}")
        report_lines.append("")
    
    # Intent
    if intent_metrics:
        report_lines.append("6. INTENT CLASSIFICATION")
        report_lines.append("-" * 40)
        report_lines.append(f"   Accuracy: {intent_metrics.get('accuracy', 'N/A'):.4f}")
        report_lines.append(f"   Precision: {intent_metrics.get('precision', 'N/A'):.4f}")
        report_lines.append(f"   Recall: {intent_metrics.get('recall', 'N/A'):.4f}")
        report_lines.append(f"   F1-Score: {intent_metrics.get('f1_score', 'N/A'):.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    return '\n'.join(report_lines)


def export_layer_predictions(
    df: pd.DataFrame,
    output_path: str,
    include_text: bool = True
):
    """
    Export layered predictions to CSV
    
    Args:
        df: DataFrame with predictions
        output_path: Path to save CSV
        include_text: Whether to include original text (set False if text is long/sensitive)
    """
    if not include_text and 'text' in df.columns:
        df = df.drop(columns=['text'])
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Layered predictions exported to: {output_path}")
