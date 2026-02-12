"""
Evaluation Metrics for Multi-Task Emotion Recognition

Provides comprehensive evaluation for:
1. Categorical emotion classification
2. Valence-Arousal regression
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_ccc(y_true, y_pred):
    """
    Calculate Concordance Correlation Coefficient (CCC)
    
    CCC measures the agreement between two variables
    Range: [-1, 1], where 1 = perfect agreement
    
    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values
        
    Returns:
        float: CCC value
    """
    # Ensure numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Calculate covariance
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 0.0
    
    ccc = (2 * covariance) / denominator
    
    return ccc


def evaluate_classification(y_true, y_pred, emotion_labels=None):
    """
    Evaluate classification performance
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        emotion_labels (list, optional): Emotion class names
        
    Returns:
        dict: Classification metrics
    """
    if emotion_labels is None:
        emotion_labels = [
            'Neutral', 'Happy', 'Sad', 'Surprise',
            'Fear', 'Disgust', 'Anger', 'Contempt'
        ]
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_f1': dict(zip(emotion_labels[:len(per_class_f1)], per_class_f1)),
        'per_class_precision': dict(zip(emotion_labels[:len(per_class_precision)], per_class_precision)),
        'per_class_recall': dict(zip(emotion_labels[:len(per_class_recall)], per_class_recall)),
        'confusion_matrix': cm
    }
    
    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression performance for valence-arousal
    
    Args:
        y_true (np.array): True values, shape (N, 2)
        y_pred (np.array): Predicted values, shape (N, 2)
        
    Returns:
        dict: Regression metrics
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(overall_mse)
    
    # Separate valence and arousal
    valence_true = y_true[:, 0]
    valence_pred = y_pred[:, 0]
    arousal_true = y_true[:, 1]
    arousal_pred = y_pred[:, 1]
    
    # Valence metrics
    valence_mse = mean_squared_error(valence_true, valence_pred)
    valence_mae = mean_absolute_error(valence_true, valence_pred)
    valence_rmse = np.sqrt(valence_mse)
    valence_ccc = calculate_ccc(valence_true, valence_pred)
    valence_pearson, valence_p_value = pearsonr(valence_true, valence_pred)
    
    # Arousal metrics
    arousal_mse = mean_squared_error(arousal_true, arousal_pred)
    arousal_mae = mean_absolute_error(arousal_true, arousal_pred)
    arousal_rmse = np.sqrt(arousal_mse)
    arousal_ccc = calculate_ccc(arousal_true, arousal_pred)
    arousal_pearson, arousal_p_value = pearsonr(arousal_true, arousal_pred)
    
    # Mean CCC (primary metric)
    mean_ccc = (valence_ccc + arousal_ccc) / 2
    
    metrics = {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'valence_mse': valence_mse,
        'valence_mae': valence_mae,
        'valence_rmse': valence_rmse,
        'valence_ccc': valence_ccc,
        'valence_pearson': valence_pearson,
        'arousal_mse': arousal_mse,
        'arousal_mae': arousal_mae,
        'arousal_rmse': arousal_rmse,
        'arousal_ccc': arousal_ccc,
        'arousal_pearson': arousal_pearson,
        'mean_ccc': mean_ccc
    }
    
    return metrics


def evaluate_model(model, dataloader, criterion, device, emotion_labels=None):
    """
    Comprehensive model evaluation
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Evaluation data
        criterion: Loss function
        device (torch.device): Device to run on
        emotion_labels (list, optional): Emotion class names
        
    Returns:
        dict: Complete evaluation metrics
    """
    model.eval()
    
    # Storage for predictions and targets
    all_emotion_preds = []
    all_emotion_targets = []
    all_va_preds = []
    all_va_targets = []
    
    total_loss = 0.0
    total_emotion_loss = 0.0
    total_va_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, emotions, va_targets) in enumerate(dataloader):
            images = images.to(device)
            emotions = emotions.to(device)
            va_targets = va_targets.to(device)
            
            # Forward pass
            emotion_logits, va_preds = model(images)
            
            # Calculate loss
            loss, emotion_loss, va_loss = criterion(emotion_logits, va_preds, emotions, va_targets)
            
            total_loss += loss.item()
            total_emotion_loss += emotion_loss.item()
            total_va_loss += va_loss.item()
            
            # Get predictions
            emotion_preds = torch.argmax(emotion_logits, dim=1)
            
            # Store predictions and targets
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_targets.extend(emotions.cpu().numpy())
            all_va_preds.extend(va_preds.cpu().numpy())
            all_va_targets.extend(va_targets.cpu().numpy())
    
    # Convert to numpy arrays
    all_emotion_preds = np.array(all_emotion_preds)
    all_emotion_targets = np.array(all_emotion_targets)
    all_va_preds = np.array(all_va_preds)
    all_va_targets = np.array(all_va_targets)
    
    # Calculate metrics
    classification_metrics = evaluate_classification(
        all_emotion_targets, all_emotion_preds, emotion_labels
    )
    regression_metrics = evaluate_regression(all_va_targets, all_va_preds)
    
    # Combine all metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'emotion_loss': total_emotion_loss / len(dataloader),
        'va_loss': total_va_loss / len(dataloader),
        **classification_metrics,
        **regression_metrics
    }
    
    return metrics


def print_evaluation_summary(metrics, dataset_name='Validation'):
    """
    Print formatted evaluation summary
    
    Args:
        metrics (dict): Evaluation metrics
        dataset_name (str): Name of dataset (e.g., 'Validation', 'Test')
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} SET RESULTS")
    print(f"{'='*80}")
    
    print(f"\nLoss Metrics:")
    print(f"  Total Loss:        {metrics['loss']:.4f}")
    print(f"  Emotion Loss:      {metrics['emotion_loss']:.4f}")
    print(f"  VA Loss:           {metrics['va_loss']:.4f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    
    print(f"\nRegression Metrics:")
    print(f"  Overall MSE:       {metrics['overall_mse']:.4f}")
    print(f"  Overall MAE:       {metrics['overall_mae']:.4f}")
    print(f"  Overall RMSE:      {metrics['overall_rmse']:.4f}")
    
    print(f"\n  Valence:")
    print(f"    MSE:             {metrics['valence_mse']:.4f}")
    print(f"    MAE:             {metrics['valence_mae']:.4f}")
    print(f"    RMSE:            {metrics['valence_rmse']:.4f}")
    print(f"    CCC:             {metrics['valence_ccc']:.4f}")
    print(f"    Pearson:         {metrics['valence_pearson']:.4f}")
    
    print(f"\n  Arousal:")
    print(f"    MSE:             {metrics['arousal_mse']:.4f}")
    print(f"    MAE:             {metrics['arousal_mae']:.4f}")
    print(f"    RMSE:            {metrics['arousal_rmse']:.4f}")
    print(f"    CCC:             {metrics['arousal_ccc']:.4f}")
    print(f"    Pearson:         {metrics['arousal_pearson']:.4f}")
    
    print(f"\n  Mean CCC:          {metrics['mean_ccc']:.4f}")
    print(f"{'='*80}\n")


def plot_confusion_matrix(cm, emotion_labels, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm (np.array): Confusion matrix
        emotion_labels (list): Class labels
        save_path (str, optional): Path to save figure
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_va_scatter(y_true, y_pred, save_path=None):
    """
    Plot valence-arousal scatter plots
    
    Args:
        y_true (np.array): True VA values, shape (N, 2)
        y_pred (np.array): Predicted VA values, shape (N, 2)
        save_path (str, optional): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Valence scatter
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=10)
    axes[0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Valence', fontsize=12)
    axes[0].set_ylabel('Predicted Valence', fontsize=12)
    axes[0].set_title('Valence Predictions', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    
    # Arousal scatter
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=10, color='orange')
    axes[1].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Arousal', fontsize=12)
    axes[1].set_ylabel('Predicted Arousal', fontsize=12)
    axes[1].set_title('Arousal Predictions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"VA scatter plots saved to {save_path}")
    
    plt.show()
