"""
Loss Functions for Multi-Task Learning

Implements combined loss for simultaneous emotion classification
and valence-arousal regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task emotion recognition
    
    L_total = alpha * L_classification + beta * L_regression
    
    where:
    - L_classification: Cross Entropy Loss for emotion categories
    - L_regression: Mean Squared Error for valence-arousal
    """
    
    def __init__(self, alpha=1.0, beta=0.5, class_weights=None):
        """
        Initialize multi-task loss
        
        Args:
            alpha (float): Weight for classification loss (default: 1.0)
            beta (float): Weight for regression loss (default: 0.5)
            class_weights (torch.Tensor, optional): Class weights for imbalanced data
        """
        super(MultiTaskLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        # Classification loss (Cross Entropy)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Regression loss (Mean Squared Error)
        self.mse_loss = nn.MSELoss()
        
        print(f"MultiTaskLoss initialized: alpha={alpha}, beta={beta}")
    
    def forward(self, emotion_logits, va_pred, emotion_target, va_target):
        """
        Compute combined loss
        
        Args:
            emotion_logits (torch.Tensor): Predicted emotion logits, shape (batch_size, num_classes)
            va_pred (torch.Tensor): Predicted valence-arousal, shape (batch_size, 2)
            emotion_target (torch.Tensor): True emotion labels, shape (batch_size,)
            va_target (torch.Tensor): True valence-arousal, shape (batch_size, 2)
            
        Returns:
            tuple: (total_loss, emotion_loss, va_loss)
        """
        # Classification loss
        loss_emotion = self.ce_loss(emotion_logits, emotion_target)
        
        # Regression loss
        loss_va = self.mse_loss(va_pred, va_target)
        
        # Combined weighted loss
        total_loss = self.alpha * loss_emotion + self.beta * loss_va
        
        return total_loss, loss_emotion, loss_va


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Focuses training on hard examples
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha (float): Weighting factor (default: 1.0)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (torch.Tensor): Predicted logits
            targets (torch.Tensor): True labels
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskFocalLoss(nn.Module):
    """
    Multi-task loss with Focal Loss for classification
    
    Better for imbalanced emotion datasets
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0):
        """
        Initialize
        
        Args:
            alpha (float): Weight for classification
            beta (float): Weight for regression
            gamma (float): Focal loss focusing parameter
        """
        super(MultiTaskFocalLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        self.focal_loss = FocalLoss(gamma=gamma)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, emotion_logits, va_pred, emotion_target, va_target):
        """Compute combined loss with focal loss"""
        loss_emotion = self.focal_loss(emotion_logits, emotion_target)
        loss_va = self.mse_loss(va_pred, va_target)
        
        total_loss = self.alpha * loss_emotion + self.beta * loss_va
        
        return total_loss, loss_emotion, loss_va


class HuberLoss(nn.Module):
    """
    Huber Loss for robust regression
    
    More robust to outliers than MSE
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber Loss
        
        Args:
            delta (float): Threshold for switching between L1 and L2
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """
        Compute Huber loss
        
        Args:
            pred (torch.Tensor): Predictions
            target (torch.Tensor): Ground truth
            
        Returns:
            torch.Tensor: Huber loss
        """
        error = pred - target
        abs_error = torch.abs(error)
        
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(error.device))
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class MultiTaskHuberLoss(nn.Module):
    """
    Multi-task loss with Huber Loss for regression
    
    More robust to outliers in valence-arousal annotations
    """
    
    def __init__(self, alpha=1.0, beta=0.5, delta=1.0):
        """
        Initialize
        
        Args:
            alpha (float): Classification weight
            beta (float): Regression weight
            delta (float): Huber loss threshold
        """
        super(MultiTaskHuberLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = HuberLoss(delta=delta)
    
    def forward(self, emotion_logits, va_pred, emotion_target, va_target):
        """Compute combined loss with Huber loss"""
        loss_emotion = self.ce_loss(emotion_logits, emotion_target)
        loss_va = self.huber_loss(va_pred, va_target)
        
        total_loss = self.alpha * loss_emotion + self.beta * loss_va
        
        return total_loss, loss_emotion, loss_va


def create_loss_function(loss_type='multitask', alpha=1.0, beta=0.5, **kwargs):
    """
    Factory function to create loss function
    
    Args:
        loss_type (str): Type of loss ('multitask', 'focal', 'huber')
        alpha (float): Classification weight
        beta (float): Regression weight
        **kwargs: Additional loss-specific parameters
        
    Returns:
        nn.Module: Loss function
    """
    if loss_type == 'multitask':
        return MultiTaskLoss(alpha=alpha, beta=beta, **kwargs)
    elif loss_type == 'focal':
        return MultiTaskFocalLoss(alpha=alpha, beta=beta, **kwargs)
    elif loss_type == 'huber':
        return MultiTaskHuberLoss(alpha=alpha, beta=beta, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(emotion_counts, device='cpu'):
    """
    Compute class weights for imbalanced dataset
    
    Args:
        emotion_counts (dict or list): Emotion class counts
        device (str): Device to place tensor on
        
    Returns:
        torch.Tensor: Class weights
    """
    if isinstance(emotion_counts, dict):
        counts = torch.tensor(list(emotion_counts.values()), dtype=torch.float32)
    else:
        counts = torch.tensor(emotion_counts, dtype=torch.float32)
    
    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (len(counts) * counts)
    
    # Normalize weights
    weights = weights / weights.sum() * len(counts)
    
    return weights.to(device)
