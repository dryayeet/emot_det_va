"""
Utility Functions for Training and Deployment

Provides helper functions for:
- Data preprocessing
- Model saving/loading
- Visualization
- Configuration management
"""

import os
import json
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Get training data augmentation transforms
    
    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Get validation/test transforms (no augmentation)
    
    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms():
    """
    Get inference transforms for deployment
    
    Returns:
        torchvision.transforms.Compose: Inference transforms
    """
    return get_val_transforms()


def save_checkpoint(model, optimizer, epoch, metrics, filepath, config=None):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        metrics (dict): Performance metrics
        filepath (str): Path to save checkpoint
        config (dict, optional): Training configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if config:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (optional): Optimizer to load state into
        device (str): Device to load model on
        
    Returns:
        dict: Checkpoint data including metrics and config
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    return checkpoint


def save_model_weights(model, filepath):
    """
    Save only model weights
    
    Args:
        model (nn.Module): Model
        filepath (str): Save path
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath, device='cpu'):
    """
    Load model weights
    
    Args:
        model (nn.Module): Model
        filepath (str): Weight file path
        device (str): Device
    """
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {filepath}")


def export_to_torchscript(model, save_path, device='cpu'):
    """
    Export model to TorchScript format
    
    Args:
        model (nn.Module): Model to export
        save_path (str): Path to save TorchScript model
        device (str): Device
    """
    model.eval()
    model = model.to(device)
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")
    
    # Verify
    loaded = torch.jit.load(save_path)
    loaded.eval()
    print("TorchScript model verified successfully")


def export_to_onnx(model, save_path, device='cpu'):
    """
    Export model to ONNX format
    
    Args:
        model (nn.Module): Model to export
        save_path (str): Path to save ONNX model
        device (str): Device
    """
    import torch.onnx
    
    model.eval()
    model = model.to(device)
    
    # Example input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['emotion_logits', 'va_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'emotion_logits': {0: 'batch_size'},
            'va_output': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to {save_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def save_config(config, filepath):
    """
    Save configuration to JSON
    
    Args:
        config (dict): Configuration dictionary
        filepath (str): Save path
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filepath}")


def load_config(filepath):
    """
    Load configuration from JSON
    
    Args:
        filepath (str): Config file path
        
    Returns:
        dict: Configuration
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config


def plot_training_history(history, save_path=None):
    """
    Plot training history curves
    
    Args:
        history (dict): Training history
        save_path (str, optional): Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Macro F1
    if 'val_f1' in history:
        axes[1, 0].plot(history['val_f1'], label='Val Macro F1', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Macro F1', fontsize=12)
        axes[1, 0].set_title('Validation Macro F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Mean CCC
    if 'val_ccc' in history:
        axes[1, 1].plot(history['val_ccc'], label='Val Mean CCC', linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Mean CCC', fontsize=12)
        axes[1, 1].set_title('Validation Mean CCC', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (list): Normalization mean
        std (list): Normalization std
        
    Returns:
        np.array: Denormalized image
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    return img


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if should stop training
        
        Args:
            score (float): Current metric value
            
        Returns:
            bool: True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def create_output_directory(base_dir='outputs'):
    """
    Create timestamped output directory
    
    Args:
        base_dir (str): Base directory
        
    Returns:
        str: Created directory path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model (nn.Module): Model
        
    Returns:
        dict: Parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
