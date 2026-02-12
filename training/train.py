"""
Main Training Script for Multi-Task Emotion Recognition

Trains a deep learning model for simultaneous:
1. Categorical emotion classification
2. Valence-arousal regression

Usage:
    python train.py --data_dir /path/to/affectnet --epochs 50 --batch_size 64
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import project modules
from dataset import (
    load_affectnet_data, create_data_splits,
    AffectNetDataset, get_class_distribution
)
from model import create_model
from losses import MultiTaskLoss
from evaluate import evaluate_model, print_evaluation_summary, plot_confusion_matrix, plot_va_scatter
from utils import (
    get_train_transforms, get_val_transforms,
    save_checkpoint, save_model_weights, save_config,
    export_to_torchscript, export_to_onnx,
    plot_training_history, EarlyStopping,
    create_output_directory, set_seed, count_parameters
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Multi-Task Emotion Recognition Model'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to AffectNet dataset directory')
    parser.add_argument('--annotation_file', type=str, default=None,
                       help='Path to annotation CSV file (optional)')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    
    # Loss arguments
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for regression loss')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'step', 'cosine'],
                       help='Learning rate scheduler type')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch
    
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    
    running_loss = 0.0
    running_emotion_loss = 0.0
    running_va_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, emotions, va_targets) in enumerate(dataloader):
        images = images.to(device)
        emotions = emotions.to(device)
        va_targets = va_targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        emotion_logits, va_preds = model(images)
        
        # Calculate loss
        loss, emotion_loss, va_loss = criterion(emotion_logits, va_preds, emotions, va_targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        running_emotion_loss += emotion_loss.item()
        running_va_loss += va_loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(emotion_logits, dim=1)
        correct += (preds == emotions).sum().item()
        total += emotions.size(0)
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = correct / total
            print(f"Epoch [{epoch+1}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {batch_loss:.4f} "
                  f"Acc: {batch_acc:.4f}")
    
    epoch_time = time.time() - start_time
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_emotion_loss = running_emotion_loss / len(dataloader)
    epoch_va_loss = running_va_loss / len(dataloader)
    epoch_acc = correct / total
    
    metrics = {
        'loss': epoch_loss,
        'emotion_loss': epoch_emotion_loss,
        'va_loss': epoch_va_loss,
        'accuracy': epoch_acc,
        'time': epoch_time
    }
    
    return metrics


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Save configuration
    config = vars(args)
    config['output_dir'] = output_dir
    save_config(config, os.path.join(output_dir, 'config.json'))
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss Weights: α={args.alpha}, β={args.beta}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # ========== Load Dataset ==========
    print("Loading dataset...")
    df = load_affectnet_data(args.data_dir, args.annotation_file)
    
    # Print class distribution
    print("\nEmotion Distribution:")
    class_dist = get_class_distribution(df)
    print(class_dist)
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    # ========== Create Datasets and DataLoaders ==========
    print("\nCreating data loaders...")
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = AffectNetDataset(train_df, args.data_dir, transform=train_transform)
    val_dataset = AffectNetDataset(val_df, args.data_dir, transform=val_transform)
    test_dataset = AffectNetDataset(test_df, args.data_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ========== Create Model ==========
    print("\nInitializing model...")
    model = create_model(
        architecture=args.architecture,
        num_classes=8,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    
    # ========== Loss Function ==========
    criterion = MultiTaskLoss(alpha=args.alpha, beta=args.beta)
    
    # ========== Optimizer ==========
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # ========== Learning Rate Scheduler ==========
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    
    # ========== Early Stopping ==========
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # ========== Training Loop ==========
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_ccc': []
    }
    
    best_f1 = 0.0
    best_ccc = 0.0
    best_combined_score = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # ========== Training ==========
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # ========== Validation ==========
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_ccc'].append(val_metrics['mean_ccc'])
        
        # Learning rate scheduling
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['macro_f1'])
        else:
            scheduler.step()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"EPOCH [{epoch+1}/{args.epochs}] - Time: {epoch_time:.2f}s")
        print(f"{'='*80}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val CCC - Valence: {val_metrics['valence_ccc']:.4f} | "
              f"Arousal: {val_metrics['arousal_ccc']:.4f} | "
              f"Mean: {val_metrics['mean_ccc']:.4f}")
        print(f"{'='*80}\n")
        
        # Save best model
        combined_score = val_metrics['macro_f1'] + val_metrics['mean_ccc']
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_f1 = val_metrics['macro_f1']
            best_ccc = val_metrics['mean_ccc']
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(output_dir, 'best_model.pth'),
                config=config
            )
            print(f"✓ Best model saved! F1: {best_f1:.4f}, CCC: {best_ccc:.4f}\n")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if early_stopping(combined_score):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # ========== Training Complete ==========
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Best Validation CCC: {best_ccc:.4f}")
    print(f"Best Combined Score: {best_combined_score:.4f}")
    print("="*80 + "\n")
    
    # ========== Plot Training History ==========
    print("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(output_dir, 'training_history.png'))
    
    # ========== Load Best Model and Evaluate on Test Set ==========
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print_evaluation_summary(test_metrics, dataset_name='Test')
    
    # Plot confusion matrix
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        emotion_labels,
        save_path=os.path.join(output_dir, 'confusion_matrix.png'),
        title='Test Set Confusion Matrix'
    )
    
    # ========== Export Models ==========
    print("\nExporting models...")
    
    # PyTorch weights
    save_model_weights(model, os.path.join(output_dir, 'model_weights.pth'))
    
    # TorchScript
    try:
        export_to_torchscript(
            model,
            os.path.join(output_dir, 'model_torchscript.pt'),
            device=device
        )
    except Exception as e:
        print(f"TorchScript export failed: {e}")
    
    # ONNX
    try:
        export_to_onnx(
            model.cpu(),
            os.path.join(output_dir, 'model.onnx'),
            device='cpu'
        )
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    # Save final configuration with test metrics
    final_config = config.copy()
    final_config['test_metrics'] = {
        'accuracy': float(test_metrics['accuracy']),
        'macro_f1': float(test_metrics['macro_f1']),
        'mean_ccc': float(test_metrics['mean_ccc']),
        'valence_ccc': float(test_metrics['valence_ccc']),
        'arousal_ccc': float(test_metrics['arousal_ccc'])
    }
    save_config(final_config, os.path.join(output_dir, 'final_config.json'))
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - best_model.pth")
    print("  - model_weights.pth")
    print("  - model_torchscript.pt")
    print("  - model.onnx")
    print("  - config.json")
    print("  - final_config.json")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("="*80)


if __name__ == '__main__':
    main()
