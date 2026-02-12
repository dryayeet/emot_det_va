# ============================================================================
# MULTI-TASK FACIAL EMOTION RECOGNITION - GOOGLE COLAB TRAINING
# ============================================================================
# This is a multi-cell Colab notebook in Python script format
# Copy each cell to separate Colab cells in order
# ============================================================================

# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================
"""
!nvidia-smi
!pip install kaggle torch torchvision matplotlib seaborn scikit-learn pandas numpy opencv-python pillow onnx tf2onnx
"""

# ============================================================================
# CELL 2: Mount Google Drive and Setup Kaggle
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

# Upload your kaggle.json to Google Drive or use Colab files
# Then copy it to the right location
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
"""

# ============================================================================
# CELL 3: Download AffectNet Dataset from Kaggle
# ============================================================================
"""
# Download the dataset
!kaggle datasets download -d noamsegal/affectnet-training-data
!unzip -q affectnet-training-data.zip -d /content/affectnet_data

# Alternatively, if using mstjebashazida version:
# !kaggle datasets download -d mstjebashazida/affectnet
# !unzip -q affectnet.zip -d /content/affectnet_data

print("Dataset downloaded and extracted!")
"""

# ============================================================================
# CELL 4: Import Libraries
# ============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
"""

# ============================================================================
# CELL 5: Define Dataset Class
# ============================================================================
"""
class AffectNetDataset(Dataset):
    '''
    AffectNet Dataset Loader
    
    Loads images with categorical emotion labels and continuous valence/arousal values
    '''
    def __init__(self, data_df, img_dir, transform=None):
        '''
        Args:
            data_df: DataFrame with columns ['image_path', 'emotion', 'valence', 'arousal']
            img_dir: Root directory containing images
            transform: Optional transforms to apply
        '''
        self.data_df = data_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # Get image path and load image
        img_name = self.data_df.loc[idx, 'image_path']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback to black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get labels
        emotion = int(self.data_df.loc[idx, 'emotion'])
        valence = float(self.data_df.loc[idx, 'valence'])
        arousal = float(self.data_df.loc[idx, 'arousal'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, emotion, torch.tensor([valence, arousal], dtype=torch.float32)
"""

# ============================================================================
# CELL 6: Data Preparation and Loading
# ============================================================================
"""
# Update this path based on your dataset structure
DATA_DIR = '/content/affectnet_data'

# Load annotations CSV
# Adjust column names based on actual CSV structure
# Expected columns: image_path, emotion, valence, arousal
annotations_file = os.path.join(DATA_DIR, 'annotations.csv')  # Update filename

# If CSV doesn't exist, create from folder structure
# This is a placeholder - adjust based on actual dataset structure
def load_affectnet_annotations(data_dir):
    '''Load or create annotations DataFrame'''
    csv_path = os.path.join(data_dir, 'annotations.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Create annotations from folder structure if needed
        # This is dataset-specific - update based on actual structure
        print("Creating annotations from dataset structure...")
        data_list = []
        
        # Example: if images are in emotion-named folders
        emotion_map = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
            'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7
        }
        
        for emotion_name, emotion_id in emotion_map.items():
            emotion_dir = os.path.join(data_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        data_list.append({
                            'image_path': os.path.join(emotion_name, img_name),
                            'emotion': emotion_id,
                            'valence': 0.0,  # Placeholder if not available
                            'arousal': 0.0   # Placeholder if not available
                        })
        
        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False)
        print(f"Created {len(df)} annotations")
    
    return df

# Load data
df = load_affectnet_annotations(DATA_DIR)
print(f"Total samples: {len(df)}")
print(f"\\nEmotion distribution:\\n{df['emotion'].value_counts().sort_index()}")

# Train/Val/Test split (70/15/15)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['emotion'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['emotion'])

print(f"\\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
"""

# ============================================================================
# CELL 7: Data Transforms
# ============================================================================
"""
# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create datasets
train_dataset = AffectNetDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset = AffectNetDataset(val_df, DATA_DIR, transform=val_transform)
test_dataset = AffectNetDataset(test_df, DATA_DIR, transform=val_transform)

# Create dataloaders
BATCH_SIZE = 64
NUM_WORKERS = 2

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
"""

# ============================================================================
# CELL 8: Multi-Task Model Architecture
# ============================================================================
"""
class MultiTaskEmotionNet(nn.Module):
    '''
    Multi-Task Emotion Recognition Model
    
    Architecture:
    - Shared ResNet-18 backbone (pretrained on ImageNet)
    - Branch 1: Categorical emotion classification (8 classes)
    - Branch 2: Valence-Arousal regression (2 continuous values)
    '''
    def __init__(self, num_classes=8, pretrained=True):
        super(MultiTaskEmotionNet, self).__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract feature extractor (remove final FC layer)
        self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-18
        feature_dim = 512
        
        # Categorical emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Valence-Arousal regression head
        self.va_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, x):
        # Shared feature extraction
        features = self.shared_backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Multi-task outputs
        emotion_logits = self.emotion_classifier(features)
        va_output = self.va_regressor(features)
        
        return emotion_logits, va_output

# Initialize model
model = MultiTaskEmotionNet(num_classes=8, pretrained=True)
model = model.to(device)

# Print model summary
print(model)
print(f"\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
"""

# ============================================================================
# CELL 9: Loss Functions
# ============================================================================
"""
class MultiTaskLoss(nn.Module):
    '''
    Combined loss for multi-task learning
    
    L_total = alpha * CrossEntropyLoss + beta * MSELoss
    '''
    def __init__(self, alpha=1.0, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, emotion_logits, va_pred, emotion_target, va_target):
        # Classification loss
        loss_emotion = self.ce_loss(emotion_logits, emotion_target)
        
        # Regression loss
        loss_va = self.mse_loss(va_pred, va_target)
        
        # Combined loss
        total_loss = self.alpha * loss_emotion + self.beta * loss_va
        
        return total_loss, loss_emotion, loss_va

# Initialize loss
criterion = MultiTaskLoss(alpha=1.0, beta=0.5)
print(f"Loss weights - Alpha (Classification): {criterion.alpha}, Beta (Regression): {criterion.beta}")
"""

# ============================================================================
# CELL 10: Evaluation Metrics
# ============================================================================
"""
def calculate_ccc(y_true, y_pred):
    '''
    Calculate Concordance Correlation Coefficient (CCC)
    Measures agreement between predicted and actual values
    '''
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)
    return ccc

def evaluate_model(model, dataloader, criterion, device):
    '''
    Comprehensive model evaluation
    
    Returns:
        Dict with classification and regression metrics
    '''
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
        for images, emotions, va_targets in dataloader:
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
    
    # Calculate classification metrics
    accuracy = np.mean(all_emotion_preds == all_emotion_targets)
    macro_f1 = f1_score(all_emotion_targets, all_emotion_preds, average='macro')
    
    # Calculate regression metrics
    va_mse = mean_squared_error(all_va_targets, all_va_preds)
    va_mae = mean_absolute_error(all_va_targets, all_va_preds)
    
    # Separate valence and arousal
    valence_true = all_va_targets[:, 0]
    valence_pred = all_va_preds[:, 0]
    arousal_true = all_va_targets[:, 1]
    arousal_pred = all_va_preds[:, 1]
    
    # Calculate CCC for valence and arousal
    ccc_valence = calculate_ccc(valence_true, valence_pred)
    ccc_arousal = calculate_ccc(arousal_true, arousal_pred)
    mean_ccc = (ccc_valence + ccc_arousal) / 2
    
    # Pearson correlation
    pearson_valence, _ = pearsonr(valence_true, valence_pred)
    pearson_arousal, _ = pearsonr(arousal_true, arousal_pred)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'emotion_loss': total_emotion_loss / len(dataloader),
        'va_loss': total_va_loss / len(dataloader),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'va_mse': va_mse,
        'va_mae': va_mae,
        'ccc_valence': ccc_valence,
        'ccc_arousal': ccc_arousal,
        'mean_ccc': mean_ccc,
        'pearson_valence': pearson_valence,
        'pearson_arousal': pearson_arousal,
        'confusion_matrix': confusion_matrix(all_emotion_targets, all_emotion_preds)
    }
    
    return metrics

print("Evaluation functions defined!")
"""

# ============================================================================
# CELL 11: Training Configuration
# ============================================================================
"""
# Hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'val_f1': [],
    'val_ccc': []
}

# Best model tracking
best_f1 = 0.0
best_ccc = 0.0
best_combined_score = 0.0  # f1 + ccc
epochs_no_improve = 0

print(f"Training Configuration:")
print(f"- Learning Rate: {LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Epochs: {NUM_EPOCHS}")
print(f"- Optimizer: Adam")
print(f"- Scheduler: ReduceLROnPlateau")
print(f"- Early Stopping Patience: {PATIENCE}")
"""

# ============================================================================
# CELL 12: Training Loop
# ============================================================================
"""
import time

print("Starting training...\\n")

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, emotions, va_targets) in enumerate(train_loader):
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
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        preds = torch.argmax(emotion_logits, dim=1)
        train_correct += (preds == emotions).sum().item()
        train_total += emotions.size(0)
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    # Calculate training metrics
    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    # ========== VALIDATION ==========
    val_metrics = evaluate_model(model, val_loader, criterion, device)
    
    # Update history
    history['train_loss'].append(train_loss_avg)
    history['val_loss'].append(val_metrics['loss'])
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_f1'].append(val_metrics['macro_f1'])
    history['val_ccc'].append(val_metrics['mean_ccc'])
    
    # Learning rate scheduling
    scheduler.step(val_metrics['macro_f1'])
    
    # Print epoch summary
    epoch_time = time.time() - epoch_start
    print(f"\\n{'='*80}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {epoch_time:.2f}s")
    print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
    print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Val CCC - Valence: {val_metrics['ccc_valence']:.4f} | Arousal: {val_metrics['ccc_arousal']:.4f} | Mean: {val_metrics['mean_ccc']:.4f}")
    print(f"Val MSE: {val_metrics['va_mse']:.4f} | Val MAE: {val_metrics['va_mae']:.4f}")
    print(f"{'='*80}\\n")
    
    # Save best model
    combined_score = val_metrics['macro_f1'] + val_metrics['mean_ccc']
    
    if combined_score > best_combined_score:
        best_combined_score = combined_score
        best_f1 = val_metrics['macro_f1']
        best_ccc = val_metrics['mean_ccc']
        epochs_no_improve = 0
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_metrics['macro_f1'],
            'val_ccc': val_metrics['mean_ccc'],
        }, '/content/drive/MyDrive/best_model.pth')
        
        print(f"✓ Best model saved! F1: {best_f1:.4f}, CCC: {best_ccc:.4f}\\n")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)\\n")
    
    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("\\nTraining completed!")
print(f"Best Validation F1: {best_f1:.4f}")
print(f"Best Validation CCC: {best_ccc:.4f}")
"""

# ============================================================================
# CELL 13: Plot Training History
# ============================================================================
"""
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].plot(history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Macro F1
axes[1, 0].plot(history['val_f1'], label='Val Macro F1', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Macro F1')
axes[1, 0].set_title('Validation Macro F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Mean CCC
axes[1, 1].plot(history['val_ccc'], label='Val Mean CCC', color='orange')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Mean CCC')
axes[1, 1].set_title('Validation Mean CCC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_history.png', dpi=300, bbox_inches='tight')
plt.show()
"""

# ============================================================================
# CELL 14: Load Best Model and Evaluate on Test Set
# ============================================================================
"""
# Load best model
checkpoint = torch.load('/content/drive/MyDrive/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print("Best model loaded!")

# Evaluate on test set
print("\\nEvaluating on test set...")
test_metrics = evaluate_model(model, test_loader, criterion, device)

print("\\n" + "="*80)
print("TEST SET RESULTS")
print("="*80)
print(f"Loss: {test_metrics['loss']:.4f}")
print(f"\\nClassification Metrics:")
print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  - Macro F1: {test_metrics['macro_f1']:.4f}")
print(f"\\nRegression Metrics:")
print(f"  - MSE: {test_metrics['va_mse']:.4f}")
print(f"  - MAE: {test_metrics['va_mae']:.4f}")
print(f"  - CCC Valence: {test_metrics['ccc_valence']:.4f}")
print(f"  - CCC Arousal: {test_metrics['ccc_arousal']:.4f}")
print(f"  - Mean CCC: {test_metrics['mean_ccc']:.4f}")
print(f"  - Pearson Valence: {test_metrics['pearson_valence']:.4f}")
print(f"  - Pearson Arousal: {test_metrics['pearson_arousal']:.4f}")
print("="*80)
"""

# ============================================================================
# CELL 15: Plot Confusion Matrix
# ============================================================================
"""
# Emotion labels
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
"""

# ============================================================================
# CELL 16: Export Model - PyTorch (.pth)
# ============================================================================
"""
# Already saved during training as best_model.pth
# Create a clean export with just model weights
torch.save(model.state_dict(), '/content/drive/MyDrive/emotion_model_weights.pth')
print("✓ PyTorch model weights saved: emotion_model_weights.pth")
"""

# ============================================================================
# CELL 17: Export Model - TorchScript (.pt)
# ============================================================================
"""
# Set model to evaluation mode
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224).to(device)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save TorchScript model
traced_model.save('/content/drive/MyDrive/emotion_model_torchscript.pt')
print("✓ TorchScript model saved: emotion_model_torchscript.pt")

# Test loading
loaded_ts = torch.jit.load('/content/drive/MyDrive/emotion_model_torchscript.pt')
loaded_ts.eval()
print("✓ TorchScript model verified!")
"""

# ============================================================================
# CELL 18: Export Model - ONNX (.onnx)
# ============================================================================
"""
import torch.onnx

# Set model to eval mode
model.eval()
model = model.cpu()

# Example input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    '/content/drive/MyDrive/emotion_model.onnx',
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
print("✓ ONNX model saved: emotion_model.onnx")

# Verify ONNX model
import onnx
onnx_model = onnx.load('/content/drive/MyDrive/emotion_model.onnx')
onnx.checker.check_model(onnx_model)
print("✓ ONNX model verified!")
"""

# ============================================================================
# CELL 19: Export Model - Keras/H5 (.h5)
# ============================================================================
"""
# Note: Converting PyTorch to Keras requires intermediate ONNX conversion
# We'll use onnx2keras or manually reconstruct in TensorFlow

# Method 1: Using onnx2keras (install if needed)
# !pip install onnx2keras tensorflow

try:
    import onnx
    from onnx2keras import onnx_to_keras
    import tensorflow as tf
    
    # Load ONNX model
    onnx_model = onnx.load('/content/drive/MyDrive/emotion_model.onnx')
    
    # Convert to Keras
    keras_model = onnx_to_keras(onnx_model, ['input'])
    
    # Save as H5
    keras_model.save('/content/drive/MyDrive/emotion_model.h5')
    print("✓ Keras H5 model saved: emotion_model.h5")
    
except Exception as e:
    print(f"Warning: H5 export failed: {e}")
    print("Alternative: Use TensorFlow to manually reconstruct the model")
    
    # Alternative: Create equivalent Keras model
    import tensorflow as tf
    from tensorflow import keras
    
    # Define equivalent Keras model
    def create_keras_emotion_model():
        # Base ResNet18 (pretrained weights need separate loading)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        # Freeze base layers
        base_model.trainable = False
        
        # Input
        inputs = keras.Input(shape=(224, 224, 3))
        
        # Shared features
        x = base_model(inputs)
        
        # Emotion classification branch
        emotion_branch = keras.layers.Dropout(0.5)(x)
        emotion_branch = keras.layers.Dense(256, activation='relu')(emotion_branch)
        emotion_branch = keras.layers.Dropout(0.3)(emotion_branch)
        emotion_logits = keras.layers.Dense(8, name='emotion_logits')(emotion_branch)
        
        # VA regression branch
        va_branch = keras.layers.Dropout(0.5)(x)
        va_branch = keras.layers.Dense(128, activation='relu')(va_branch)
        va_branch = keras.layers.Dropout(0.3)(va_branch)
        va_output = keras.layers.Dense(2, activation='tanh', name='va_output')(va_branch)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[emotion_logits, va_output])
        return model
    
    keras_model = create_keras_emotion_model()
    keras_model.save('/content/drive/MyDrive/emotion_model_keras.h5')
    print("✓ Keras H5 model (architecture) saved: emotion_model_keras.h5")
    print("Note: Weights need to be transferred separately from PyTorch")
"""

# ============================================================================
# CELL 20: Create Model Configuration File
# ============================================================================
"""
import json

# Model configuration
config = {
    'model_name': 'MultiTaskEmotionNet',
    'architecture': 'ResNet18',
    'num_classes': 8,
    'emotion_labels': ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
    'input_size': [224, 224],
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'training': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'loss_weights': {
            'alpha': 1.0,
            'beta': 0.5
        }
    },
    'performance': {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_macro_f1': float(test_metrics['macro_f1']),
        'test_mean_ccc': float(test_metrics['mean_ccc']),
        'test_mse': float(test_metrics['va_mse']),
        'test_mae': float(test_metrics['va_mae'])
    }
}

# Save configuration
with open('/content/drive/MyDrive/model_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("✓ Model configuration saved: model_config.json")
print("\\nConfiguration:")
print(json.dumps(config, indent=2))
"""

# ============================================================================
# CELL 21: Download All Files
# ============================================================================
"""
# Create a zip with all exported models
!cd /content/drive/MyDrive && zip -r emotion_models.zip \\
    best_model.pth \\
    emotion_model_weights.pth \\
    emotion_model_torchscript.pt \\
    emotion_model.onnx \\
    emotion_model.h5 \\
    model_config.json \\
    training_history.png \\
    confusion_matrix.png

print("\\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\\nExported Files:")
print("1. best_model.pth - Full checkpoint with optimizer state")
print("2. emotion_model_weights.pth - Model weights only")
print("3. emotion_model_torchscript.pt - TorchScript format")
print("4. emotion_model.onnx - ONNX format")
print("5. emotion_model.h5 - Keras/H5 format")
print("6. model_config.json - Configuration file")
print("7. training_history.png - Training plots")
print("8. confusion_matrix.png - Confusion matrix")
print("\\nAll files saved to Google Drive!")
print("="*80)
"""
