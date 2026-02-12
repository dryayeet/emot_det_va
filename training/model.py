"""
Multi-Task Emotion Recognition Model

Implements a multi-task deep learning architecture for:
1. Categorical emotion classification (8 classes)
2. Valence-Arousal regression (continuous values)

Architecture:
- Shared ResNet-18 backbone (pretrained on ImageNet)
- Separate task-specific heads
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskEmotionNet(nn.Module):
    """
    Multi-Task Emotion Recognition Network
    
    Uses transfer learning with ResNet-18 as shared feature extractor,
    followed by two task-specific heads:
    - Emotion classification head (8 classes)
    - Valence-Arousal regression head (2 continuous outputs)
    """
    
    def __init__(self, num_classes=8, pretrained=True, dropout_rate=0.5):
        """
        Initialize model
        
        Args:
            num_classes (int): Number of emotion classes (default: 8)
            pretrained (bool): Use ImageNet pretrained weights (default: True)
            dropout_rate (float): Dropout probability (default: 0.5)
        """
        super(MultiTaskEmotionNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract feature extractor (all layers except final FC)
        # This creates a shared backbone for both tasks
        self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet-18 feature dimension
        self.feature_dim = 512
        
        # ========== Task 1: Emotion Classification Head ==========
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Lower dropout for second layer
            nn.Linear(256, num_classes)
        )
        
        # ========== Task 2: Valence-Arousal Regression Head ==========
        self.va_regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 2),
            nn.Tanh()  # Output range [-1, 1] for valence and arousal
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images, shape (batch_size, 3, 224, 224)
            
        Returns:
            tuple: (emotion_logits, va_output)
                - emotion_logits: shape (batch_size, num_classes)
                - va_output: shape (batch_size, 2) - [valence, arousal]
        """
        # Shared feature extraction
        features = self.shared_backbone(x)
        
        # Flatten features: (batch_size, 512, 1, 1) -> (batch_size, 512)
        features = features.view(features.size(0), -1)
        
        # Task-specific outputs
        emotion_logits = self.emotion_classifier(features)
        va_output = self.va_regressor(features)
        
        return emotion_logits, va_output
    
    def get_emotion_prediction(self, x):
        """
        Get emotion class prediction with confidence
        
        Args:
            x (torch.Tensor): Input image
            
        Returns:
            tuple: (predicted_class, confidence, probabilities)
        """
        self.eval()
        with torch.no_grad():
            emotion_logits, _ = self.forward(x)
            probabilities = torch.softmax(emotion_logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()
    
    def get_va_prediction(self, x):
        """
        Get valence-arousal prediction
        
        Args:
            x (torch.Tensor): Input image
            
        Returns:
            numpy.ndarray: [valence, arousal] values in range [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            _, va_output = self.forward(x)
            
        return va_output.cpu().numpy()
    
    def freeze_backbone(self):
        """Freeze shared backbone for fine-tuning only task heads"""
        for param in self.shared_backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only task heads will be trained.")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end training"""
        for param in self.shared_backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Full model will be trained.")
    
    def count_parameters(self):
        """
        Count total and trainable parameters
        
        Returns:
            dict: Parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class MobileNetV2EmotionNet(nn.Module):
    """
    Lightweight alternative using MobileNetV2
    
    Better for CPU inference and real-time applications
    """
    
    def __init__(self, num_classes=8, pretrained=True, dropout_rate=0.5):
        """
        Initialize lightweight model
        
        Args:
            num_classes (int): Number of emotion classes
            pretrained (bool): Use ImageNet pretrained weights
            dropout_rate (float): Dropout probability
        """
        super(MobileNetV2EmotionNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract feature extractor
        self.shared_backbone = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # MobileNetV2 feature dimension
        self.feature_dim = 1280
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )
        
        # VA regression head
        self.va_regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, 2),
            nn.Tanh()
        )
    
    def forward(self, x):
        """Forward pass"""
        features = self.shared_backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        
        emotion_logits = self.emotion_classifier(features)
        va_output = self.va_regressor(features)
        
        return emotion_logits, va_output


def create_model(architecture='resnet18', num_classes=8, pretrained=True):
    """
    Factory function to create emotion recognition model
    
    Args:
        architecture (str): Model architecture ('resnet18' or 'mobilenetv2')
        num_classes (int): Number of emotion classes
        pretrained (bool): Use pretrained weights
        
    Returns:
        nn.Module: Emotion recognition model
    """
    if architecture.lower() == 'resnet18':
        model = MultiTaskEmotionNet(num_classes=num_classes, pretrained=pretrained)
        print(f"Created ResNet-18 based model")
    elif architecture.lower() == 'mobilenetv2':
        model = MobileNetV2EmotionNet(num_classes=num_classes, pretrained=pretrained)
        print(f"Created MobileNetV2 based model")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Print parameter count
    param_info = model.count_parameters() if hasattr(model, 'count_parameters') else {}
    if param_info:
        print(f"Total parameters: {param_info['total']:,}")
        print(f"Trainable parameters: {param_info['trainable']:,}")
    
    return model
