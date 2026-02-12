"""
Emotion Recognition Inference

Handles loading trained models and performing inference
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# Emotion labels (AffectNet 8 classes)
EMOTION_LABELS = [
    'Neutral', 'Happy', 'Sad', 'Surprise',
    'Fear', 'Disgust', 'Anger', 'Contempt'
]


# Emotion colors for visualization (BGR format)
EMOTION_COLORS = {
    'Neutral': (200, 200, 200),    # Gray
    'Happy': (0, 255, 0),           # Green
    'Sad': (255, 0, 0),             # Blue
    'Surprise': (0, 255, 255),      # Yellow
    'Fear': (128, 0, 128),          # Purple
    'Disgust': (0, 128, 0),         # Dark Green
    'Anger': (0, 0, 255),           # Red
    'Contempt': (128, 128, 0)       # Teal
}


class EmotionRecognizer:
    """
    Multi-Task Emotion Recognition Inference Engine
    
    Performs inference for:
    1. Categorical emotion classification
    2. Valence-Arousal regression
    """
    
    def __init__(self, model_path, config_path=None, device='cpu'):
        """
        Initialize emotion recognizer
        
        Args:
            model_path (str): Path to trained model weights (.pth file)
            config_path (str, optional): Path to model config JSON
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.emotion_labels = EMOTION_LABELS
        self.emotion_colors = EMOTION_COLORS
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'num_classes': 8,
                'architecture': 'resnet18'
            }
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"EmotionRecognizer initialized on {device}")
        print(f"  Model: {model_path}")
        print(f"  Architecture: {self.config.get('architecture', 'Unknown')}")
    
    def _load_model(self, model_path):
        """Load trained model from file"""
        # Import model architecture
        from training.model import MultiTaskEmotionNet, MobileNetV2EmotionNet
        
        # Determine architecture
        architecture = self.config.get('architecture', 'resnet18').lower()
        num_classes = self.config.get('num_classes', 8)
        
        # Create model
        if architecture == 'resnet18':
            model = MultiTaskEmotionNet(num_classes=num_classes, pretrained=False)
        elif architecture == 'mobilenetv2':
            model = MobileNetV2EmotionNet(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load weights
        if model_path.endswith('.pth'):
            # Load checkpoint or weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(self, input_tensor):
        """
        Perform inference on input tensor
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor, shape (1, 3, 224, 224)
            
        Returns:
            dict: Prediction results containing:
                - emotion_id: Predicted emotion class ID
                - emotion: Emotion label name
                - confidence: Prediction confidence
                - probabilities: Probability distribution over all classes
                - valence: Predicted valence value [-1, 1]
                - arousal: Predicted arousal value [-1, 1]
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        emotion_logits, va_output = self.model(input_tensor)
        
        # Get emotion predictions
        probabilities = F.softmax(emotion_logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        emotion_id = predicted_class.item()
        emotion_label = self.emotion_labels[emotion_id]
        confidence_score = confidence.item()
        
        # Get valence-arousal values
        valence, arousal = va_output[0].cpu().numpy()
        
        results = {
            'emotion_id': emotion_id,
            'emotion': emotion_label,
            'confidence': confidence_score,
            'probabilities': probabilities[0].cpu().numpy(),
            'valence': float(valence),
            'arousal': float(arousal)
        }
        
        return results
    
    def predict_batch(self, input_tensors):
        """
        Perform batch inference
        
        Args:
            input_tensors (torch.Tensor): Batch of images, shape (N, 3, 224, 224)
            
        Returns:
            list: List of prediction dictionaries
        """
        input_tensors = input_tensors.to(self.device)
        
        # Forward pass
        emotion_logits, va_outputs = self.model(input_tensors)
        
        # Get predictions
        probabilities = F.softmax(emotion_logits, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        
        results = []
        for i in range(input_tensors.size(0)):
            emotion_id = predicted_classes[i].item()
            emotion_label = self.emotion_labels[emotion_id]
            confidence_score = confidences[i].item()
            valence, arousal = va_outputs[i].cpu().numpy()
            
            results.append({
                'emotion_id': emotion_id,
                'emotion': emotion_label,
                'confidence': confidence_score,
                'probabilities': probabilities[i].cpu().numpy(),
                'valence': float(valence),
                'arousal': float(arousal)
            })
        
        return results
    
    def get_top_k_emotions(self, probabilities, k=3):
        """
        Get top-k predicted emotions
        
        Args:
            probabilities (np.array): Probability distribution
            k (int): Number of top predictions to return
            
        Returns:
            list: List of (emotion, probability) tuples
        """
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        
        top_emotions = [
            (self.emotion_labels[idx], probabilities[idx])
            for idx in top_k_indices
        ]
        
        return top_emotions
    
    def interpret_valence_arousal(self, valence, arousal):
        """
        Interpret valence-arousal values
        
        Args:
            valence (float): Valence value [-1, 1]
            arousal (float): Arousal value [-1, 1]
            
        Returns:
            dict: Interpretation with quadrant and description
        """
        # Determine quadrant
        if valence > 0 and arousal > 0:
            quadrant = "High Arousal Positive"
            description = "Excited, Alert, Elated"
        elif valence > 0 and arousal < 0:
            quadrant = "Low Arousal Positive"
            description = "Calm, Relaxed, Content"
        elif valence < 0 and arousal > 0:
            quadrant = "High Arousal Negative"
            description = "Tense, Angry, Stressed"
        else:  # valence < 0 and arousal < 0
            quadrant = "Low Arousal Negative"
            description = "Sad, Depressed, Bored"
        
        return {
            'quadrant': quadrant,
            'description': description,
            'valence_label': 'Positive' if valence > 0 else 'Negative',
            'arousal_label': 'High' if arousal > 0 else 'Low'
        }
    
    def get_emotion_color(self, emotion):
        """
        Get color for emotion visualization
        
        Args:
            emotion (str): Emotion label
            
        Returns:
            tuple: BGR color
        """
        return self.emotion_colors.get(emotion, (255, 255, 255))


class ONNXEmotionRecognizer:
    """
    ONNX Runtime-based emotion recognizer
    
    Faster CPU inference using ONNX Runtime
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize ONNX recognizer
        
        Args:
            model_path (str): Path to ONNX model file
            device (str): 'cpu' or 'cuda'
        """
        import onnxruntime as ort
        
        self.emotion_labels = EMOTION_LABELS
        self.emotion_colors = EMOTION_COLORS
        
        # Create ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"ONNX EmotionRecognizer initialized on {device}")
        print(f"  Model: {model_path}")
    
    def predict(self, input_tensor):
        """
        Perform ONNX inference
        
        Args:
            input_tensor (torch.Tensor or np.array): Input image
            
        Returns:
            dict: Prediction results
        """
        # Convert to numpy if torch tensor
        if isinstance(input_tensor, torch.Tensor):
            input_array = input_tensor.cpu().numpy()
        else:
            input_array = input_tensor
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_array})
        
        emotion_logits = outputs[0]
        va_output = outputs[1]
        
        # Process outputs
        probabilities = self._softmax(emotion_logits[0])
        emotion_id = np.argmax(probabilities)
        confidence = probabilities[emotion_id]
        
        valence, arousal = va_output[0]
        
        results = {
            'emotion_id': int(emotion_id),
            'emotion': self.emotion_labels[emotion_id],
            'confidence': float(confidence),
            'probabilities': probabilities,
            'valence': float(valence),
            'arousal': float(arousal)
        }
        
        return results
    
    @staticmethod
    def _softmax(x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_emotion_color(self, emotion):
        """Get color for emotion"""
        return self.emotion_colors.get(emotion, (255, 255, 255))


def create_recognizer(model_path, config_path=None, device='cpu', use_onnx=False):
    """
    Factory function to create emotion recognizer
    
    Args:
        model_path (str): Path to model file
        config_path (str, optional): Path to config JSON
        device (str): Device to use
        use_onnx (bool): Use ONNX Runtime if True
        
    Returns:
        EmotionRecognizer: Initialized recognizer
    """
    if use_onnx or model_path.endswith('.onnx'):
        return ONNXEmotionRecognizer(model_path, device=device)
    else:
        return EmotionRecognizer(model_path, config_path=config_path, device=device)
