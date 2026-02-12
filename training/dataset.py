"""
AffectNet Dataset Loader

This module provides dataset handling for the AffectNet facial emotion recognition dataset.
Supports multi-task learning with categorical emotions and continuous valence/arousal values.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class AffectNetDataset(Dataset):
    """
    AffectNet Dataset for Multi-Task Emotion Recognition
    
    Loads facial images with:
    - Categorical emotion labels (8 classes)
    - Continuous valence values [-1, 1]
    - Continuous arousal values [-1, 1]
    """
    
    def __init__(self, data_df, img_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            data_df (pd.DataFrame): DataFrame with columns ['image_path', 'emotion', 'valence', 'arousal']
            img_dir (str): Root directory containing images
            transform (callable, optional): Optional transform to apply to images
        """
        self.data_df = data_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Emotion label mapping
        self.emotion_labels = [
            'Neutral', 'Happy', 'Sad', 'Surprise', 
            'Fear', 'Disgust', 'Anger', 'Contempt'
        ]
        
    def __len__(self):
        """Return total number of samples"""
        return len(self.data_df)
    
    def __getitem__(self, idx):
        """
        Get single sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image, emotion_label, va_values)
                - image: Transformed PIL Image or Tensor
                - emotion_label: Integer emotion class [0-7]
                - va_values: Tensor of shape (2,) with [valence, arousal]
        """
        # Get image path
        img_name = self.data_df.loc[idx, 'image_path']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get labels
        emotion = int(self.data_df.loc[idx, 'emotion'])
        valence = float(self.data_df.loc[idx, 'valence'])
        arousal = float(self.data_df.loc[idx, 'arousal'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create valence-arousal tensor
        va_values = torch.tensor([valence, arousal], dtype=torch.float32)
        
        return image, emotion, va_values
    
    def get_emotion_name(self, emotion_id):
        """
        Convert emotion ID to name
        
        Args:
            emotion_id (int): Emotion class ID [0-7]
            
        Returns:
            str: Emotion name
        """
        return self.emotion_labels[emotion_id]


def load_affectnet_data(data_dir, annotation_file=None):
    """
    Load AffectNet dataset annotations
    
    Args:
        data_dir (str): Root directory of dataset
        annotation_file (str, optional): Path to annotation CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['image_path', 'emotion', 'valence', 'arousal']
    """
    # Try to load existing annotation file
    if annotation_file and os.path.exists(annotation_file):
        df = pd.read_csv(annotation_file)
        print(f"Loaded {len(df)} annotations from {annotation_file}")
        return df
    
    # Check for default annotation file
    default_csv = os.path.join(data_dir, 'annotations.csv')
    if os.path.exists(default_csv):
        df = pd.read_csv(default_csv)
        print(f"Loaded {len(df)} annotations from {default_csv}")
        return df
    
    # Create annotations from folder structure
    print("No annotation file found. Creating from folder structure...")
    data_list = []
    
    # Emotion mapping
    emotion_map = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
        'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7
    }
    
    # Scan directories
    for emotion_name, emotion_id in emotion_map.items():
        emotion_dir = os.path.join(data_dir, emotion_name)
        if os.path.exists(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    data_list.append({
                        'image_path': os.path.join(emotion_name, img_name),
                        'emotion': emotion_id,
                        'valence': 0.0,  # Placeholder if not available
                        'arousal': 0.0   # Placeholder if not available
                    })
    
    if not data_list:
        raise ValueError(f"No images found in {data_dir}. Please check dataset structure.")
    
    df = pd.DataFrame(data_list)
    
    # Save annotations
    df.to_csv(default_csv, index=False)
    print(f"Created {len(df)} annotations and saved to {default_csv}")
    
    return df


def create_data_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split dataset into train/validation/test sets
    
    Args:
        df (pd.DataFrame): Full dataset
        train_ratio (float): Proportion for training (default: 0.7)
        val_ratio (float): Proportion for validation (default: 0.15)
        test_ratio (float): Proportion for testing (default: 0.15)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df['emotion']
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_df['emotion']
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def get_class_distribution(df):
    """
    Get emotion class distribution
    
    Args:
        df (pd.DataFrame): Dataset DataFrame
        
    Returns:
        pd.Series: Class counts
    """
    emotion_labels = [
        'Neutral', 'Happy', 'Sad', 'Surprise',
        'Fear', 'Disgust', 'Anger', 'Contempt'
    ]
    
    counts = df['emotion'].value_counts().sort_index()
    counts.index = [emotion_labels[i] for i in counts.index]
    
    return counts
