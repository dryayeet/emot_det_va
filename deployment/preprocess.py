"""
Image Preprocessing for Emotion Recognition

Handles preprocessing of face images for model inference
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EmotionPreprocessor:
    """
    Preprocessor for emotion recognition inference
    
    Handles resizing, normalization, and tensor conversion
    """
    
    def __init__(self, target_size=(224, 224), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        """
        Initialize preprocessor
        
        Args:
            target_size (tuple): Target image size (height, width)
            mean (list): Normalization mean values
            std (list): Normalization std values
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
        
        # Create transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        print(f"Preprocessor initialized:")
        print(f"  Target size: {target_size}")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")
    
    def preprocess_cv2(self, image):
        """
        Preprocess OpenCV image (BGR format)
        
        Args:
            image (np.array): OpenCV image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, 3, H, W)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_pil(self, pil_image):
        """
        Preprocess PIL Image
        
        Args:
            pil_image (PIL.Image): PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_numpy(self, image):
        """
        Preprocess numpy array (RGB format)
        
        Args:
            image (np.array): Image in RGB format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        return self.preprocess_pil(pil_image)
    
    def preprocess_batch(self, images):
        """
        Preprocess batch of images
        
        Args:
            images (list): List of images (OpenCV, PIL, or numpy)
            
        Returns:
            torch.Tensor: Batch tensor of shape (N, 3, H, W)
        """
        batch = []
        
        for img in images:
            if isinstance(img, np.ndarray):
                # Assume BGR format if from OpenCV
                if len(img.shape) == 3 and img.shape[2] == 3:
                    tensor = self.preprocess_cv2(img)
                else:
                    raise ValueError(f"Invalid image shape: {img.shape}")
            elif isinstance(img, Image.Image):
                tensor = self.preprocess_pil(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            batch.append(tensor)
        
        # Concatenate batch
        return torch.cat(batch, dim=0)
    
    def denormalize(self, tensor):
        """
        Denormalize tensor for visualization
        
        Args:
            tensor (torch.Tensor): Normalized image tensor
            
        Returns:
            np.array: Denormalized image in RGB format [0, 255]
        """
        tensor = tensor.clone()
        
        # Denormalize
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        # Clip to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        return image


def resize_with_aspect_ratio(image, target_size, pad_color=(0, 0, 0)):
    """
    Resize image while maintaining aspect ratio with padding
    
    Args:
        image (np.array): Input image
        target_size (tuple): Target (height, width)
        pad_color (tuple): Padding color (B, G, R)
        
    Returns:
        np.array: Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Calculate padding offsets
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def augment_face_crop(face_crop, expand_ratio=1.2):
    """
    Expand face crop to include more context
    
    Useful for better emotion recognition as it includes
    surrounding facial features
    
    Args:
        face_crop (np.array): Cropped face image
        expand_ratio (float): Expansion ratio (1.2 = 20% expansion)
        
    Returns:
        np.array: Expanded face crop
    """
    h, w = face_crop.shape[:2]
    
    # Calculate expansion
    new_h = int(h * expand_ratio)
    new_w = int(w * expand_ratio)
    
    # Create expanded canvas
    expanded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # Calculate offsets
    y_offset = (new_h - h) // 2
    x_offset = (new_w - w) // 2
    
    # Place original crop in center
    expanded[y_offset:y_offset+h, x_offset:x_offset+w] = face_crop
    
    return expanded


def histogram_equalization(image):
    """
    Apply histogram equalization for better contrast
    
    Args:
        image (np.array): Input image in BGR
        
    Returns:
        np.array: Equalized image
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Apply histogram equalization to Y channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    # Convert back to BGR
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return equalized


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Better than regular histogram equalization for faces
    
    Args:
        image (np.array): Input image in BGR
        clip_limit (float): Contrast limiting threshold
        tile_grid_size (tuple): Grid size for local equalization
        
    Returns:
        np.array: Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def validate_face_crop(face_crop, min_size=64):
    """
    Validate if face crop is suitable for processing
    
    Args:
        face_crop (np.array): Cropped face image
        min_size (int): Minimum dimension size
        
    Returns:
        bool: True if valid, False otherwise
    """
    if face_crop is None or face_crop.size == 0:
        return False
    
    h, w = face_crop.shape[:2]
    
    # Check minimum size
    if h < min_size or w < min_size:
        return False
    
    # Check if image is too dark or too bright
    mean_brightness = np.mean(face_crop)
    if mean_brightness < 10 or mean_brightness > 245:
        return False
    
    return True


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for deployment
    
    Combines face crop validation, enhancement, and tensor conversion
    """
    
    def __init__(self, enhance=True, validation=True):
        """
        Initialize pipeline
        
        Args:
            enhance (bool): Apply image enhancement
            validation (bool): Validate face crops
        """
        self.preprocessor = EmotionPreprocessor()
        self.enhance = enhance
        self.validation = validation
    
    def __call__(self, face_crop):
        """
        Process face crop through pipeline
        
        Args:
            face_crop (np.array): Cropped face image
            
        Returns:
            torch.Tensor or None: Preprocessed tensor or None if invalid
        """
        # Validate
        if self.validation and not validate_face_crop(face_crop):
            return None
        
        # Enhance
        if self.enhance:
            face_crop = clahe_enhancement(face_crop)
        
        # Preprocess to tensor
        tensor = self.preprocessor.preprocess_cv2(face_crop)
        
        return tensor
