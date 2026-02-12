"""
Face Detection Module using MediaPipe

Provides real-time face detection for emotion recognition deployment
"""

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeFaceDetector:
    """
    Face detector using MediaPipe Face Detection
    
    Fast and efficient face detection suitable for real-time applications
    """
    
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initialize MediaPipe Face Detection
        
        Args:
            min_detection_confidence (float): Minimum confidence for detection (0-1)
            model_selection (int): 0 for short-range (within 2 meters),
                                  1 for full-range (within 5 meters)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        self.min_confidence = min_detection_confidence
        
        print(f"MediaPipe Face Detector initialized")
        print(f"  Min confidence: {min_detection_confidence}")
        print(f"  Model selection: {'Short-range' if model_selection == 0 else 'Full-range'}")
    
    def detect_faces(self, image):
        """
        Detect faces in image
        
        Args:
            image (np.array): Input image in BGR format (OpenCV)
            
        Returns:
            list: List of detected face bounding boxes
                Each box is (x, y, w, h) in pixel coordinates
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_detection.process(image_rgb)
        
        faces = []
        
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Get confidence score
                confidence = detection.score[0]
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': confidence
                })
        
        return faces
    
    def detect_single_face(self, image):
        """
        Detect the most confident face in image
        
        Args:
            image (np.array): Input image
            
        Returns:
            tuple: (bbox, confidence) or (None, None) if no face detected
                bbox is (x, y, w, h)
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None, None
        
        # Return face with highest confidence
        best_face = max(faces, key=lambda f: f['confidence'])
        return best_face['bbox'], best_face['confidence']
    
    def crop_face(self, image, bbox, padding=0.2):
        """
        Crop face region from image with optional padding
        
        Args:
            image (np.array): Input image
            bbox (tuple): Bounding box (x, y, w, h)
            padding (float): Padding ratio (default: 0.2 = 20% padding)
            
        Returns:
            np.array: Cropped face region
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop
    
    def draw_detections(self, image, faces):
        """
        Draw bounding boxes and confidence scores on image
        
        Args:
            image (np.array): Input image
            faces (list): List of face detections
            
        Returns:
            np.array: Image with drawn detections
        """
        output_image = image.copy()
        
        for face in faces:
            bbox = face['bbox']
            confidence = face['confidence']
            
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            text = f"Face: {confidence:.2f}"
            cv2.putText(
                output_image, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return output_image
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


class DlibFaceDetector:
    """
    Alternative face detector using dlib (if available)
    
    Fallback option if MediaPipe is not available
    """
    
    def __init__(self):
        """Initialize dlib face detector"""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            print("Dlib Face Detector initialized")
        except ImportError:
            raise ImportError("dlib not installed. Install with: pip install dlib")
    
    def detect_faces(self, image):
        """Detect faces using dlib"""
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detections = self.detector(gray, 1)
        
        faces = []
        for detection in detections:
            x = detection.left()
            y = detection.top()
            w = detection.right() - x
            h = detection.bottom() - y
            
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0  # dlib doesn't provide confidence
            })
        
        return faces
    
    def detect_single_face(self, image):
        """Detect single face"""
        faces = self.detect_faces(image)
        if not faces:
            return None, None
        return faces[0]['bbox'], faces[0]['confidence']
    
    def crop_face(self, image, bbox, padding=0.2):
        """Crop face region"""
        x, y, w, h = bbox
        
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]


def create_face_detector(detector_type='mediapipe', **kwargs):
    """
    Factory function to create face detector
    
    Args:
        detector_type (str): 'mediapipe' or 'dlib'
        **kwargs: Additional detector-specific arguments
        
    Returns:
        FaceDetector: Initialized face detector
    """
    if detector_type.lower() == 'mediapipe':
        return MediaPipeFaceDetector(**kwargs)
    elif detector_type.lower() == 'dlib':
        return DlibFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
