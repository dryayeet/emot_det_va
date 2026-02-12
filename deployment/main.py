"""
Real-Time Emotion Recognition from Webcam

Main application for real-time facial emotion recognition using webcam feed
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Import deployment modules
from face_detector import MediaPipeFaceDetector
from preprocess import EmotionPreprocessor
from inference import create_recognizer


class EmotionRecognitionApp:
    """
    Real-time emotion recognition application
    
    Processes webcam feed to detect faces and recognize emotions
    """
    
    def __init__(self, model_path, config_path=None, device='cpu',
                 use_onnx=False, camera_id=0, display_fps=True):
        """
        Initialize application
        
        Args:
            model_path (str): Path to trained model
            config_path (str, optional): Path to model config
            device (str): Device for inference ('cpu' or 'cuda')
            use_onnx (bool): Use ONNX Runtime
            camera_id (int): Camera device ID
            display_fps (bool): Display FPS on screen
        """
        print("Initializing Emotion Recognition App...")
        
        # Initialize components
        self.face_detector = MediaPipeFaceDetector(
            min_detection_confidence=0.5,
            model_selection=0  # Short-range for webcam
        )
        
        self.preprocessor = EmotionPreprocessor()
        
        self.recognizer = create_recognizer(
            model_path=model_path,
            config_path=config_path,
            device=device,
            use_onnx=use_onnx
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.display_fps = display_fps
        self.fps = 0
        
        # UI settings
        self.show_va_space = True  # Show valence-arousal space
        self.show_probabilities = True  # Show emotion probabilities
        
        print("Initialization complete!")
        print(f"Camera: Device {camera_id}")
        print(f"Device: {device}")
        print("Press 'q' to quit, 'v' to toggle VA space, 'p' to toggle probabilities")
    
    def draw_results(self, frame, bbox, results):
        """
        Draw detection and prediction results on frame
        
        Args:
            frame (np.array): Input frame
            bbox (tuple): Face bounding box (x, y, w, h)
            results (dict): Prediction results
            
        Returns:
            np.array: Frame with drawn results
        """
        x, y, w, h = bbox
        
        # Get emotion and color
        emotion = results['emotion']
        confidence = results['confidence']
        color = self.recognizer.get_emotion_color(emotion)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background for text
        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw valence-arousal
        valence = results['valence']
        arousal = results['arousal']
        va_text = f"V: {valence:.2f} A: {arousal:.2f}"
        cv2.putText(frame, va_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_info_panel(self, frame, results):
        """
        Draw information panel with probabilities and VA space
        
        Args:
            frame (np.array): Input frame
            results (dict): Prediction results
            
        Returns:
            np.array: Frame with info panel
        """
        h, w = frame.shape[:2]
        panel_width = 300
        panel_x = w - panel_width
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        y_offset = 30
        
        # Title
        cv2.putText(frame, "Emotion Analysis", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 40
        
        # Emotion probabilities
        if self.show_probabilities:
            cv2.putText(frame, "Probabilities:", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            
            probabilities = results['probabilities']
            emotion_labels = self.recognizer.emotion_labels
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            
            for i in sorted_indices[:5]:  # Show top 5
                emotion = emotion_labels[i]
                prob = probabilities[i]
                color = self.recognizer.get_emotion_color(emotion)
                
                # Draw bar
                bar_width = int(prob * 200)
                cv2.rectangle(frame, (panel_x + 10, y_offset - 10),
                            (panel_x + 10 + bar_width, y_offset + 5), color, -1)
                
                # Draw text
                text = f"{emotion}: {prob:.2%}"
                cv2.putText(frame, text, (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset += 25
        
        # Valence-Arousal space
        if self.show_va_space:
            y_offset += 20
            cv2.putText(frame, "Valence-Arousal:", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
            
            # Draw VA space
            space_size = 200
            space_x = panel_x + 50
            space_y = y_offset
            
            # Draw axes
            cv2.rectangle(frame, (space_x, space_y),
                         (space_x + space_size, space_y + space_size), (100, 100, 100), 1)
            cv2.line(frame, (space_x + space_size//2, space_y),
                    (space_x + space_size//2, space_y + space_size), (100, 100, 100), 1)
            cv2.line(frame, (space_x, space_y + space_size//2),
                    (space_x + space_size, space_y + space_size//2), (100, 100, 100), 1)
            
            # Labels
            cv2.putText(frame, "High A", (space_x + space_size//2 - 30, space_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            cv2.putText(frame, "Low A", (space_x + space_size//2 - 25, space_y + space_size + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            cv2.putText(frame, "Neg", (space_x - 30, space_y + space_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            cv2.putText(frame, "Pos", (space_x + space_size + 5, space_y + space_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            
            # Plot point
            valence = results['valence']
            arousal = results['arousal']
            
            # Map [-1, 1] to [0, space_size]
            point_x = int((valence + 1) * space_size / 2)
            point_y = int((1 - arousal) * space_size / 2)  # Invert Y axis
            
            color = self.recognizer.get_emotion_color(results['emotion'])
            cv2.circle(frame, (space_x + point_x, space_y + point_y), 8, color, -1)
            cv2.circle(frame, (space_x + point_x, space_y + point_y), 8, (255, 255, 255), 2)
        
        return frame
    
    def draw_fps(self, frame):
        """Draw FPS counter"""
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Args:
            frame (np.array): Input frame from webcam
            
        Returns:
            np.array: Processed frame with annotations
        """
        # Detect face
        bbox, confidence = self.face_detector.detect_single_face(frame)
        
        if bbox is not None:
            # Crop face
            face_crop = self.face_detector.crop_face(frame, bbox, padding=0.2)
            
            # Preprocess
            input_tensor = self.preprocessor.preprocess_cv2(face_crop)
            
            # Predict emotion
            results = self.recognizer.predict(input_tensor)
            
            # Draw results
            frame = self.draw_results(frame, bbox, results)
            frame = self.draw_info_panel(frame, results)
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Run the application"""
        print("\nStarting emotion recognition...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'v' - Toggle Valence-Arousal space")
        print("  'p' - Toggle Probability bars")
        print("  's' - Save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    self.fps = frame_count / elapsed_time
                
                # Draw FPS
                if self.display_fps:
                    processed_frame = self.draw_fps(processed_frame)
                
                # Display frame
                cv2.imshow('Emotion Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('v'):
                    self.show_va_space = not self.show_va_space
                    print(f"VA space: {'ON' if self.show_va_space else 'OFF'}")
                elif key == ord('p'):
                    self.show_probabilities = not self.show_probabilities
                    print(f"Probabilities: {'ON' if self.show_probabilities else 'OFF'}")
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print(f"\nSession Statistics:")
            print(f"  Total frames: {frame_count}")
            print(f"  Average FPS: {self.fps:.2f}")
            print(f"  Duration: {elapsed_time:.2f}s")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Real-Time Emotion Recognition from Webcam'
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth or .onnx)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config JSON')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--onnx', action='store_true',
                       help='Use ONNX Runtime for inference')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create and run application
    app = EmotionRecognitionApp(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        use_onnx=args.onnx,
        camera_id=args.camera,
        display_fps=not args.no_fps
    )
    
    app.run()


if __name__ == '__main__':
    main()
