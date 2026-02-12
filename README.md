# Multi-Task Facial Emotion Recognition

A deep learning system for real-time facial emotion recognition using multi-task learning.

## Features

- **Categorical Emotion Classification**: Predicts 8 emotion classes (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **Valence-Arousal Regression**: Predicts continuous valence and arousal values in range [-1, 1]
- **Real-Time Processing**: 15-30 FPS on CPU, 60+ FPS on GPU
- **Multiple Export Formats**: PyTorch (.pth), TorchScript (.pt), ONNX (.onnx), Keras (.h5)

## Project Structure

```
emot_recog/
├── deployment/
│   ├── face_detector.py        # MediaPipe face detection
│   ├── inference.py            # Model inference
│   ├── main.py                 # Webcam application
│   └── preprocess.py           # Image preprocessing
├── docs/
│   ├── DELIVERY_OVERVIEW.md
│   ├── PROJECT_SUMMARY.md
│   ├── SETUP_GUIDE.md
│   └── emotion_recognition_project_file_struct.txt
├── training/
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── evaluate.py             # Evaluation metrics
│   ├── losses.py               # Loss functions
│   ├── model.py                # Model architectures
│   ├── train.py                # Main training script
│   └── utils.py                # Utility functions
├── .gitignore
├── colab_training_notebook.py  # Google Colab training script
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd emot_recog

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Training

### Google Colab

1. Upload `colab_training_notebook.py` to Google Colab
2. Configure Kaggle API for AffectNet dataset access
3. Run cells sequentially to complete training
4. Download trained models from Google Drive

### Local Training

```bash
python training/train.py \
    --data_dir /path/to/affectnet \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --output_dir outputs/
```

## Deployment

### Command Line

```bash
# PyTorch model
python deployment/main.py --model models/model_weights.pth --device cpu

# ONNX model (faster CPU inference)
python deployment/main.py --model models/model.onnx --onnx --device cpu
```

### Programmatic Usage

```python
from deployment.face_detector import MediaPipeFaceDetector
from deployment.preprocess import EmotionPreprocessor
from deployment.inference import create_recognizer
import cv2

detector = MediaPipeFaceDetector()
preprocessor = EmotionPreprocessor()
recognizer = create_recognizer('models/model_weights.pth', device='cpu')

image = cv2.imread('face.jpg')
bbox, _ = detector.detect_single_face(image)

if bbox:
    face_crop = detector.crop_face(image, bbox)
    input_tensor = preprocessor.preprocess_cv2(face_crop)
    results = recognizer.predict(input_tensor)
    
    print(f"Emotion: {results['emotion']}")
    print(f"Valence: {results['valence']:.2f}, Arousal: {results['arousal']:.2f}")
```

## Model Architecture

- **Backbone**: ResNet-18 (pretrained on ImageNet) or MobileNetV2
- **Shared Feature Extraction**: CNN backbone with 512-dimensional features
- **Dual Output Heads**: Classification head (8 classes) and regression head (valence, arousal)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss Function | CrossEntropy (weight=1.0) + MSE (weight=0.5) |

## Evaluation Metrics

- **Classification**: Accuracy, Macro F1, Confusion Matrix
- **Regression**: MSE, MAE, Concordance Correlation Coefficient (CCC)

## Webcam Controls

| Key | Action |
|-----|--------|
| q | Quit |
| v | Toggle VA visualization |
| p | Toggle probability bars |
| s | Save screenshot |

## Dataset

AffectNet dataset from Kaggle (`noamsegal/affectnet-training-data`):
- ~400,000 facial images
- 8 emotion classes
- Valence and arousal annotations

## References

1. Mollahosseini, A., et al. "AffectNet: A database for facial expression, valence, and arousal computing in the wild." IEEE TAC (2017).
2. He, K., et al. "Deep residual learning for image recognition." CVPR (2016).
3. Lugaresi, C., et al. "MediaPipe: A framework for building perception pipelines." arXiv (2019).

## License

This project is provided for educational and research purposes.
