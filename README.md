# Multi-Task Facial Emotion Recognition

A production-ready deep learning system for real-time facial emotion recognition with multi-task learning.

## 🎯 Features

### Multi-Task Learning
- **Categorical Emotion Classification**: Predicts 8 emotion classes (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **Valence-Arousal Regression**: Predicts continuous valence and arousal values in range [-1, 1]

### Architecture
- **Backbone**: ResNet-18 (pretrained on ImageNet) or MobileNetV2
- **Shared Feature Extraction**: Efficient multi-task learning with shared CNN backbone
- **Dual Output Heads**: Separate heads for classification and regression

### Deployment
- **Real-Time Processing**: ~15-30 FPS on CPU
- **MediaPipe Face Detection**: Fast and accurate face detection
- **Multiple Export Formats**: PyTorch (.pth), TorchScript (.pt), ONNX (.onnx), Keras (.h5)

---

## 📁 Project Structure

```
emotion_recognition_project/
│
├── training/                    # Training modules
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── model.py                # Model architectures
│   ├── losses.py               # Loss functions
│   ├── evaluate.py             # Evaluation metrics
│   ├── utils.py                # Utility functions
│   └── train.py                # Main training script
│
├── deployment/                  # Deployment modules
│   ├── face_detector.py        # MediaPipe face detection
│   ├── preprocess.py           # Image preprocessing
│   ├── inference.py            # Model inference
│   └── main.py                 # Webcam application
│
├── models/                      # Saved models (created after training)
├── docs/                        # Documentation
└── requirements.txt             # Dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd emotion_recognition_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training on Google Colab

#### Step 1: Setup Kaggle API
1. Create a Kaggle account and generate API token from https://www.kaggle.com/settings
2. Download `kaggle.json` and upload to Google Drive

#### Step 2: Run Training Notebook
1. Open Google Colab
2. Upload the provided `colab_training_notebook.py` file
3. Copy each cell section to separate Colab cells
4. Run cells in order:
   - Cell 1: Install dependencies
   - Cell 2: Mount Google Drive
   - Cell 3: Download AffectNet dataset
   - Cell 4-21: Complete training pipeline

#### Step 3: Download Trained Models
After training completes, download from Google Drive:
- `best_model.pth` - Full checkpoint
- `model_weights.pth` - Weights only
- `model_torchscript.pt` - TorchScript format
- `model.onnx` - ONNX format
- `model.h5` - Keras format (optional)
- `model_config.json` - Model configuration

### 3. Local Deployment

#### Option A: Using Command Line

```bash
# Run with PyTorch model
python deployment/main.py --model models/model_weights.pth --device cpu

# Run with ONNX model (faster CPU inference)
python deployment/main.py --model models/model.onnx --onnx --device cpu

# With custom camera
python deployment/main.py --model models/model_weights.pth --camera 1
```

#### Option B: Programmatic Usage

```python
from deployment.face_detector import MediaPipeFaceDetector
from deployment.preprocess import EmotionPreprocessor
from deployment.inference import create_recognizer
import cv2

# Initialize components
detector = MediaPipeFaceDetector()
preprocessor = EmotionPreprocessor()
recognizer = create_recognizer('models/model_weights.pth', device='cpu')

# Process single image
image = cv2.imread('face.jpg')
bbox, _ = detector.detect_single_face(image)

if bbox:
    face_crop = detector.crop_face(image, bbox)
    input_tensor = preprocessor.preprocess_cv2(face_crop)
    results = recognizer.predict(input_tensor)
    
    print(f"Emotion: {results['emotion']}")
    print(f"Confidence: {results['confidence']:.2f}")
    print(f"Valence: {results['valence']:.2f}")
    print(f"Arousal: {results['arousal']:.2f}")
```

---

## 📊 Training Details

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch Size** | 64 | Optimal GPU memory usage; stable gradients |
| **Learning Rate** | 0.001 | Standard Adam starting point |
| **Epochs** | 50 | Sufficient for convergence with early stopping |
| **Optimizer** | Adam | Adaptive LR; excellent for multi-task |
| **Scheduler** | ReduceLROnPlateau | Reduces LR when validation plateaus |
| **Loss Weights** | α=1.0, β=0.5 | Classification primary; regression secondary |
| **Data Augmentation** | Flip + Rotation ±15° | Preserves facial structure |

### Loss Function

```
L_total = α × CrossEntropyLoss + β × MSELoss
```

Where:
- α = 1.0 (classification weight)
- β = 0.5 (regression weight)

### Data Split
- Training: 70%
- Validation: 15%
- Testing: 15%

---

## 📈 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Macro F1**: F1 score averaged across classes
- **Confusion Matrix**: Per-class performance

### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **CCC**: Concordance Correlation Coefficient (primary metric)
- **Pearson Correlation**: Linear correlation

### Model Selection
Best model selected based on:
```
Combined Score = Macro F1 + Mean CCC
```

---

## 🎮 Webcam Controls

When running the webcam application:

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `v` | Toggle Valence-Arousal space visualization |
| `p` | Toggle probability bars |
| `s` | Save screenshot |

---

## 🔧 Advanced Usage

### Custom Training

```bash
python training/train.py \
    --data_dir /path/to/affectnet \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --alpha 1.0 \
    --beta 0.5 \
    --output_dir outputs/run1
```

### Model Export

```python
from training.utils import export_to_onnx, export_to_torchscript
from training.model import create_model

# Load model
model = create_model('resnet18', num_classes=8)
model.load_state_dict(torch.load('model_weights.pth'))

# Export to ONNX
export_to_onnx(model, 'model.onnx', device='cpu')

# Export to TorchScript
export_to_torchscript(model, 'model.pt', device='cpu')
```

### Batch Inference

```python
from deployment.inference import create_recognizer
from deployment.preprocess import EmotionPreprocessor
import torch

recognizer = create_recognizer('model.pth')
preprocessor = EmotionPreprocessor()

# Prepare batch
images = [...]  # List of face crops
batch_tensor = preprocessor.preprocess_batch(images)

# Predict
results = recognizer.predict_batch(batch_tensor)

for i, result in enumerate(results):
    print(f"Image {i}: {result['emotion']} ({result['confidence']:.2f})")
```

---

## 📝 Dataset Information

### AffectNet Dataset
- **Source**: Kaggle (`noamsegal/affectnet-training-data`)
- **Size**: ~400,000 facial images
- **Classes**: 8 emotions (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **Annotations**: Categorical labels + Valence/Arousal values

### Expected CSV Format

```csv
image_path,emotion,valence,arousal
train/0001.jpg,1,0.8,0.6
train/0002.jpg,2,-0.7,-0.3
...
```

Where:
- `emotion`: Integer label 0-7
- `valence`: Float in [-1, 1] (negative to positive)
- `arousal`: Float in [-1, 1] (calm to excited)

---

## 🔬 Model Architecture

```
Input Image (224×224×3)
        ↓
ResNet-18 Backbone (Pretrained)
        ↓
Shared Features (512-dim)
        ↓
    ┌───────────┴───────────┐
    ↓                       ↓
Emotion Classifier      VA Regressor
(FC → ReLU → FC)        (FC → ReLU → FC → Tanh)
    ↓                       ↓
8 Class Logits          [Valence, Arousal]
```

---

## 🎯 Performance Targets

### Real-Time Inference
- **CPU (Intel i5+)**: 15-20 FPS with PyTorch
- **CPU with ONNX**: 25-30 FPS
- **GPU (CUDA)**: 60+ FPS

### Model Size
- **ResNet-18**: ~45 MB
- **MobileNetV2**: ~15 MB

### Accuracy (Expected on AffectNet)
- **Classification Accuracy**: 60-65%
- **Macro F1**: 0.55-0.60
- **Mean CCC**: 0.50-0.60

---

## 🐛 Troubleshooting

### Issue: "No face detected"
**Solution**: Adjust MediaPipe confidence threshold
```python
detector = MediaPipeFaceDetector(min_detection_confidence=0.3)
```

### Issue: Low FPS on CPU
**Solution**: Use ONNX Runtime
```bash
python deployment/main.py --model model.onnx --onnx
```

### Issue: CUDA out of memory during training
**Solution**: Reduce batch size
```bash
python training/train.py --batch_size 32
```

### Issue: Model not loading
**Solution**: Ensure model and config paths are correct
```python
recognizer = create_recognizer(
    model_path='models/model_weights.pth',
    config_path='models/model_config.json'
)
```

---

## 📚 References

1. **AffectNet**: Mollahosseini, A., et al. "AffectNet: A database for facial expression, valence, and arousal computing in the wild." IEEE TAC (2017).

2. **ResNet**: He, K., et al. "Deep residual learning for image recognition." CVPR (2016).

3. **MediaPipe**: Lugaresi, C., et al. "MediaPipe: A framework for building perception pipelines." arXiv (2019).

---

## 📄 License

This project is provided for educational and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## 🙏 Acknowledgments

- AffectNet dataset creators
- PyTorch and MediaPipe teams
- Open-source community

---

**Built with ❤️ for emotion AI research**
