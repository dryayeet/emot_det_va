# Setup Guide - Multi-Task Facial Emotion Recognition

Step-by-step guide from training to deployment.

---

## Part 1: Training on Google Colab

### Step 1: Setup Kaggle API

1. **Create Kaggle Account**
   - Navigate to https://www.kaggle.com
   - Sign up or log in

2. **Generate API Token**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Upload to Google Drive**
   - Upload `kaggle.json` to your Google Drive
   - Note the path (e.g., `/content/drive/MyDrive/kaggle.json`)

### Step 2: Prepare Colab Notebook

1. **Open Google Colab**
   - Navigate to https://colab.research.google.com

2. **Create New Notebook**
   - Click "New Notebook"
   - Name it appropriately (e.g., "Emotion_Recognition_Training")

3. **Set Runtime**
   - Runtime > Change runtime type > GPU (T4 or better)
   - Save

### Step 3: Copy Training Code

Open `colab_training_notebook.py` from the project folder. The file contains 21 cells marked with comments:

```python
# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================
```

Copy each cell section to a separate Colab cell in order.

### Step 4: Run Training

Execute cells in sequence:

| Cell | Description | Duration |
|------|-------------|----------|
| 1 | Install dependencies | ~2 minutes |
| 2 | Mount Google Drive | Requires authorization |
| 3 | Download dataset | ~10-15 minutes |
| 4-10 | Setup code (imports, model, losses) | Quick |
| 11-12 | Training loop | ~2-4 hours |
| 13-15 | Evaluation and visualization | ~5 minutes |
| 16-19 | Model export | ~5 minutes |
| 20-21 | Configuration and packaging | Quick |

### Step 5: Download Trained Models

After training completes, download from Google Drive:

**Required Files:**
- `best_model.pth` - Full checkpoint
- `model_weights.pth` - Weights only (recommended for deployment)
- `model_config.json` - Model configuration

**Optional Export Formats:**
- `model_torchscript.pt` - TorchScript format
- `model.onnx` - ONNX format (recommended for CPU inference)
- `model.h5` - Keras format

**Visualizations:**
- `training_history.png` - Training curves
- `confusion_matrix.png` - Test set confusion matrix

---

## Part 2: Local Deployment

### Step 1: Setup Local Environment

1. **Navigate to Project Directory**
```bash
cd /path/to/emot_recog
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Platform-Specific Notes:**
- Windows: May require Visual C++ Build Tools
- macOS: May need `brew install cmake` for dlib
- Linux: May need `sudo apt-get install python3-dev`

### Step 2: Organize Model Files

Create a `models` directory and place downloaded files:

```
emot_recog/
├── models/
│   ├── model_weights.pth      # From Colab
│   ├── model_config.json      # From Colab
│   └── model.onnx             # Optional
├── training/
├── deployment/
└── ...
```

### Step 3: Test Installation

```bash
python -c "import torch; import cv2; import mediapipe; print('All imports successful')"
```

### Step 4: Run Webcam Application

**PyTorch Model:**
```bash
python deployment/main.py --model models/model_weights.pth --config models/model_config.json --device cpu
```

**ONNX Model (Faster):**
```bash
python deployment/main.py --model models/model.onnx --onnx --device cpu
```

**With GPU:**
```bash
python deployment/main.py --model models/model_weights.pth --device cuda
```

### Step 5: Webcam Controls

| Key | Action |
|-----|--------|
| q | Quit |
| v | Toggle Valence-Arousal visualization |
| p | Toggle probability bars |
| s | Save screenshot |

---

## Part 3: Programmatic Usage

### Basic Usage

```python
from deployment.face_detector import MediaPipeFaceDetector
from deployment.preprocess import EmotionPreprocessor
from deployment.inference import create_recognizer
import cv2

# Initialize
detector = MediaPipeFaceDetector()
preprocessor = EmotionPreprocessor()
recognizer = create_recognizer('models/model_weights.pth', device='cpu')

# Load image
image = cv2.imread('test_image.jpg')

# Detect face
bbox, confidence = detector.detect_single_face(image)

if bbox:
    # Crop face
    face_crop = detector.crop_face(image, bbox, padding=0.2)
    
    # Preprocess
    input_tensor = preprocessor.preprocess_cv2(face_crop)
    
    # Predict
    results = recognizer.predict(input_tensor)
    
    # Output results
    print(f"Emotion: {results['emotion']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Valence: {results['valence']:.2f}")
    print(f"Arousal: {results['arousal']:.2f}")
else:
    print("No face detected")
```

### Batch Processing

```python
import glob

image_paths = glob.glob('images/*.jpg')

for img_path in image_paths:
    image = cv2.imread(img_path)
    bbox, _ = detector.detect_single_face(image)
    
    if bbox:
        face_crop = detector.crop_face(image, bbox)
        input_tensor = preprocessor.preprocess_cv2(face_crop)
        results = recognizer.predict(input_tensor)
        
        print(f"{img_path}: {results['emotion']} ({results['confidence']:.2%})")
```

### Video File Processing

```python
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    bbox, _ = detector.detect_single_face(frame)
    
    if bbox:
        face_crop = detector.crop_face(frame, bbox)
        input_tensor = preprocessor.preprocess_cv2(face_crop)
        results = recognizer.predict(input_tensor)
        
        # Draw on frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, results['emotion'], (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Troubleshooting

### Issue: "No module named 'training'"

**Solution**: Add project root to Python path or run from project root:
```bash
cd emot_recog
python deployment/main.py --model models/model_weights.pth
```

### Issue: "Failed to open camera 0"

**Solution**: Try different camera ID:
```bash
python deployment/main.py --model models/model_weights.pth --camera 1
```

Check available cameras:
```bash
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Issue: Low FPS (< 10 FPS)

**Solutions**:
1. Use ONNX model with `--onnx` flag
2. Reduce camera resolution in code
3. Use MobileNetV2 instead of ResNet-18

### Issue: MediaPipe not detecting faces

**Solution**: Lower detection threshold:
```python
detector = MediaPipeFaceDetector(min_detection_confidence=0.3)
```

### Issue: "CUDA out of memory" during training

**Solution**: Reduce batch size in Cell 7 of Colab notebook:
```python
BATCH_SIZE = 32
```

### Issue: Package installation errors

**Windows**: Install Visual C++ Build Tools from https://visualstudio.microsoft.com/downloads/

**macOS**: Install Xcode Command Line Tools:
```bash
xcode-select --install
```

**Linux**: Install build essentials:
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

---

## Performance Optimization

### For Faster CPU Inference

1. **Use ONNX Runtime**
```bash
pip install onnxruntime
python deployment/main.py --model model.onnx --onnx
```

2. **Use MobileNetV2**
   - Train with `--architecture mobilenetv2`
   - Approximately 3x faster than ResNet-18

3. **Reduce Input Resolution**
   - Modify preprocessor target_size to (112, 112)
   - Trade-off: slight accuracy decrease

### For Better Accuracy

1. **Use ResNet-50**
   - Modify model.py to use ResNet-50
   - More parameters, better accuracy

2. **Longer Training**
   - Increase epochs: `--epochs 100`

3. **Additional Data Augmentation**
   - Add more augmentations in utils.py

---

## Next Steps

### Customization Options

1. **Fine-tune on custom data**
   - Collect custom emotion dataset
   - Resume training from pretrained weights

2. **Add more emotions**
   - Modify num_classes in model
   - Update EMOTION_LABELS list

3. **Deploy to mobile**
   - Convert to TensorFlow Lite
   - Use with Android/iOS

4. **Create web application**
   - Use FastAPI with WebSocket
   - Stream predictions in real-time

---

## Support

For issues:
1. Review this guide
2. Check README.md
3. Review code comments
4. Open GitHub issue with:
   - Error message
   - Python version
   - OS and GPU information
   - Steps to reproduce
