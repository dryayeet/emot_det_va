# Setup Guide - Multi-Task Facial Emotion Recognition

Complete step-by-step guide from training to deployment.

---

## Part 1: Training on Google Colab

### Step 1: Setup Kaggle API

1. **Create Kaggle Account**
   - Go to https://www.kaggle.com
   - Sign up or log in

2. **Generate API Token**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

3. **Upload to Google Drive**
   - Upload `kaggle.json` to your Google Drive
   - Remember the path (e.g., `/content/drive/MyDrive/kaggle.json`)

### Step 2: Prepare Colab Notebook

1. **Open Google Colab**
   - Go to https://colab.research.google.com

2. **Create New Notebook**
   - Click "New Notebook"
   - Name it "Emotion_Recognition_Training"

3. **Set Runtime**
   - Runtime → Change runtime type → GPU (T4 or better)
   - Save

### Step 3: Copy Training Code

Open `colab_training_notebook.py` from the project folder. The file contains 21 cells marked with comments like:

```python
# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================
```

Copy each cell section to a separate Colab cell in order.

### Step 4: Run Training

Execute cells in sequence:

**Cell 1** - Install dependencies (takes ~2 minutes)
```
!nvidia-smi
!pip install kaggle torch torchvision ...
```

**Cell 2** - Mount Google Drive
- Authorize access when prompted

**Cell 3** - Download dataset (takes ~10-15 minutes)
- Downloads AffectNet from Kaggle
- ~2-3 GB download

**Cell 4-10** - Setup code
- Imports libraries
- Defines dataset, model, losses
- Quick execution

**Cell 11-12** - Training loop (takes ~2-4 hours for 50 epochs)
- Monitor training progress
- Early stopping may trigger earlier

**Cell 13-15** - Evaluation and visualization
- Plots training curves
- Confusion matrix
- Test set evaluation

**Cell 16-19** - Model export (takes ~5 minutes)
- Exports to .pth, .pt, .onnx, .h5
- Saves to Google Drive

**Cell 20-21** - Final configuration and download prep
- Creates model config
- Zips all files

### Step 5: Download Trained Models

After training completes, download these files from Google Drive:

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

1. **Clone/Download Project**
```bash
cd /path/to/projects
# Extract or clone the project
cd emotion_recognition_project
```

2. **Create Virtual Environment**
```bash
# Create venv
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: If you get errors:
- **Windows**: Install Visual C++ Build Tools for some packages
- **macOS**: May need `brew install cmake` for dlib
- **Linux**: May need `sudo apt-get install python3-dev`

### Step 2: Organize Model Files

Create a `models` directory and place downloaded files:

```
emotion_recognition_project/
├── models/
│   ├── model_weights.pth          # From Colab
│   ├── model_config.json          # From Colab
│   └── model.onnx                 # Optional, from Colab
├── training/
├── deployment/
└── ...
```

### Step 3: Test Installation

```bash
# Test imports
python -c "import torch; import cv2; import mediapipe; print('✓ All imports successful')"
```

If successful, proceed to webcam test.

### Step 4: Run Webcam Application

**Option A: PyTorch Model**
```bash
python deployment/main.py --model models/model_weights.pth --config models/model_config.json --device cpu
```

**Option B: ONNX Model (Faster)**
```bash
python deployment/main.py --model models/model.onnx --onnx --device cpu
```

**With GPU (if available):**
```bash
python deployment/main.py --model models/model_weights.pth --device cuda
```

### Step 5: Webcam Controls

Once running:
- **Look at camera** - System will detect face and predict emotion
- **Press 'q'** - Quit
- **Press 'v'** - Toggle Valence-Arousal visualization
- **Press 'p'** - Toggle probability bars
- **Press 's'** - Save screenshot

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
    
    # Print results
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

# Get all images
image_paths = glob.glob('images/*.jpg')

# Process each
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

## Common Issues & Solutions

### Issue 1: "No module named 'training'"

**Solution**: Add project root to Python path
```python
import sys
sys.path.insert(0, '/path/to/emotion_recognition_project')
```

Or run from project root:
```bash
cd emotion_recognition_project
python deployment/main.py --model models/model_weights.pth
```

### Issue 2: "Failed to open camera 0"

**Solutions**:
```bash
# Try different camera ID
python deployment/main.py --model models/model_weights.pth --camera 1

# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Issue 3: Low FPS (< 10 FPS)

**Solutions**:
1. Use ONNX model: `--onnx`
2. Reduce camera resolution in code
3. Use MobileNetV2 instead of ResNet-18

### Issue 4: MediaPipe not detecting faces

**Solution**: Lower detection threshold
```python
detector = MediaPipeFaceDetector(min_detection_confidence=0.3)
```

### Issue 5: "CUDA out of memory" during training

**Solution**: Reduce batch size
```python
BATCH_SIZE = 32  # In Cell 7 of Colab notebook
```

### Issue 6: Package installation errors

**Windows**: Install Visual C++ Build Tools
- Download from: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"

**macOS**: Install Xcode Command Line Tools
```bash
xcode-select --install
```

**Linux**: Install build essentials
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

---

## Performance Optimization

### For Faster CPU Inference:

1. **Use ONNX Runtime**
```bash
pip install onnxruntime
python deployment/main.py --model model.onnx --onnx
```

2. **Use MobileNetV2**
- Train with `--architecture mobilenetv2`
- ~3x faster than ResNet-18

3. **Reduce Input Resolution**
- Modify preprocessor target_size to (112, 112)
- Trade-off: slight accuracy decrease

### For Better Accuracy:

1. **Use ResNet-50**
- Modify model.py to use ResNet-50
- More parameters = better accuracy

2. **Longer Training**
- Increase epochs: `--epochs 100`
- Better convergence

3. **Data Augmentation**
- Add more augmentations in utils.py
- Reduce overfitting

---

## Next Steps

### Customize for Your Use Case:

1. **Fine-tune on your data**
   - Collect custom emotion dataset
   - Resume training from pretrained weights

2. **Add more emotions**
   - Modify num_classes in model
   - Update EMOTION_LABELS list

3. **Deploy to mobile**
   - Convert to TensorFlow Lite
   - Use with Android/iOS

4. **Create web application**
   - Use FastAPI + JavaScript
   - Stream predictions over WebSocket

---

## Support

If you encounter issues:
1. Check this guide thoroughly
2. Review README.md
3. Check code comments
4. Open GitHub issue with:
   - Error message
   - Python version
   - OS and GPU info
   - Steps to reproduce

---

**Happy Coding! 🎉**
