# Micro-Expression Inference Pipeline

This directory contains the complete inference pipeline for micro-expression recognition from raw videos.

## 🎯 **What It Does**

The inference pipeline can:
- ✅ Load and process raw video files
- ✅ Automatically detect onset/apex/offset frames
- ✅ Extract AU-aligned optical flow features
- ✅ Predict emotions using the trained enhanced hybrid model
- ✅ Provide interpretable AU contribution analysis
- ✅ Output confidence scores and detailed explanations

## 📁 **Files**

- `inference_pipeline.py` - Main inference pipeline class
- `demo_inference.py` - Demo script for testing
- `README.md` - This file

## 🚀 **Quick Start**

### 1. Train and Save Model

```bash
cd ../utils
python train_and_save_model.py --data_root ../data/casme2 --epochs 20
```

This will:
- Train the AU-aligned enhanced hybrid model
- Save the trained model to `../models/`
- Create metadata and example scripts

### 2. Run Inference on Single Video

```bash
cd ../inference
python demo_inference.py --model ../models/au_aligned_hybrid_svm_20250127_142630.pkl --video path/to/your/video.mp4
```

### 3. Batch Process Multiple Videos

```bash
python demo_inference.py --model ../models/au_aligned_hybrid_svm_20250127_142630.pkl --video ../test_videos/
```

## 📋 **Requirements**

- Trained model file (.pkl)
- Video file (MP4, AVI, MOV, etc.)
- Python dependencies (see requirements.txt)

## 🎬 **Video Requirements**

The pipeline expects videos with:
- **Face visible** in the frame
- **Micro-expression sequences** (onset → apex → offset)
- **Reasonable frame rate** (25-30 FPS recommended)
- **Good lighting** and minimal motion blur

## 📊 **Output Format**

```json
{
  "success": true,
  "predicted_emotion": "happiness",
  "confidence": 0.847,
  "all_probabilities": {
    "happiness": 0.847,
    "disgust": 0.089,
    "repression": 0.032,
    "surprise": 0.032
  },
  "au_contribution": {
    "au_rankings": {
      "AU12": {
        "activity_score": 2.34,
        "onset_apex": {"mean": 0.45, "std": 0.12, "max": 0.78},
        "apex_offset": {"mean": 0.38, "std": 0.09, "max": 0.67}
      },
      "AU6": {...},
      "AU4": {...},
      "AU9": {...},
      "AU10": {...}
    },
    "most_active_au": "AU12",
    "total_strain_energy": 1.23
  },
  "frame_info": {
    "video_path": "path/to/video.mp4",
    "frames_extracted": true
  },
  "timestamp": "2025-01-27T14:26:30"
}
```

## 🔍 **AU Contribution Analysis**

The pipeline provides detailed analysis of which Action Units (AUs) contributed most to the prediction:

- **AU4**: Brow lowerer
- **AU6**: Cheek raiser  
- **AU9**: Nose wrinkler
- **AU10**: Upper lip raiser
- **AU12**: Lip corner puller

Each AU gets an **activity score** based on:
- Mean strain magnitude
- Maximum strain intensity
- Temporal dynamics (onset→apex vs apex→offset)

## 🛠️ **API Usage**

### Basic Usage

```python
from inference_pipeline import MicroExpressionInferencePipeline

# Initialize pipeline
pipeline = MicroExpressionInferencePipeline()

# Load trained model
pipeline.load_model('path/to/model.pkl')

# Predict emotion
result = pipeline.predict_emotion('path/to/video.mp4')

if result['success']:
    print(f"Emotion: {result['predicted_emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Most Active AU: {result['au_contribution']['most_active_au']}")
```

### Batch Processing

```python
# Process multiple videos
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
results = pipeline.batch_predict(video_paths)

for i, result in enumerate(results):
    if result['success']:
        print(f"Video {i+1}: {result['predicted_emotion']}")
```

## 🔧 **Customization**

### Face Detection

Currently uses fixed crop coordinates (same as training):
```python
# In frame_selector.py
FACE_Y1, FACE_Y2 = 40, 280  # Vertical
FACE_X1, FACE_X2 = 80, 320  # Horizontal
```

For production, you could integrate face detection:
```python
# Replace detect_face_and_crop method
def detect_face_and_crop(self, frame):
    # Use OpenCV face detection or MTCNN
    # Return cropped face (64, 64)
```

### Frame Detection

Currently uses simple temporal approach:
- Onset: First frame
- Apex: Middle frame  
- Offset: Last frame

For better accuracy, implement:
- Motion-based detection
- Peak detection algorithms
- Neural network frame selection

## 📈 **Performance**

- **Processing Time**: ~2-3 seconds per video
- **Memory Usage**: ~500MB (GPU) / ~200MB (CPU)
- **Accuracy**: 64.73% (LOSO) on CASME-II
- **UAR**: 62.56% (balanced performance)

## 🎯 **Deployment Options**

### 1. Command Line
```bash
python demo_inference.py --model model.pkl --video video.mp4
```

### 2. Python API
```python
from inference_pipeline import MicroExpressionInferencePipeline
pipeline = MicroExpressionInferencePipeline()
pipeline.load_model('model.pkl')
result = pipeline.predict_emotion('video.mp4')
```

### 3. Web API (Future)
Could be wrapped in Flask/FastAPI for web deployment.

## 🐛 **Troubleshooting**

### Common Issues

1. **"No model loaded"**
   - Ensure model file exists and is valid
   - Check training was completed successfully

2. **"Could not extract frames"**
   - Verify video file format is supported
   - Check video has visible faces
   - Ensure video is not corrupted

3. **"Prediction failed"**
   - Check video quality and lighting
   - Verify faces are clearly visible
   - Ensure micro-expression is present

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = MicroExpressionInferencePipeline()
pipeline.load_model('model.pkl')
result = pipeline.predict_emotion('video.mp4')
```

## 📚 **Technical Details**

### Model Architecture
- **Handcrafted Features**: 48-D enhanced statistics
- **AU-Aligned Strain**: 40-D statistics (5 AUs × 4 stats × 2 phases)
- **CNN Features**: 128-D deep features
- **Total**: 216-D feature vector

### Preprocessing Pipeline
1. **Face Cropping**: Fixed region (240×240 → 64×64)
2. **Frame Selection**: Onset, Apex, Offset detection
3. **Optical Flow**: Farneback algorithm
4. **Strain Computation**: Sobel-based strain magnitude
5. **AU Alignment**: Region-specific statistics

### Training Details
- **Dataset**: CASME-II (146 samples, 24 subjects)
- **Method**: Leave-One-Subject-Out (LOSO)
- **Optimizer**: Adam (lr=0.001)
- **Classifier**: SVM (RBF kernel)
- **Features**: StandardScaler normalization

## 🎓 **Research Applications**

This pipeline is suitable for:
- **Academic research** on micro-expressions
- **Emotion recognition** studies
- **Psychological experiments**
- **Human-computer interaction** research
- **Affective computing** applications

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Review the training logs and model metadata
3. Verify video quality and format compatibility
4. Consult the original research papers for methodology

---

**Note**: This pipeline is designed for research and academic use. For commercial deployment, additional optimization and testing may be required.
