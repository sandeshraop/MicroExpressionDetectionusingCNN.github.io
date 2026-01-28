# Deterministic Micro-Expression Recognition System

A deterministic micro-expression recognition system using the CASME-II dataset that validates the correctness of the entire ML pipeline by achieving 100% accuracy on training data.

## 🎯 Project Goal

**Primary Objective**: Train a model using CASME-II such that when the SAME CASME-II frames are given as input, the model predicts the EXACT SAME micro-emotion label with 100% accuracy.

This proves:
- ✅ Correct data loading
- ✅ Correct label mapping  
- ✅ Correct feature extraction
- ✅ Correct model training

## ⚠️ Important Note on Training Method

**This project trains and tests on the SAME data intentionally.**

**Why?**
- To validate the entire ML pipeline correctness
- To prove 100% memorization capability
- To eliminate all hidden bugs before generalization experiments
- To establish a deterministic baseline

**This is NOT:**
- A generalization study ❌
- A SOTA comparison ❌  
- A real-world deployment system ❌

**For generalization experiments**, use LOSO (Leave-One-Subject-Out) evaluation after this baseline is established.

## 🚫 What This Project Is NOT

This project does NOT aim to:
- Generalize to unseen subjects ❌
- Perform LOSO evaluation ❌
- Achieve SOTA results ❌
- Handle real-world noisy inputs ❌
- Use complex deep learning architectures ❌

Those come after correctness is proven.

## 📁 Project Structure

```
microexpression_project/
│
├── data/
│   └── casme2/                    # CASME-II dataset
│
├── src/
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── features.py                # Deterministic feature extraction
│   ├── model.py                   # Simple neural network model
│   ├── train.py                   # Training script
│   └── predict.py                 # Inference script
│
├── models/                        # Saved models (auto-created)
├── plots/                         # Training plots (auto-created)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Details

### Dataset Used
- **CASME-II** (Chinese Academy of Sciences Micro-Expression II)

### Emotion Classes
- Happiness
- Disgust  
- Repression
- Surprise

### Input Specification
- **Frame Selection**: Exactly 10 frames per sequence
- **Frame Indices**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (deterministic)
- **Color**: Grayscale
- **Resolution**: 64 × 64
- **Normalization**: pixel / 255.0

## 🔧 Feature Extraction

### Deterministic Motion Features
For each frame sequence:
1. Let `F₀` = first frame
2. Let `Fᵢ` = frame i (i = 1 to 9)
3. Compute: `feature[i] = mean(|Fᵢ − F₀|)`
4. Final feature vector shape: `(9,)`

This guarantees:
- Same input → same features
- No dependency on detectors
- No randomness

## 🧠 Model Architecture

**Intentionally Simple Model**:
```
Input (9) → Linear(32) → ReLU → Linear(4)
```

**Why this model?**
- Can easily memorize training data
- Easy to debug
- No overfitting prevention (INTENTIONAL)

## 🏋️ Training Strategy

### Deterministic Training Rules
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: Until training accuracy = 100%
- **Shuffle**: False
- **Batch size**: All samples

### What's NOT Used
- ❌ No validation
- ❌ No early stopping
- ❌ No class weights
- ❌ No focal loss
- ❌ No dropout

## 🎯 Success Criteria

The project is **SUCCESSFUL** if:
- ✅ Training accuracy = 100%
- ✅ Same input sequence → same predicted label
- ✅ Prediction is stable across runs
- ✅ Confusion matrix = perfect diagonal

## 🚀 Usage

### Training
```bash
cd src
python train.py
```

This will:
1. Load the CASME-II dataset
2. Extract deterministic features
3. Train until 100% accuracy
4. Evaluate and test determinism
5. Save the model and plots

### Prediction
```bash
cd src
python predict.py --demo
```

### Programmatic Usage
```python
from src.dataset import CASMEIIDataset
from src.features import extract_deterministic_features
from src.model import create_model
from src.predict import MERPredictor

# Load dataset
dataset = CASMEIIDataset("../data/casme2")
frames, label, metadata = dataset[0]

# Extract features
features = extract_deterministic_features(frames)

# Create and load model
model = create_model()
predictor = MERPredictor(model)

# Make prediction
result = predictor.predict_from_frames(frames)
print(f"Predicted emotion: {result['predicted_emotion']}")
```

## 📈 Expected Output

When training is successful, you should see:

```
✅ Training accuracy = 100%
✅ Same input → same predicted label  
✅ Confusion matrix = perfect diagonal
🎉 PROJECT SUCCESSFUL! Pipeline is correct.
```

## 🔍 Debugging Features

The system includes extensive debugging and validation:

1. **Deterministic Testing**: Verifies same input produces same output
2. **Feature Validation**: Checks feature ranges and validity
3. **Confusion Matrix**: Perfect diagonal indicates memorization
4. **Training Plots**: Visual confirmation of 100% accuracy
5. **Reproducibility**: Fixed random seeds ensure consistent results

## 🧪 Testing

Run individual component tests:

```bash
# Test dataset loading
python dataset.py

# Test feature extraction  
python features.py

# Test model
python model.py

# Test prediction
python predict.py --demo
```

## 📝 Key Design Decisions

### Why Deterministic?
- Eliminates hidden bugs in data pipeline
- Validates entire ML workflow
- Builds confidence before complex models
- Academically defensible approach

### Why Simple Model?
- Easy to debug and understand
- Can memorize training data (required for this project)
- No overfitting concerns when training = testing
- Fast training for rapid iteration

### Why Fixed Frame Selection?
- Removes dependency on apex detection algorithms
- Ensures reproducibility
- Eliminates variability from onset/offset detection
- Focus on pipeline correctness, not optimization

## 🔮 What Comes Next (Not Today)

Only after this succeeds:
- LOSO (Leave-One-Subject-Out) evaluation
- Optical flow features
- Apex detection algorithms
- CNN / GCN architectures  
- Generalization experiments
- Real-world deployment

## 🐛 Troubleshooting

### Training Accuracy < 100%
- Check data loading consistency
- Verify feature extraction determinism
- Ensure no data shuffling
- Check model initialization

### Non-Deterministic Predictions
- Verify random seeds are set
- Check for any randomness in feature extraction
- Ensure model is in eval mode during inference
- Verify no dropout/batchnorm layers

### Data Loading Issues
- Ensure CASME-II dataset structure is correct
- Check file paths and permissions
- Verify frame preprocessing pipeline

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

This is a reference implementation. For research extensions, please ensure you understand the deterministic validation approach before adding complexity.
