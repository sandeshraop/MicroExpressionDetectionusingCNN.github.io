# MicroExpression Detection using CNN

A comprehensive micro-expression recognition system built with Convolutional Neural Networks (CNNs) and advanced computer vision techniques.

## 🚀 Features

- **Real-time micro-expression detection** using deep learning models
- **Web-based interface** for easy interaction and visualization
- **Advanced preprocessing pipeline** with optical flow analysis
- **Multiple evaluation protocols** including LOSO (Leave-One-Subject-Out)
- **Comprehensive documentation** and deployment guides

## 📁 Project Structure

```
microexpression_project/
├── data/                   # Dataset directories (excluded from git)
│   ├── casme2/            # CASME2 dataset
│   ├── labels/            # Label files and metadata
│   └── predict/           # Prediction data
├── deployment/            # Docker deployment files
├── docs/                  # Documentation and reports
├── inference/             # Model inference pipelines
├── models/                # Trained models and metadata
├── scripts/               # Training and evaluation scripts
├── src/                   # Core source code
├── web/                   # Web application
└── visualizations/        # Analysis plots and charts
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA (for GPU acceleration, optional)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/sandeshraop/MicroExpressionDetectionusingCNN.github.io.git
cd MicroExpressionDetectionusingCNN.github.io
```

2. Install dependencies:
```bash
pip install -r microexpression_project/web/requirements.txt
pip install -r microexpression_project/scripts/requirements.txt
```

## 🌐 Web Application

Launch the web interface:
```bash
cd microexpression_project/web
python run.py
```

The application will be available at `http://localhost:5000`

## 📊 Model Performance

Our CNN-based micro-expression detection system achieves:
- High accuracy on CASME2 dataset
- Robust performance across different subjects
- Real-time processing capabilities

## 📚 Documentation

- [Main Documentation](microexpression_project/docs/main_README.md)
- [Web App Guide](microexpression_project/docs/web_README.md)
- [Deployment Guide](microexpression_project/deployment/README.md)
- [Inference Pipeline](microexpression_project/inference/README.md)

## 🔬 Research

This project implements state-of-the-art techniques in:
- Micro-expression recognition
- Optical flow analysis
- Temporal feature extraction
- Cross-subject validation

## 📈 Evaluation

The system includes comprehensive evaluation protocols:
- LOSO (Leave-One-Subject-Out) validation
- Confusion matrix analysis
- Temporal dynamics analysis
- Feature importance visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- CASME2 dataset providers
- Open-source computer vision community
- Deep learning research community

---

**Note**: Large dataset files are excluded from this repository for size constraints. Please download the CASME2 dataset separately and place it in the `data/casme2/` directory.
