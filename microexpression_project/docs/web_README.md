# Micro-Expression Recognition Web Interface

A modern web interface for real-time micro-expression recognition from video uploads.

## 🌐 **Features**

- **📹 Video Upload**: Drag-and-drop or click to upload videos
- **🎯 Real-time Prediction**: Instant emotion recognition
- **📊 AU Analysis**: Detailed Action Unit contribution analysis
- **📈 Confidence Scores**: Probability distributions for all emotions
- **🎨 Modern UI**: Clean, responsive interface with Tailwind CSS
- **🔄 Model Management**: Load different trained models via web interface

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
cd web
pip install -r requirements.txt
```

### 2. Start the Web Server
```bash
# Auto-load latest model
python run.py

# Or specify model path
python run.py --model ../models/au_aligned_hybrid_svm_20250127_142630.pkl

# Custom port/host
python run.py --port 8080 --host 0.0.0.0
```

### 3. Open Web Interface
Open your browser and go to: `http://localhost:5000`

## 📱 **Web Interface Overview**

### **Main Page**
- **Upload Area**: Drag-and-drop video upload zone
- **Model Status**: Shows if model is loaded
- **Model Configuration**: Load different models via path input

### **Upload Process**
1. **Drag & Drop** video file or click to browse
2. **Automatic Processing**: Frame detection, feature extraction, prediction
3. **Real-time Results**: Emotion prediction with confidence
4. **AU Analysis**: Which Action Units contributed most

### **Results Display**
- **Emotion Badge**: Predicted emotion with color coding
- **Confidence Score**: Prediction confidence percentage
- **Probability Distribution**: All emotion probabilities with visual bars
- **AU Contribution**: Ranked AU activity scores with visual indicators
- **Technical Details**: Video path, timestamp, processing info

## 🎯 **Supported Video Formats**

- **MP4** (recommended)
- **AVI**
- **MOV**
- **MKV**
- **WebM**

**File Size Limit**: 100MB maximum

## 📊 **Prediction Results**

### **Emotion Classes**
- **Happiness**: 😊 Yellow badge
- **Disgust**: 🤢 Green badge  
- **Repression**: 😔 Blue badge
- **Surprise**: 😮 Purple badge

### **AU Analysis**
The system analyzes 5 key Action Units:
- **AU4**: Brow lowerer
- **AU6**: Cheek raiser
- **9**: Nose wrinkler
- **10**: Upper lip raiser
- **12**: Lip corner puller

Each AU gets an **activity score** based on:
- Mean strain magnitude
- Maximum strain intensity
- Temporal dynamics (onset→apex vs apex→offset)

## 🔧 **API Endpoints**

### **POST /upload**
Upload and process video file.
```json
{
  "success": true,
  "result": {
    "predicted_emotion": "happiness",
    "confidence": 0.847,
    "all_probabilities": {...},
    "au_contribution": {...}
  }
}
```

### **POST /load_model**
Load a trained model.
```json
{
  "success": true,
  "message": "Model loaded successfully"
}
```

### **GET /model_info**
Get information about loaded model.
```json
{
  "success": true,
  "info": {
    "model_type": "au_aligned_hybrid_svm",
    "feature_dim": 216,
    "architecture": "Handcrafted(48) + AU-Aligned StrainStats(40) + CNN(128) -> StandardScaler -> SVM"
  }
}
```

### **GET /health**
Health check endpoint.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## 🛠️ **Configuration**

### **Environment Variables**
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `FLASK_PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Debug mode (default: False)

### **File Upload Settings**
- **Max File Size**: 100MB
- **Allowed Extensions**: mp4, avi, mov, mkv, webm
- **Upload Folder**: `uploads/` (auto-cleaned after processing)

## 🔧 **Customization**

### **Adding New Features**
1. Modify `app.py` to add new endpoints
2. Update `templates/index.html` for UI changes
3. Extend `inference_pipeline.py` for new features

### **Styling**
- Uses Tailwind CSS via CDN
- Custom CSS in `<style>` tags
- Responsive design for mobile devices

### **Model Management**
- Models stored in `../models/` directory
- Auto-loads latest model on startup
- Support for manual model loading

## 🚀 **Deployment Options**

### **Development**
```bash
python run.py --debug
```

### **Production**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Docker** (Future)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "run.py"]
```

## 📱 **Video Requirements**

### **Quality Guidelines**
- **Face Visibility**: Face should be clearly visible
- **Lighting**: Good lighting, minimal shadows
- **Motion**: Clear micro-expression sequence
- **Duration**: 2-10 seconds recommended
- **Format**: Standard video formats

### **Processing Pipeline**
1. **Face Detection**: Fixed crop (40-280, 80-320)
2. **Frame Selection**: Onset, Apex, Offset detection
3. **Feature Extraction**: Optical flow + AU-aligned statistics
4. **Prediction**: Enhanced hybrid model inference
5. **Analysis**: AU contribution and confidence scoring

## 🐛 **Troubleshooting**

### **Common Issues**

#### **"No model loaded"**
- Train a model first: `python ../utils/train_and_save_model.py`
- Check model path exists
- Verify model file is valid (.pkl format)

#### **"Processing failed"**
- Check video file format and size
- Ensure face is visible in video
- Verify video quality and lighting
- Check console for detailed error messages

#### **"Upload failed"**
- Check file size (max 100MB)
- Verify file extension is supported
- Ensure network connection is stable

#### **Server Issues**
- Check if port 5000 is available
- Verify dependencies are installed
- Check for permission issues

### **Debug Mode**
```bash
python run.py --debug
```
This enables detailed error logging and auto-reloading.

## 📞 **Performance**

### **Processing Time**
- **Small Videos** (<5MB): 1-2 seconds
- **Medium Videos** (5-50MB): 2-5 seconds
- **Large Videos** (50-100MB): 5-10 seconds

### **Memory Usage**
- **GPU**: ~500MB (if available)
- **CPU**: ~200MB
- **Upload**: Temporary storage only

### **Concurrent Users**
- **Development**: 1-2 users (debug mode)
- **Production**: 5-10 users (with gunicorn)

## 🎯 **Use Cases**

### **Research Applications**
- **Academic Studies**: Analyze micro-expression datasets
- **Psychology Experiments**: Real-time emotion detection
- **Human-Computer Interaction**: Emotion-aware interfaces

### **Demonstration**
- **Classroom Teaching**: Show ML capabilities
- **Conference Presentations**: Live demo of system
- **Proof of Concept**: Validate research results

### **Integration**
- **Web Applications**: Add emotion detection to websites
- **Mobile Apps**: Backend API for mobile applications
- **IoT Devices**: Real-time emotion recognition systems

## 🔐 **Security Considerations**

### **File Upload**
- File size limits enforced
- File type restrictions
- Temporary storage only
- Automatic cleanup after processing

### **Data Privacy**
- Videos processed locally
- No data stored permanently
- No network transmission of video content
- Only results are returned to client

### **Access Control**
- Consider adding authentication for production
- Implement rate limiting for API endpoints
- Add logging for monitoring

## 📚 **Future Enhancements**

### **Short Term**
- **Webcam Integration**: Real-time webcam capture
- **Batch Processing**: Multiple video upload
- **Result History**: Store and retrieve past predictions
- **Export Options**: Download results as JSON/CSV

### **Medium Term**
- **Face Detection**: Replace fixed crop with MTCNN
- **Advanced Analytics**: Temporal analysis
- **User Accounts**: Personal result tracking
- **API Documentation**: Swagger/OpenAPI specification

### **Long Term**
- **Mobile App**: Native mobile application
- **Cloud Processing**: Cloud-based video processing
- **Real-time Streaming**: Live video analysis
- **Database Integration**: Persistent result storage

---

**Note**: This web interface is designed for research and demonstration purposes. For production deployment, consider additional security, scalability, and monitoring features.

**Performance**: The web interface provides the same 64.73% accuracy as the command-line version, with the added convenience of a user-friendly interface for real-world testing and demonstration.
