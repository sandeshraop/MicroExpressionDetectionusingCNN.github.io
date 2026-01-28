# Micro-Expression Recognition Web Interface

A comprehensive web application for demonstrating and interacting with the micro-expression recognition system.

## Features

### 🎯 Core Functionality
- **Interactive Demo**: Upload videos for real-time emotion analysis
- **Performance Visualization**: Comprehensive charts and graphs
- **System Architecture**: Detailed pipeline visualization
- **Documentation**: Integrated technical documentation
- **Responsive Design**: Works on desktop and mobile devices

### 🎨 User Interface
- **Modern Design**: Clean, professional interface with Bootstrap 5
- **Smooth Animations**: Engaging user experience with CSS animations
- **Interactive Charts**: Real-time data visualization with Chart.js
- **Drag & Drop**: Intuitive file upload interface
- **Progress Indicators**: Real-time processing feedback

### 🔧 Technical Features
- **RESTful API**: Flask-based backend for video processing
- **Real-time Analysis**: Video frame extraction and optical flow computation
- **Model Integration**: Direct integration with trained CNN-SVM model
- **Error Handling**: Comprehensive error management and user feedback
- **Performance Monitoring**: Built-in performance metrics and logging

## Quick Start

### Prerequisites
- Python 3.8+
- Flask and dependencies
- Trained model files
- Modern web browser

### Installation

1. **Install Dependencies**:
```bash
pip install flask flask-cors opencv-python numpy
```

2. **Start the Backend API**:
```bash
cd web/api
python backend.py
```

3. **Open the Web Interface**:
Open `index.html` in your web browser or serve it with a web server:
```bash
cd web
python -m http.server 8000
```
Then visit `http://localhost:8000`

## API Endpoints

### 📡 Available Endpoints

#### `GET /api/health`
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2026-01-27T19:30:00",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### `POST /api/upload`
Upload and analyze video
- **Content-Type**: `multipart/form-data`
- **Body**: `video` file (max 100MB)
- **Response**: Analysis results with emotion predictions

#### `GET /api/model/info`
Get model information
```json
{
  "model_type": "Enhanced Hybrid CNN-SVM",
  "input_shape": "3x64x64 RGB frames + 6x64x64 optical flow",
  "output_classes": ["Happiness", "Surprise", "Disgust", "Repression"],
  "performance": {
    "accuracy": 46.3,
    "uar": 24.8,
    "happiness_recall": 71.6,
    "disgust_recall": 27.4
  }
}
```

#### `GET /api/results/sample`
Get sample analysis results for demonstration

#### `GET /api/visualizations/<filename>`
Serve visualization images

## File Structure

```
web/
├── index.html              # Main web interface
├── css/
│   └── style.css           # Custom styles
├── js/
│   └── main.js             # Frontend JavaScript
├── api/
│   └── backend.py          # Flask API server
├── README.md               # This file
└── assets/                 # Static assets (images, etc.)
```

## Usage Guide

### 🎥 Video Upload Demo

1. **Navigate to Demo Section**: Click "Try Demo" or scroll to demo section
2. **Upload Video**: 
   - Drag and drop video file onto upload area
   - Or click "Choose File" to browse
3. **Processing**: System processes video and extracts frames
4. **Results**: View emotion predictions and confidence scores
5. **Analysis**: Detailed emotion probability breakdown

### 📊 Performance Visualization

- **Confusion Matrix**: Classification results analysis
- **Performance Charts**: Subject-wise and temporal analysis
- **Feature Analysis**: AU-specific feature importance
- **System Architecture**: Complete pipeline visualization

### 📚 Documentation Access

- **Technical Documentation**: Link to comprehensive PDF
- **Model Information**: Detailed architecture and performance
- **API Documentation**: Complete endpoint reference

## Configuration

### Backend Configuration

Edit `api/backend.py` to modify:

```python
# File size limit (bytes)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Model path
model_paths = [
    '../models/augmented_model_temporal_au_specific_20260127_182653.pkl',
    '../models/augmented_model.pkl'
]
```

### Frontend Configuration

Edit `js/main.js` to modify:

```javascript
// Animation thresholds
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

// Chart configurations
const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        r: {
            beginAtZero: true,
            max: 100
        }
    }
};
```

## Development

### 🔧 Local Development

1. **Frontend Development**:
```bash
# Serve frontend with live reload
cd web
python -m http.server 8000
```

2. **Backend Development**:
```bash
# Run API server in debug mode
cd web/api
python backend.py
```

3. **Full Stack Development**:
```bash
# Terminal 1: Backend
cd web/api
python backend.py

# Terminal 2: Frontend
cd web
python -m http.server 8000
```

### 🐳 Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api/backend.py"]
```

Build and run:
```bash
docker build -t micro-expression-web .
docker run -p 5000:5000 micro-expression-web
```

### 🔒 Security Considerations

- **File Upload**: Validate file types and sizes
- **CORS**: Configured for development, restrict in production
- **Rate Limiting**: Implement for production use
- **Authentication**: Add user authentication for production

## Performance Optimization

### 📈 Frontend Optimization

- **Lazy Loading**: Images and charts load on scroll
- **Caching**: Browser caching for static assets
- **Compression**: Gzip compression for API responses
- **CDN**: Use CDN for Bootstrap and other libraries

### ⚡ Backend Optimization

- **Model Loading**: Load model once at startup
- **Frame Processing**: Optimize frame extraction
- **Memory Management**: Clean up temporary files
- **Async Processing**: Use background tasks for long videos

## Troubleshooting

### Common Issues

1. **Model Not Loading**:
   - Check model file paths in `backend.py`
   - Ensure all dependencies are installed
   - Verify model file permissions

2. **Video Upload Fails**:
   - Check file size limit (100MB max)
   - Verify file format is supported
   - Check browser console for errors

3. **Analysis Takes Too Long**:
   - Reduce video resolution
   - Limit number of frames processed
   - Check system resources

4. **Charts Not Displaying**:
   - Ensure Chart.js is loaded
   - Check browser console for JavaScript errors
   - Verify data format is correct

### Debug Mode

Enable debug mode in `backend.py`:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

Check browser console for frontend errors:
```javascript
// Open developer tools (F12)
// Check Console tab for errors
```

## Contributing

### 🤝 Contributing Guidelines

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### 📝 Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Use ES6+ features
- **CSS**: Use BEM methodology
- **HTML**: Semantic HTML5

### 🧪 Testing

```bash
# Run backend tests
python -m pytest tests/

# Run frontend tests
npm test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- 📧 Email: support@microexpression.ai
- 🐛 Issues: GitHub Issues
- 📖 Documentation: Full technical documentation available

---

**Micro-Expression Recognition Web Interface** - Advanced AI-powered emotion analysis system
