// Real JavaScript for Micro-Expression Recognition Web Interface
// Connects to actual Flask backend with real model predictions

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeNavigation();
    initializeUpload();
    initializeAnimations();
    initializeCharts();
    initializeScrollEffects();
    checkModelStatus();
});

// Navigation smooth scrolling
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update active state
                    navLinks.forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                }
            }
        });
    });
    
    // Update active nav link on scroll
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section[id]');
        const scrollY = window.pageYOffset;
        
        sections.forEach(section => {
            const sectionHeight = section.offsetHeight;
            const sectionTop = section.offsetTop - 100;
            const sectionId = section.getAttribute('id');
            
            if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + sectionId) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });
}

// File upload functionality with real backend integration
function initializeUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('videoUpload');
    const analysisResults = document.getElementById('analysisResults');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to upload
    uploadArea.addEventListener('click', function(e) {
        if (e.target === uploadArea || e.target.parentElement === uploadArea) {
            fileInput.click();
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Check model status on load
async function checkModelStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.model_loaded) {
            showAlert('Model loaded successfully! Ready for real predictions.', 'success');
            updateModelStatus(true, data.model_info);
        } else {
            showAlert('Model not loaded. Demo mode only.', 'warning');
            updateModelStatus(false, {});
        }
    } catch (error) {
        console.error('Model status check failed:', error);
        showAlert('Cannot connect to backend. Demo mode only.', 'warning');
        updateModelStatus(false, {});
    }
}

// Update model status display
function updateModelStatus(loaded, modelInfo) {
    const modelStatusElements = document.querySelectorAll('[data-model-status]');
    
    modelStatusElements.forEach(element => {
        if (loaded) {
            element.textContent = '✅ Model Ready';
            element.className = 'bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium';
        } else {
            element.textContent = '❌ Model Not Loaded';
            element.className = 'bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium';
        }
    });
}

// Handle file upload with real backend
async function handleFileUpload(file) {
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showAlert('Please upload a video file', 'warning');
        return;
    }
    
    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
        showAlert('File size must be less than 100MB', 'warning');
        return;
    }
    
    selectedFile = file;
    
    // Update upload area to show selected file
    const uploadContent = document.querySelector('.upload-content');
    if (uploadContent) {
        uploadContent.innerHTML = `
            <i class="fas fa-file-video fa-3x mb-3 text-success"></i>
            <h4>Video Selected</h4>
            <p class="mb-2"><strong>${file.name}</strong></p>
            <p class="text-muted mb-3">Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
            <div class="d-flex gap-2 justify-content-center">
                <button class="btn btn-success" onclick="uploadVideo()">
                    <i class="fas fa-play me-2"></i>Analyze Video
                </button>
                <button class="btn btn-outline-secondary" onclick="resetUpload()">
                    <i class="fas fa-times me-2"></i>Cancel
                </button>
            </div>
        `;
    }
}

// Upload and analyze video with real model predictions
async function uploadVideo() {
    if (!selectedFile) {
        showAlert('Please select a video file first', 'warning');
        return;
    }

    // Check if model is loaded
    try {
        const healthResponse = await fetch('/api/health');
        const healthData = await healthResponse.json();
        
        if (!healthData.model_loaded) {
            showAlert('Model not loaded. Please train the model first.', 'warning');
            return;
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showAlert('Cannot connect to backend. Please check server status.', 'danger');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Show processing state
    const uploadContent = document.querySelector('.upload-content');
    if (uploadContent) {
        uploadContent.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <h4>Processing Video...</h4>
                <p class="text-muted">Analyzing with CNN-SVM model</p>
                <div class="progress mt-3">
                    <div class="progress-bar" id="uploadProgressBar" role="progressbar" style="width: 0%"></div>
                </div>
            </div>
        `;
    }
    
    processingStartTime = Date.now();
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Update progress
            const progressBar = document.getElementById('uploadProgressBar');
            if (progressBar) {
                progressBar.style.width = '100%';
                progressBar.className = 'progress-bar bg-success';
            }
            
            // Show results
            displayRealResults(result);
            
            setTimeout(() => {
                const uploadArea = document.getElementById('uploadArea');
                const analysisResults = document.getElementById('analysisResults');
                
                if (uploadArea) uploadArea.classList.add('d-none');
                if (analysisResults) analysisResults.classList.remove('d-none');
            }, 1000);
            
            showAlert(`Analysis complete! Predicted: ${result.prediction}`, 'success');
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error processing video: ' + error.message, 'danger');
        resetUpload();
    }
}

function updateProcessingStatus(status, progress) {
    document.getElementById('processingStep').textContent = status;
    document.getElementById('progressBar').style.width = progress + '%';
}

function updateStepStatus(elementId, status) {
    document.getElementById(elementId).textContent = status;
}

// Display real results from model
function displayRealResults(result) {
    const processingTime = ((Date.now() - processingStartTime) / 1000).toFixed(2);
    
    // Debug logging
    console.log('🎯 Received result:', result);
    console.log('📊 Prediction:', result.prediction);
    console.log('📈 Confidence:', result.confidence);
    console.log('🔢 All probabilities:', result.all_probabilities);
    
    // Update emotion badge
    const emotionBadge = document.getElementById('emotionBadge');
    if (emotionBadge) {
        const emotionName = emotionBadge.querySelector('.emotion-name');
        const confidence = emotionBadge.querySelector('.confidence');
        if (emotionName) emotionName.textContent = result.prediction;
        if (confidence) confidence.textContent = (result.confidence * 100).toFixed(1) + '%';
        
        console.log('✅ Updated emotion badge:', result.prediction, (result.confidence * 100).toFixed(1) + '%');
    }
    
    // Update probability bars
    const probabilityBars = document.querySelector('.probability-bars');
    if (probabilityBars) {
        const items = probabilityBars.querySelectorAll('.probability-item');
        const emotions = ['happiness', 'surprise', 'disgust', 'repression'];
        
        items.forEach((item, index) => {
            const emotion = emotions[index];
            const probability = result.all_probabilities[emotion] || 0;
            const progressBar = item.querySelector('.progress-bar');
            const percentage = item.querySelector('span:last-child');
            
            // Debug log to check values
            console.log(`Emotion: ${emotion}, Probability (decimal): ${probability}, Probability (%): ${probability * 100}`);
            
            if (progressBar) {
                // Convert decimal to percentage for display
                progressBar.style.width = (probability * 100) + '%';
            }
            if (percentage) {
                // Convert decimal to percentage for display
                percentage.textContent = (probability * 100).toFixed(1) + '%';
            }
        });
    }
    
    // Update model information
    const modelType = document.getElementById('modelType');
    const modelFeatures = document.getElementById('modelFeatures');
    const modelEvaluation = document.getElementById('modelEvaluation');
    
    if (modelType && result.model_info) {
        modelType.textContent = result.model_info.model_type || 'CNN-SVM Hybrid';
    }
    if (modelFeatures && result.model_info) {
        modelFeatures.textContent = result.model_info.feature_dimensions + ' dimensions' || '228 dimensions';
    }
    if (modelEvaluation && result.model_info) {
        modelEvaluation.textContent = result.model_info.evaluation_method || 'LOS0';
    }
    
    // Add processing info if not present
    const modelInfo = document.querySelector('.model-info');
    if (modelInfo && !modelInfo.querySelector('.processing-info')) {
        const processingDiv = document.createElement('div');
        processingDiv.className = 'processing-info mt-2 pt-2 border-top border-blue-200';
        processingDiv.innerHTML = `
            <div class="text-xs text-blue-600">
                <div>Processing Time: ${processingTime}s</div>
                <div>Timestamp: ${new Date(result.timestamp).toLocaleString()}</div>
                <div>Preprocessing: ${result.preprocessing || 'Unknown'}</div>
            </div>
        `;
        modelInfo.appendChild(processingDiv);
    }
}

// Emotion icons mapping
const emotionIcons = {
    'happiness': '😊',
    'surprise': '😲',
    'disgust': '🤢',
    'repression': '😔'
};

function updateProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(probabilities),
            datasets: [{
                label: 'Probability',
                data: Object.values(probabilities).map(p => p * 100),
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(251, 146, 60, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(251, 146, 60, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function resetUpload() {
    selectedFile = null;
    
    // Reset upload area to original state
    const uploadContent = document.querySelector('.upload-content');
    if (uploadContent) {
        uploadContent.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-3x mb-3"></i>
            <h4>Upload Video for Real Analysis</h4>
            <p>Drag and drop a video file or click to browse</p>
            <div class="model-status mb-3">
                <small class="text-muted">Real CNN-SVM model predictions</small>
            </div>
            <input type="file" id="videoUpload" accept="video/*" class="d-none">
            <button class="btn btn-primary" onclick="document.getElementById('videoUpload').click()">
                <i class="fas fa-upload me-2"></i>Choose File
            </button>
        `;
    }
    
    // Hide results section
    const analysisResults = document.getElementById('analysisResults');
    if (analysisResults) {
        analysisResults.classList.add('d-none');
    }
    
    // Reinitialize upload functionality
    initializeUpload();
}

// Show alert message
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlert = document.querySelector('.alert-container');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // Create alert
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert-container';
    alertContainer.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(alertContainer);
    
    // Position at top
    alertContainer.style.position = 'fixed';
    alertContainer.style.top = '20px';
    alertContainer.style.right = '20px';
    alertContainer.style.zIndex = '9999';
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertContainer) {
            alertContainer.remove();
        }
    }, 5000);
}

// Initialize animations
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);
    
    // Observe elements
    const animatedElements = document.querySelectorAll('.feature-card, .result-card, .doc-card');
    animatedElements.forEach(el => {
        el.classList.add('fade-in');
        observer.observe(el);
    });
}

// Initialize charts
function initializeCharts() {
    // Performance metrics chart
    const ctx = document.getElementById('performanceChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'UAR', 'Happiness Recall', 'Disgust Recall', 'Temporal Preservation'],
                datasets: [{
                    label: 'Our Method',
                    data: [46.3, 24.8, 71.6, 27.4, 100],
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

// Initialize scroll effects
function initializeScrollEffects() {
    // Parallax effect for hero section
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const parallax = scrolled * 0.5;
            heroSection.style.transform = `translateY(${parallax}px)`;
        });
    }
    
    // Navbar background on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 100) {
                navbar.classList.add('navbar-scrolled');
            } else {
                navbar.classList.remove('navbar-scrolled');
            }
        });
    }
}

// Initialize variables
let modelLoaded = false;
let selectedFile = null;
let processingStartTime = null;
let probabilityChart = null;
