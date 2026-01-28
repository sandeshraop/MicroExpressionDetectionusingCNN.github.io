// Main JavaScript for Micro-Expression Recognition Web Interface

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeNavigation();
    initializeUpload();
    initializeAnimations();
    initializeCharts();
    initializeScrollEffects();
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

// File upload functionality
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

// Handle file upload
function handleFileUpload(file) {
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
    
    // Show loading state
    showUploadLoading();
    
    // Simulate processing (in real implementation, this would send to backend)
    setTimeout(() => {
        processVideo(file);
    }, 2000);
}

// Show upload loading state
function showUploadLoading() {
    const uploadArea = document.getElementById('uploadArea');
    const uploadContent = uploadArea.querySelector('.upload-content');
    
    uploadContent.innerHTML = `
        <div class="loading-spinner">
            <div class="loading"></div>
            <p class="mt-3">Processing video...</p>
        </div>
    `;
}

// Process video (simulation)
function processVideo(file) {
    // Simulate analysis results
    const results = {
        emotion: 'Happiness',
        confidence: 85,
        probabilities: {
            happiness: 85,
            surprise: 10,
            disgust: 3,
            repression: 2
        }
    };
    
    displayResults(results);
}

// Display analysis results
function displayResults(results) {
    const uploadArea = document.getElementById('uploadArea');
    const analysisResults = document.getElementById('analysisResults');
    
    // Hide upload area, show results
    uploadArea.style.display = 'none';
    analysisResults.classList.remove('d-none');
    
    // Update emotion badge
    const emotionBadge = document.getElementById('emotionBadge');
    emotionBadge.innerHTML = `
        <span class="emotion-name">${results.emotion}</span>
        <span class="confidence">${results.confidence}%</span>
    `;
    
    // Update probability bars
    const probabilityBars = analysisResults.querySelector('.probability-bars');
    probabilityBars.innerHTML = `
        <div class="probability-item">
            <span>Happiness</span>
            <div class="progress">
                <div class="progress-bar bg-success" style="width: ${results.probabilities.happiness}%"></div>
            </div>
            <span>${results.probabilities.happiness}%</span>
        </div>
        <div class="probability-item">
            <span>Surprise</span>
            <div class="progress">
                <div class="progress-bar bg-info" style="width: ${results.probabilities.surprise}%"></div>
            </div>
            <span>${results.probabilities.surprise}%</span>
        </div>
        <div class="probability-item">
            <span>Disgust</span>
            <div class="progress">
                <div class="progress-bar bg-warning" style="width: ${results.probabilities.disgust}%"></div>
            </div>
            <span>${results.probabilities.disgust}%</span>
        </div>
        <div class="probability-item">
            <span>Repression</span>
            <div class="progress">
                <div class="progress-bar bg-danger" style="width: ${results.probabilities.repression}%"></div>
            </div>
            <span>${results.probabilities.repression}%</span>
        </div>
    `;
    
    // Animate progress bars
    setTimeout(() => {
        const progressBars = probabilityBars.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 100);
    
    showAlert('Analysis completed successfully!', 'success');
}

// Reset upload
function resetUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const analysisResults = document.getElementById('analysisResults');
    const fileInput = document.getElementById('videoUpload');
    
    uploadArea.style.display = 'block';
    analysisResults.classList.add('d-none');
    fileInput.value = '';
    
    // Reset upload content
    const uploadContent = uploadArea.querySelector('.upload-content');
    uploadContent.innerHTML = `
        <i class="fas fa-cloud-upload-alt fa-3x mb-3"></i>
        <h4>Upload Video for Analysis</h4>
        <p>Drag and drop a video file or click to browse</p>
        <input type="file" id="videoUpload" accept="video/*" class="d-none">
        <button class="btn btn-primary" onclick="document.getElementById('videoUpload').click()">
            <i class="fas fa-upload me-2"></i>Choose File
        </button>
    `;
    
    // Reinitialize upload
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

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('videoUpload')?.click();
    }
    
    // Escape to reset upload
    if (e.key === 'Escape') {
        const analysisResults = document.getElementById('analysisResults');
        if (analysisResults && !analysisResults.classList.contains('d-none')) {
            resetUpload();
        }
    }
});

// Touch support for mobile
if ('ontouchstart' in window) {
    document.body.classList.add('touch-device');
}

// Performance monitoring
window.addEventListener('load', function() {
    // Log performance metrics
    const loadTime = performance.now();
    console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    
    // Monitor for errors
    window.addEventListener('error', function(e) {
        console.error('JavaScript error:', e.error);
        showAlert('An error occurred. Please refresh the page.', 'danger');
    });
});

// Service Worker registration (for PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
