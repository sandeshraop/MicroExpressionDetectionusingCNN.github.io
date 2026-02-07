#!/usr/bin/env python3
"""
Micro-Expression Detection System - Architecture Flow Visualization
Creates comprehensive diagrams showing system architecture and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_system_architecture_diagram():
    """Create overall system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Micro-Expression Detection System Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Define components with colors
    components = {
        'Data Layer': {'pos': (1, 8), 'size': (2, 1), 'color': '#FF6B6B'},
        'Preprocessing': {'pos': (4, 8), 'size': (2, 1), 'color': '#4ECDC4'},
        'Feature Extraction': {'pos': (7, 8), 'size': (2, 1), 'color': '#45B7D1'},
        
        'CASME-II Dataset': {'pos': (0.5, 6.5), 'size': (1.5, 0.8), 'color': '#FF9999'},
        'Video Processing': {'pos': (3.5, 6.5), 'size': (1.5, 0.8), 'color': '#7DD3C0'},
        'Optical Flow': {'pos': (6.5, 6.5), 'size': (1.5, 0.8), 'color': '#6BB6FF'},
        
        'Face Detection': {'pos': (3.5, 5.2), 'size': (1.5, 0.8), 'color': '#7DD3C0'},
        'CNN Features': {'pos': (6.5, 5.2), 'size': (1.5, 0.8), 'color': '#6BB6FF'},
        
        'Model Layer': {'pos': (1, 3.5), 'size': (2, 1), 'color': '#96CEB4'},
        'Training': {'pos': (4, 3.5), 'size': (2, 1), 'color': '#FFEAA7'},
        'Evaluation': {'pos': (7, 3.5), 'size': (2, 1), 'color': '#DDA0DD'},
        
        'Hybrid CNN': {'pos': (0.5, 2), 'size': (1.5, 0.8), 'color': '#B4E7CE'},
        'SVM Classifier': {'pos': (3.5, 2), 'size': (1.5, 0.8), 'color': '#FFF3B2'},
        'LOSO Validation': {'pos': (6.5, 2), 'size': (1.5, 0.8), 'color': '#E8B4F3'},
        
        'Application Layer': {'pos': (2.5, 0.5), 'size': (2, 1), 'color': '#FFB6C1'},
        'Web Interface': {'pos': (5.5, 0.5), 'size': (2, 1), 'color': '#87CEEB'},
    }
    
    # Draw components
    for name, props in components.items():
        box = FancyBboxPatch(props['pos'], props['size'][0], props['size'][1],
                           boxstyle="round,pad=0.1", 
                           facecolor=props['color'], 
                           edgecolor='black', 
                           linewidth=1.5,
                           alpha=0.8)
        ax.add_patch(box)
        ax.text(props['pos'][0] + props['size'][0]/2, 
               props['pos'][1] + props['size'][1]/2,
               name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections
    connections = [
        ((1.5, 7.2), (1.5, 6.5)),  # Data Layer to CASME-II
        ((4.5, 7.2), (4.5, 6.5)),  # Preprocessing to Video Processing
        ((7.5, 7.2), (7.5, 6.5)),  # Feature Extraction to Optical Flow
        
        ((2, 6.9), (3.5, 6.9)),    # CASME-II to Video Processing
        ((5, 6.9), (6.5, 6.9)),    # Video Processing to Optical Flow
        
        ((4.5, 6.5), (4.5, 5.6)),  # Video Processing to Face Detection
        ((7.5, 6.5), (7.5, 5.6)),  # Optical Flow to CNN Features
        
        ((2, 3.9), (2, 2.4)),      # Model Layer to Hybrid CNN
        ((4.5, 3.9), (4.5, 2.4)),  # Training to SVM Classifier
        ((7.5, 3.9), (7.5, 2.4)),  # Evaluation to LOSO Validation
        
        ((3, 2.4), (3.5, 2.4)),    # Hybrid CNN to SVM Classifier
        ((5, 2.4), (6.5, 2.4)),    # SVM Classifier to LOSO Validation
        
        ((3.5, 1.5), (3.5, 1.2)),  # Model to Application Layer
        ((6.5, 1.5), (6.5, 1.2)),  # Evaluation to Web Interface
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    return fig

def create_data_processing_pipeline():
    """Create detailed data processing pipeline diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Data Processing Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Pipeline stages
    stages = [
        {'name': 'Raw Video Input', 'pos': (0.5, 6), 'size': (2, 0.8), 'color': '#FF6B6B'},
        {'name': 'Face Detection', 'pos': (3, 6), 'size': (2, 0.8), 'color': '#4ECDC4'},
        {'name': 'Face Cropping', 'pos': (5.5, 6), 'size': (2, 0.8), 'color': '#45B7D1'},
        {'name': 'Resizing (64x64)', 'pos': (8, 6), 'size': (2, 0.8), 'color': '#96CEB4'},
        {'name': 'Normalization', 'pos': (10.5, 6), 'size': (2, 0.8), 'color': '#FFEAA7'},
        
        {'name': 'Frame Selection', 'pos': (0.5, 4.5), 'size': (2, 0.8), 'color': '#DDA0DD'},
        {'name': 'Onset Frame', 'pos': (3, 4.5), 'size': (1.8, 0.8), 'color': '#FFB6C1'},
        {'name': 'Apex Frame', 'pos': (5.2, 4.5), 'size': (1.8, 0.8), 'color': '#87CEEB'},
        {'name': 'Offset Frame', 'pos': (7.4, 4.5), 'size': (1.8, 0.8), 'color': '#98D8C8'},
        
        {'name': 'Optical Flow', 'pos': (2, 3), 'size': (2, 0.8), 'color': '#F7DC6F'},
        {'name': 'Strain Analysis', 'pos': (5, 3), 'size': (2, 0.8), 'color': '#BB8FCE'},
        {'name': 'Feature Vector', 'pos': (8, 3), 'size': (2, 0.8), 'color': '#85C1E2'},
        
        {'name': 'CNN Features', 'pos': (2, 1.5), 'size': (2, 0.8), 'color': '#F8B739'},
        {'name': 'AU Statistics', 'pos': (5, 1.5), 'size': (2, 0.8), 'color': '#52BE80'},
        {'name': 'Hybrid Features', 'pos': (8, 1.5), 'size': (2, 0.8), 'color': '#EC7063'},
    ]
    
    # Draw stages
    for stage in stages:
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                           boxstyle="round,pad=0.05", 
                           facecolor=stage['color'], 
                           edgecolor='black', 
                           linewidth=1.2,
                           alpha=0.8)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1]/2,
               stage['name'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw flow arrows
    flow_connections = [
        ((2.5, 6.4), (3, 6.4)),      # Raw to Face Detection
        ((5, 6.4), (5.5, 6.4)),      # Face Detection to Cropping
        ((7.5, 6.4), (8, 6.4)),      # Cropping to Resizing
        ((10, 6.4), (10.5, 6.4)),    # Resizing to Normalization
        
        ((1.5, 5.6), (1.5, 4.9)),    # Normalization to Frame Selection
        ((2.5, 4.9), (3, 4.9)),      # Frame Selection to Onset
        ((4.8, 4.9), (5.2, 4.9)),    # Onset to Apex
        ((7, 4.9), (7.4, 4.9)),      # Apex to Offset
        
        ((3.9, 4.1), (3, 3.4)),      # Frame Selection to Optical Flow
        ((6.1, 4.1), (5, 3.4)),      # Frame Selection to Strain
        ((4, 3.4), (5, 3.4)),        # Optical Flow to Strain
        ((7, 3.4), (8, 3.4)),        # Strain to Feature Vector
        
        ((3, 2.6), (3, 1.9)),        # Optical Flow to CNN Features
        ((6, 2.6), (6, 1.9)),        # Strain to AU Statistics
        ((4, 1.9), (5, 1.9)),        # CNN to AU
        ((7, 1.9), (8, 1.9)),        # AU to Hybrid
    ]
    
    for start, end in flow_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    plt.tight_layout()
    return fig

def create_model_training_flow():
    """Create model training methodology flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'Model Training Methodology', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Training phases
    phases = [
        {'name': 'Dataset Preparation', 'pos': (0.5, 4.5), 'size': (2.5, 0.8), 'color': '#FF6B6B'},
        {'name': 'Feature Extraction', 'pos': (3.5, 4.5), 'size': (2.5, 0.8), 'color': '#4ECDC4'},
        {'name': 'Model Initialization', 'pos': (6.5, 4.5), 'size': (2.5, 0.8), 'color': '#45B7D1'},
        
        {'name': 'Hybrid CNN\nArchitecture', 'pos': (0.5, 3.2), 'size': (2.5, 1), 'color': '#96CEB4'},
        {'name': 'SVM Classifier\nTraining', 'pos': (3.5, 3.2), 'size': (2.5, 1), 'color': '#FFEAA7'},
        {'name': 'Hyperparameter\nOptimization', 'pos': (6.5, 3.2), 'size': (2.5, 1), 'color': '#DDA0DD'},
        
        {'name': 'LOSO Validation', 'pos': (1.5, 1.5), 'size': (2, 0.8), 'color': '#FFB6C1'},
        {'name': 'Performance Metrics', 'pos': (4.5, 1.5), 'size': (2, 0.8), 'color': '#87CEEB'},
        {'name': 'Model Selection', 'pos': (7.5, 1.5), 'size': (2, 0.8), 'color': '#98D8C8'},
    ]
    
    # Draw phases
    for phase in phases:
        box = FancyBboxPatch(phase['pos'], phase['size'][0], phase['size'][1],
                           boxstyle="round,pad=0.05", 
                           facecolor=phase['color'], 
                           edgecolor='black', 
                           linewidth=1.2,
                           alpha=0.8)
        ax.add_patch(box)
        ax.text(phase['pos'][0] + phase['size'][0]/2, 
               phase['pos'][1] + phase['size'][1]/2,
               phase['name'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add training loop
    loop_box = FancyBboxPatch((2.5, 0.2), 5, 0.8,
                            boxstyle="round,pad=0.1", 
                            facecolor='#F0F0F0', 
                            edgecolor='red', 
                            linewidth=2,
                            alpha=0.6)
    ax.add_patch(loop_box)
    ax.text(5, 0.6, 'Training Loop: Epoch Iteration Until Convergence', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    # Draw connections
    connections = [
        ((3, 4.9), (3.5, 4.9)),      # Dataset to Features
        ((6, 4.9), (6.5, 4.9)),      # Features to Model Init
        
        ((1.75, 4.1), (1.75, 3.7)),  # Dataset to CNN
        ((4.75, 4.1), (4.75, 3.7)),  # Features to SVM
        ((7.75, 4.1), (7.75, 3.7)),  # Model Init to Hyperparams
        
        ((2.5, 3.2), (3.5, 3.2)),    # CNN to SVM
        ((6, 3.2), (6.5, 3.2)),      # SVM to Hyperparams
        
        ((2.5, 2.7), (2.5, 1.9)),    # CNN to LOSO
        ((4.75, 2.7), (4.75, 1.9)),  # SVM to Metrics
        ((7.5, 2.7), (7.5, 1.9)),    # Hyperparams to Selection
        
        ((3.5, 1.5), (4.5, 1.5)),    # LOSO to Metrics
        ((6.5, 1.5), (7.5, 1.5)),    # Metrics to Selection
        
        # Training loop connections
        ((5, 1.1), (5, 1.0)),        # Metrics to loop
        ((2.5, 0.6), (1.75, 0.6)),   # Loop to CNN
        ((7.5, 0.6), (7.75, 0.6)),   # Loop to Hyperparams
        ((1.75, 0.6), (1.75, 1.1)),  # Loop up to CNN
        ((7.75, 0.6), (7.75, 1.1)),  # Loop up to Hyperparams
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    plt.tight_layout()
    return fig

def create_inference_pipeline():
    """Create inference/prediction flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'Real-time Inference Pipeline', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Inference stages
    stages = [
        {'name': 'Input Video\nStream', 'pos': (0.5, 4.5), 'size': (2, 1), 'color': '#FF6B6B'},
        {'name': 'Preprocessing\nPipeline', 'pos': (3, 4.5), 'size': (2, 1), 'color': '#4ECDC4'},
        {'name': 'Feature\nExtraction', 'pos': (5.5, 4.5), 'size': (2, 1), 'color': '#45B7D1'},
        {'name': 'Model\nPrediction', 'pos': (8, 4.5), 'size': (2, 1), 'color': '#96CEB4'},
        
        {'name': 'Face Detection\n& Tracking', 'pos': (0.5, 3), 'size': (2, 0.8), 'color': '#FFEAA7'},
        {'name': 'Optical Flow\nComputation', 'pos': (3, 3), 'size': (2, 0.8), 'color': '#DDA0DD'},
        {'name': 'CNN Feature\nExtraction', 'pos': (5.5, 3), 'size': (2, 0.8), 'color': '#FFB6C1'},
        {'name': 'Emotion\nClassification', 'pos': (8, 3), 'size': (2, 0.8), 'color': '#87CEEB'},
        
        {'name': 'Confidence\nScore', 'pos': (2, 1.5), 'size': (1.8, 0.8), 'color': '#98D8C8'},
        {'name': 'Visualization\nOutput', 'pos': (4.5, 1.5), 'size': (1.8, 0.8), 'color': '#F7DC6F'},
        {'name': 'Web Interface\nDisplay', 'pos': (7, 1.5), 'size': (1.8, 0.8), 'color': '#BB8FCE'},
    ]
    
    # Draw stages
    for stage in stages:
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                           boxstyle="round,pad=0.05", 
                           facecolor=stage['color'], 
                           edgecolor='black', 
                           linewidth=1.2,
                           alpha=0.8)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1]/2,
               stage['name'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw real-time indicator
    rt_circle = Circle((9.5, 5.2), 0.3, color='red', alpha=0.7)
    ax.add_patch(rt_circle)
    ax.text(9.5, 5.7, 'Real-time', ha='center', fontsize=8, fontweight='bold', color='red')
    
    # Draw connections
    connections = [
        ((2.5, 5), (3, 5)),          # Input to Preprocessing
        ((5, 5), (5.5, 5)),          # Preprocessing to Features
        ((7.5, 5), (8, 5)),          # Features to Model
        
        ((1.5, 4.5), (1.5, 3.8)),    # Input to Face Detection
        ((4, 4.5), (4, 3.4)),        # Preprocessing to Optical Flow
        ((6.5, 4.5), (6.5, 3.4)),    # Features to CNN
        ((9, 4.5), (9, 3.4)),        # Model to Classification
        
        ((2.5, 3.4), (3, 3.4)),      # Face Detection to Optical Flow
        ((5, 3.4), (5.5, 3.4)),      # Optical Flow to CNN
        ((7.5, 3.4), (8, 3.4)),      # CNN to Classification
        
        ((9, 3), (8.5, 2.3)),        # Classification to Confidence
        ((8.5, 2.3), (5.4, 2.3)),    # Confidence to Visualization
        ((5.4, 2.3), (7, 1.9)),      # Visualization to Web Interface
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    plt.tight_layout()
    return fig

def create_comprehensive_methodology():
    """Create comprehensive methodology overview diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Comprehensive Micro-Expression Recognition Methodology', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Main methodology components
    components = [
        # Data Collection & Preparation
        {'name': 'Data Collection\n& Preparation', 'pos': (0.5, 6.5), 'size': (2.5, 1), 'color': '#FF6B6B'},
        {'name': 'CASME-II Dataset\n(4 emotions)', 'pos': (0.5, 5.2), 'size': (2.5, 0.8), 'color': '#FF9999'},
        {'name': 'Label Mapping\n(Happiness, Surprise,\nDisgust, Repression)', 'pos': (0.5, 4), 'size': (2.5, 1.2), 'color': '#FFCCCC'},
        
        # Feature Engineering
        {'name': 'Feature Engineering\n& Extraction', 'pos': (3.5, 6.5), 'size': (2.5, 1), 'color': '#4ECDC4'},
        {'name': 'Optical Flow\nAnalysis', 'pos': (3.5, 5.2), 'size': (2.5, 0.8), 'color': '#7DD3C0'},
        {'name': 'AU-aligned\nStrain Statistics', 'pos': (3.5, 4), 'size': (2.5, 0.8), 'color': '#A8E6CF'},
        {'name': 'CNN Flow Features\n(128-D)', 'pos': (3.5, 2.8), 'size': (2.5, 0.8), 'color': '#C8F7DC'},
        
        # Model Architecture
        {'name': 'Model Architecture\n& Training', 'pos': (6.5, 6.5), 'size': (2.5, 1), 'color': '#45B7D1'},
        {'name': 'Hybrid CNN\nModel', 'pos': (6.5, 5.2), 'size': (2.5, 0.8), 'color': '#6BB6FF'},
        {'name': 'SVM Classifier', 'pos': (6.5, 4), 'size': (2.5, 0.8), 'color': '#8FC9FF'},
        {'name': '216-D Feature\nVector', 'pos': (6.5, 2.8), 'size': (2.5, 0.8), 'color': '#B3D9FF'},
        
        # Evaluation & Validation
        {'name': 'Evaluation\n& Validation', 'pos': (9.5, 6.5), 'size': (2.5, 1), 'color': '#96CEB4'},
        {'name': 'LOSO Cross-\nValidation', 'pos': (9.5, 5.2), 'size': (2.5, 0.8), 'color': '#B4E7CE'},
        {'name': 'Performance\nMetrics', 'pos': (9.5, 4), 'size': (2.5, 0.8), 'color': '#D4F1E0'},
        {'name': 'Confusion\nMatrix', 'pos': (9.5, 2.8), 'size': (2.5, 0.8), 'color': '#E8F8F0'},
        
        # Deployment
        {'name': 'Deployment\n& Application', 'pos': (5, 1.2), 'size': (3, 0.8), 'color': '#FFEAA7'},
    ]
    
    # Draw components
    for comp in components:
        box = FancyBboxPatch(comp['pos'], comp['size'][0], comp['size'][1],
                           boxstyle="round,pad=0.05", 
                           facecolor=comp['color'], 
                           edgecolor='black', 
                           linewidth=1.2,
                           alpha=0.8)
        ax.add_patch(box)
        ax.text(comp['pos'][0] + comp['size'][0]/2, 
               comp['pos'][1] + comp['size'][1]/2,
               comp['name'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw main flow connections
    main_flow = [
        ((3, 7), (3.5, 7)),          # Data to Features
        ((6, 7), (6.5, 7)),          # Features to Model
        ((9, 7), (9.5, 7)),          # Model to Evaluation
        
        # Vertical connections within each column
        ((1.75, 6.5), (1.75, 6)),    # Data to CASME-II
        ((1.75, 5.2), (1.75, 5.2)),  # CASME-II to Labels
        ((1.75, 4), (1.75, 3.8)),    # Labels to bottom
        
        ((4.75, 6.5), (4.75, 6)),    # Features to Optical Flow
        ((4.75, 5.2), (4.75, 4.8)),  # Optical Flow to AU
        ((4.75, 4), (4.75, 3.6)),    # AU to CNN
        
        ((7.75, 6.5), (7.75, 6)),    # Model to Hybrid CNN
        ((7.75, 5.2), (7.75, 4.8)),  # CNN to SVM
        ((7.75, 4), (7.75, 3.6)),    # SVM to Features
        
        ((10.75, 6.5), (10.75, 6)),  # Eval to LOSO
        ((10.75, 5.2), (10.75, 4.8)), # LOSO to Metrics
        ((10.75, 4), (10.75, 3.6)),   # Metrics to Confusion
        
        # Connection to deployment
        ((6.5, 2.8), (6.5, 2)),      # Model to Deployment
        ((10.75, 2.8), (8, 1.6)),     # Evaluation to Deployment
        ((4.75, 2.8), (5, 2)),        # Features to Deployment
    ]
    
    for start, end in main_flow:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add methodology annotations
    ax.text(1.75, 0.8, 'Phase 1:\nData Prep', ha='center', fontsize=9, fontweight='bold', color='#FF6B6B')
    ax.text(4.75, 0.8, 'Phase 2:\nFeature Eng.', ha='center', fontsize=9, fontweight='bold', color='#4ECDC4')
    ax.text(7.75, 0.8, 'Phase 3:\nModeling', ha='center', fontsize=9, fontweight='bold', color='#45B7D1')
    ax.text(10.75, 0.8, 'Phase 4:\nEvaluation', ha='center', fontsize=9, fontweight='bold', color='#96CEB4')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualization diagrams"""
    print("🎨 Generating Micro-Expression Detection System Visualizations...")
    
    # Create output directory
    output_dir = Path("c:/New Project/microexpression_project/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all diagrams
    diagrams = [
        ('system_architecture', create_system_architecture_diagram),
        ('data_processing_pipeline', create_data_processing_pipeline),
        ('model_training_flow', create_model_training_flow),
        ('inference_pipeline', create_inference_pipeline),
        ('comprehensive_methodology', create_comprehensive_methodology),
    ]
    
    for name, create_func in diagrams:
        print(f"📊 Creating {name} diagram...")
        fig = create_func()
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Saved: {output_path}")
    
    print("\n🎉 All visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Display summary
    print("\n📋 Generated Diagrams:")
    for name, _ in diagrams:
        print(f"  - {name}.png")

if __name__ == "__main__":
    main()
