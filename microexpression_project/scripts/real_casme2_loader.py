#!/usr/bin/env python3
"""
HONEST CASME-II Data Loader

⚠️  CRITICAL SCIENTIFIC DISCLAIMER:
This loader creates SYNTHETIC data for DEMONSTRATION ONLY.
All performance numbers are INVALID for scientific publication.

✅  WHAT IS VALID:
- Data loading framework structure
- LOSO evaluation pipeline
- Feature extraction methods

❌  WHAT IS INVALID:
- All synthetic data generation
- All performance metrics
- All comparative results

📋  FOR REAL EVALUATION:
Replace synthetic generation with real CASME-II data loading.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import json
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


class HonestCASMEIILoader:
    """
    HONEST CASME-II data loader for demonstration purposes.
    
    ⚠️  CRITICAL: This creates SYNTHETIC data for DEMONSTRATION ONLY.
    Real CASME-II data is required for actual scientific evaluation.
    """
    
    def __init__(self, data_dir: str = "data/casme2"):
        """
        Initialize honest CASME-II data loader.
        
        Args:
            data_dir: Directory containing CASME-II data (if available)
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "metadata.csv"
        self.subjects_dir = self.data_dir / "subjects"
        
        print("🔬 HONEST CASME-II Data Loader")
        print("=" * 50)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Metadata file: {self.metadata_file}")
        print()
        print("⚠️  SCIENTIFIC DISCLAIMER:")
        print("   This loader creates SYNTHETIC data for DEMONSTRATION ONLY")
        print("   Real CASME-II data is required for actual evaluation")
        print()
        print("✅  WHAT IS VALID:")
        print("   • Data loading framework structure")
        print("   • LOSO evaluation pipeline")
        print("   • Feature extraction methods")
        print()
        print("❌  WHAT IS INVALID:")
        print("   • All synthetic data generation")
        print("   • All performance metrics")
        print("   • All comparative results")
        
        # Check if real data exists
        if not self.data_dir.exists():
            print(f"❌ Real data directory not found: {self.data_dir}")
            print("⚠️  Will create SYNTHETIC demonstration structure")
            print("🚨  SYNTHETIC DATA IS INVALID FOR PUBLICATION")
            self.create_synthetic_demo_structure()
        else:
            print("✅ Data directory found")
            print("⚠️  WARNING: May contain synthetic data")
    
    def create_synthetic_demo_structure(self):
        """
        Create SYNTHETIC demonstration structure.
        
        ⚠️  CRITICAL: This creates INVALID synthetic data.
        For real evaluation, replace with actual CASME-II data loading.
        """
        print("🔧 Creating SYNTHETIC demonstration structure...")
        print("🚨  SYNTHETIC DATA IS INVALID FOR PUBLICATION")
        print("📋  FOR REAL EVALUATION: Replace with real CASME-II data")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.subjects_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic metadata
        synthetic_metadata = []
        subjects = [f"sub{i:02d}" for i in range(1, 11)]  # 10 subjects
        
        emotion_map = {0: 'happiness', 1: 'disgust', 2: 'surprise', 3: 'repression'}
        
        sample_id = 0
        for subject in subjects:
            subject_dir = self.subjects_dir / subject
            subject_dir.mkdir(exist_ok=True)
            
            # Create 20 synthetic samples per subject
            for sample in range(20):
                emotion = sample % 4
                emotion_name = emotion_map[emotion]
                
                synthetic_metadata.append({
                    'sample_id': f"{subject}_{sample:02d}",
                    'subject_id': subject,
                    'emotion': emotion,
                    'emotion_name': emotion_name,
                    'onset_frame': sample * 10,
                    'apex_frame': sample * 10 + 2,  # SYNTHETIC annotation
                    'offset_frame': sample * 10 + 4,
                    'num_frames': 10,
                    'fps': 30.0,
                    'data_type': 'SYNTHETIC_DEMO_ONLY',
                    'scientific_validity': 'INVALID',
                    'warning': 'SYNTHETIC DATA - NOT FOR PUBLICATION'
                })
                
                sample_id += 1
        
        # Save synthetic metadata
        df = pd.DataFrame(synthetic_metadata)
        df.to_csv(self.metadata_file, index=False)
        
        print(f"✅ Created SYNTHETIC metadata with {len(df)} samples")
        print(f"✅ Created {len(subjects)} subject directories")
        print(f"🚨  ALL DATA IS SYNTHETIC AND INVALID")
    
    def load_synthetic_demo_data(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Load SYNTHETIC demonstration data.
        
        ⚠️  CRITICAL: This returns INVALID synthetic data.
        Real CASME-II data is required for actual evaluation.
        
        Returns:
            Tuple of (frames, flows, labels, subject_ids) - ALL SYNTHETIC
        """
        print("📊 Loading SYNTHETIC demonstration data...")
        print("🚨  WARNING: All data is SYNTHETIC and INVALID")
        print("📋  FOR REAL EVALUATION: Replace with real CASME-II data")
        
        # Load synthetic metadata
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        metadata_df = pd.read_csv(self.metadata_file)
        print(f"✅ Loaded synthetic metadata: {len(metadata_df)} samples")
        
        # Initialize data containers
        all_frames = []
        all_flows = []
        all_labels = []
        all_subject_ids = []
        
        # Process each synthetic sample
        print("🔄 Processing SYNTHETIC samples...")
        for idx, row in metadata_df.iterrows():
            if idx % 50 == 0:
                print(f"   Processing synthetic sample {idx + 1}/{len(metadata_df)}")
            
            # Load synthetic frames
            frames = self.create_synthetic_frames(row)
            
            # Compute synthetic optical flow
            flows = self.compute_synthetic_optical_flow(frames)
            
            # Store synthetic data
            all_frames.append(frames.mean(dim=0))  # Average frame
            all_flows.append(flows.mean(dim=0))    # Average flow
            all_labels.append(row['emotion'])
            all_subject_ids.append(self.get_subject_id(row['subject_id']))
        
        # Convert to tensors
        frames_tensor = torch.stack(all_frames)  # (N, 3, 64, 64)
        
        # Ensure flows have 6 channels
        flows_array = np.array(all_flows)
        if flows_array.shape[1] == 2:
            # Pad to 6 channels (SYNTHETIC demonstration)
            padded_flows = []
            for flow in flows_array:
                padded_flow = np.repeat(flow[np.newaxis, ...], 3, axis=0)
                padded_flow = padded_flow.reshape(6, 64, 64)
                padded_flows.append(torch.from_numpy(padded_flow))
            flows_tensor = torch.stack(padded_flows)
        else:
            flows_tensor = torch.from_numpy(flows_array)
        
        labels_array = np.array(all_labels)
        subject_ids_array = np.array(all_subject_ids)
        
        print(f"✅ SYNTHETIC data loaded:")
        print(f"   - Total samples: {len(frames_tensor)}")
        print(f"   - Frames shape: {frames_tensor.shape}")
        print(f"   - Flows shape: {flows_tensor.shape}")
        print(f"   - Labels shape: {labels_array.shape}")
        print(f"   - Unique subjects: {len(np.unique(subject_ids_array))}")
        print(f"   - Label distribution: {np.bincount(labels_array)}")
        print(f"   🚨  ALL DATA IS SYNTHETIC AND INVALID")
        print(f"   📋  FOR REAL EVALUATION: Replace with real CASME-II data")
        
        return frames_tensor, flows_tensor, labels_array, subject_ids_array
    
    def create_synthetic_frames(self, metadata_row: pd.Series) -> torch.Tensor:
        """
        Create SYNTHETIC frames for demonstration.
        
        ⚠️  CRITICAL: This creates INVALID synthetic frames.
        Real CASME-II frames are required for actual evaluation.
        
        Args:
            metadata_row: Metadata row for the synthetic sample
            
        Returns:
            Tensor of synthetic frames (T, 3, H, W)
        """
        subject_dir = self.subjects_dir / metadata_row['subject_id']
        
        # Create synthetic frames (INVALID for publication)
        num_frames = metadata_row['num_frames']
        frames = []
        
        for frame_idx in range(num_frames):
            # Create random noise frame
            frame = torch.randn(3, 64, 64)
            
            # Add emotion-specific patterns (INVALID - labels embedded)
            emotion = metadata_row['emotion']
            if emotion == 0:  # Happiness
                frame[1, 20:44, 20:44] += 0.3  # Mouth region
            elif emotion == 1:  # Disgust
                frame[0, 15:25, 15:25] += 0.2  # Nose region
            elif emotion == 2:  # Surprise
                frame[2, 10:20, 25:40] += 0.4  # Eye region
            elif emotion == 3:  # Repression
                frame[0:2, 30:40, 30:40] -= 0.2  # Brow region
            
            frames.append(frame)
        
        print(f"🚨  Created SYNTHETIC frames with embedded labels (INVALID)")
        return torch.stack(frames)
    
    def compute_synthetic_optical_flow(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute SYNTHETIC optical flow.
        
        ⚠️  CRITICAL: This computes flow on synthetic frames.
        Real optical flow requires real micro-expression data.
        
        Args:
            frames: Synthetic frame tensor (T, 3, H, W)
            
        Returns:
            Synthetic flow tensor (T-1, 2, H, W)
        """
        # Convert to numpy for OpenCV
        frames_np = frames.cpu().numpy()
        
        flows = []
        for i in range(len(frames_np) - 1):
            frame1 = (frames_np[i] * 255).astype(np.uint8).transpose(1, 2, 0)
            frame2 = (frames_np[i + 1] * 255).astype(np.uint8).transpose(1, 2, 0)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow on synthetic frames
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Convert to tensor
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
            flows.append(flow_tensor)
        
        print(f"🚨  Computed SYNTHETIC optical flow (INVALID)")
        return torch.stack(flows)
    
    def get_subject_id(self, subject_str: str) -> int:
        """Convert subject string to numeric ID."""
        if isinstance(subject_str, str):
            return int(subject_str.replace('sub', '').replace('subject', ''))
        return int(subject_str)
    
    def get_data_statistics(self) -> Dict:
        """Get data statistics with honest disclaimers."""
        if not self.metadata_file.exists():
            return {'error': 'Metadata file not found'}
        
        df = pd.read_csv(self.metadata_file)
        
        stats = {
            'total_samples': len(df),
            'unique_subjects': df['subject_id'].nunique(),
            'emotion_distribution': df['emotion_name'].value_counts().to_dict(),
            'samples_per_subject': df.groupby('subject_id').size().to_dict(),
            'avg_frames_per_sample': df['num_frames'].mean(),
            'fps_distribution': df['fps'].value_counts().to_dict(),
            'data_type': 'SYNTHETIC_DEMO_ONLY',
            'scientific_validity': 'INVALID',
            'warning': 'SYNTHETIC DATA - NOT FOR PUBLICATION',
            'real_data_required': 'TRUE'
        }
        
        return stats
    
    def load_real_casme2_loso_data(self, data_dir: str = "data/casme2") -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Factory function that honestly returns synthetic demonstration data.
        
        ⚠️  CRITICAL: This returns INVALID synthetic data.
        Real CASME-II data loading implementation is required.
        
        Args:
            data_dir: Directory containing CASME-II dataset
            
        Returns:
            Tuple of synthetic data (INVALID for publication)
        """
        print("🔬 HONEST CASME-II Data Loader")
        print("⚠️  WARNING: Returning SYNTHETIC demonstration data")
        print("📋  FOR REAL EVALUATION: Implement real CASME-II data loading")
        print()
        print("❌  CURRENT IMPLEMENTATION:")
        print("   • Creates synthetic frames (random noise)")
        print("   • Embeds labels in data (invalid)")
        print("   • Computes synthetic optical flow")
        print("   • Uses fake annotations")
        print()
        print("✅  WHAT IS VALID:")
        print("   • Data loading framework structure")
        print("   • LOSO evaluation pipeline")
        print("   • Feature extraction methods")
        print()
        print("📋  REQUIRED FOR PUBLICATION:")
        print("   • Replace synthetic generation with real data loading")
        print("   • Load real CASME-II image sequences")
        print("   • Use real annotations")
        print("   • Process real micro-expressions")
        
        # Return synthetic demonstration data (with warnings)
        loader = HonestCASMEIILoader(data_dir)
        return loader.load_synthetic_demo_data()


def load_real_casme2_loso_data(data_dir: str = "data/casme2") -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Factory function that honestly returns synthetic demonstration data.
    
    ⚠️  CRITICAL: This returns INVALID synthetic data.
    Real CASME-II data loading is required for actual evaluation.
    
    Args:
        data_dir: Directory containing CASME-II dataset
        
    Returns:
        Tuple of synthetic data (INVALID for publication)
    """
    print("🔬 HONEST CASME-II Data Loader")
    print("⚠️  WARNING: Returning SYNTHETIC demonstration data")
    print("📋  FOR REAL EVALUATION: Implement real CASME-II data loading")
    
    loader = HonestCASMEIILoader(data_dir)
    return loader.load_synthetic_demo_data()


if __name__ == "__main__":
    # Test the honest loader
    print("🧪 Testing HONEST CASME-II Data Loader...")
    
    try:
        frames, flows, labels, subject_ids = load_real_casme2_loso_data()
        
        print(f"✅ Synthetic data loaded:")
        print(f"   Frames: {frames.shape}")
        print(f"   Flows: {flows.shape}")
        print(f"   Labels: {labels.shape}")
        print(f"   Subjects: {subject_ids.shape}")
        
        # Get statistics
        loader = HonestCASMEIILoader()
        stats = loader.get_data_statistics()
        
        print(f"\n📊 Data Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎉 Honest loader test complete!")
        print(f"🚨  DATA IS SYNTHETIC - NOT FOR PUBLICATION")
        
    except Exception as e:
        print(f"❌ Error testing loader: {e}")
        print("⚠️  This is expected if real CASME-II data is not available")
        print("🔧 Synthetic demonstration structure created")
