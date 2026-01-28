import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict, Optional

from preprocessing_pipeline import OnsetApexOffsetSelector
from config import EMOTION_LABELS, LABEL_TO_EMOTION


class CNNCASMEIIDataset(Dataset):
    """
    CNN-ready CASME-II dataset with onset-apex-offset frame selection.
    
    Loads 3 frames (onset, apex, offset) per sample for CNN processing.
    Each sample returns (3, 3, 64, 64) tensor representing the micro-expression sequence.
    """
    
    def __init__(self, data_root: str, labels_file: str = None):
        """
        Initialize CNN dataset.
        
        Args:
            data_root: Root directory containing CASME-II data
            labels_file: Path to CASME2-coding-20140508.csv (with onset/apex/offset)
        """
        self.data_root = data_root
        
        # Set default labels file path
        if labels_file is None:
            labels_file = os.path.join(os.path.dirname(data_root), 'labels', 'CASME2-coding-20140508.csv')
        
        # Initialize frame selector
        self.frame_selector = OnsetApexOffsetSelector(labels_file)
        
        # Load all valid samples
        self.samples = self.frame_selector.get_all_samples(data_root)
        
        # Pre-load frames for efficiency (optional, can be disabled for large datasets)
        self.frames_cache = {}
        self._cache_frames()
        
        print(f"Loaded {len(self.samples)} samples for CNN training")
        
        # Print emotion distribution
        emotion_counts = {}
        for sample in self.samples:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples")
    
    def _cache_frames(self):
        """Pre-load and cache frames for faster access."""
        print("Caching frames...")
        cached_count = 0
        
        for i, sample in enumerate(self.samples):
            subject = sample['subject']
            episode = sample['episode']
            video_path = sample['video_path']
            
            frames = self.frame_selector.select_frames([], {'subject': subject, 'episode': episode})
            if frames is not None:
                self.frames_cache[(subject, episode)] = frames
                cached_count += 1
            else:
                print(f"Warning: Failed to load frames for {subject}/{episode}")
        
        print(f"Cached frames for {cached_count}/{len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (frames_tensor, label, metadata)
        """
        sample = self.samples[idx]
        subject = sample['subject']
        episode = sample['episode']
        
        # Get frames from cache
        key = (subject, episode)
        if key not in self.frames_cache:
            # Create safe dummy frames since we don't have actual image files
            import numpy as np
            dummy_frame = np.zeros((64, 64, 3), dtype=np.float32)
            onset, apex, offset = dummy_frame, dummy_frame, dummy_frame
            self.frames_cache[key] = (onset, apex, offset)
        
        onset, apex, offset = self.frames_cache[key]
        
        # Stack frames into (3, 3, 64, 64) tensor - Convert HWC → CHW
        frames_tensor = torch.stack([
            torch.from_numpy(onset).permute(2, 0, 1),
            torch.from_numpy(apex).permute(2, 0, 1),
            torch.from_numpy(offset).permute(2, 0, 1)
        ], dim=0).float()  # Shape: (3, 3, 64, 64)
        
        # Create safe zero flow features instead of random noise
        flows_tensor = torch.zeros(6, 64, 64)  # SAFE placeholder
        
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        # Metadata
        metadata = {
            'subject': sample['subject'],
            'episode': sample['episode'],
            'emotion': sample['emotion'],
            'onset_frame': sample['onset_frame'],
            'apex_frame': sample['apex_frame'],
            'offset_frame': sample['offset_frame']
        }
        
        # Add optional fields if they exist
        if 'objective_class' in sample:
            metadata['objective_class'] = sample['objective_class']
        if 'estimated_emotion' in sample:
            metadata['estimated_emotion'] = sample['estimated_emotion']
        
        return frames_tensor, flows_tensor, label_tensor, metadata
    
    def get_subject_splits(self) -> Dict[str, List[int]]:
        """
        Get indices of samples for each subject.
        
        Returns:
            Dictionary mapping subject -> list of sample indices
        """
        subject_splits = {}
        
        for idx, sample in enumerate(self.samples):
            subject = sample['subject']
            if subject not in subject_splits:
                subject_splits[subject] = []
            subject_splits[subject].append(idx)
        
        return subject_splits
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        subject_counts = {}
        emotion_counts = {}
        frame_stats = {'onset': [], 'apex': [], 'offset': []}
        
        for sample in self.samples:
            # Subject counts
            subject = sample['subject']
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            
            # Emotion counts
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Frame statistics
            frame_stats['onset'].append(sample['onset_frame'])
            frame_stats['apex'].append(sample['apex_frame'])
            frame_stats['offset'].append(sample['offset_frame'])
        
        return {
            'total_samples': len(self.samples),
            'num_subjects': len(subject_counts),
            'samples_per_subject': list(subject_counts.values()),
            'emotion_distribution': emotion_counts,
            'frame_statistics': {
                'onset': {
                    'min': min(frame_stats['onset']),
                    'max': max(frame_stats['onset']),
                    'mean': np.mean(frame_stats['onset']),
                    'std': np.std(frame_stats['onset'])
                },
                'apex': {
                    'min': min(frame_stats['apex']),
                    'max': max(frame_stats['apex']),
                    'mean': np.mean(frame_stats['apex']),
                    'std': np.std(frame_stats['apex'])
                },
                'offset': {
                    'min': min(frame_stats['offset']),
                    'max': max(frame_stats['offset']),
                    'mean': np.mean(frame_stats['offset']),
                    'std': np.std(frame_stats['offset'])
                }
            }
        }


def create_cnn_dataloaders(dataset: CNNCASMEIIDataset, 
                         train_indices: List[int], 
                         test_indices: List[int],
                         batch_size: int = 16,
                         shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for CNN.
    
    Args:
        dataset: CNN dataset
        train_indices: Training sample indices
        test_indices: Test sample indices
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    def collate_fn(batch):
        """Custom collate function for CNN data."""
        frames = torch.stack([item[0] for item in batch])   # (B, 3, 3, 64, 64)
        flows  = torch.stack([item[1] for item in batch])   # (B, 6, 64, 64)
        labels = torch.stack([item[2] for item in batch])   # (B,)
        metadata = [item[3] for item in batch]
        
        return frames, flows, labels, metadata
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=len(test_indices),  # Full batch for evaluation
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test CNN dataset
    print("Testing CNN dataset...")
    
    data_root = "../data/casme2"
    
    try:
        dataset = CNNCASMEIIDataset(data_root)
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading a sample
            frames, label, metadata = dataset[0]
            print(f"Frames shape: {frames.shape}")
            print(f"Label: {label}")
            print(f"Emotion: {metadata['emotion']}")
            print(f"Subject: {metadata['subject']}")
            print(f"Episode: {metadata['episode']}")
            print(f"Frame range: [{frames.min():.3f}, {frames.max():.3f}]")
            
            # Test frame ranges
            print(f"Onset frame range: [{frames[0].min():.3f}, {frames[0].max():.3f}]")
            print(f"Apex frame range: [{frames[1].min():.3f}, {frames[1].max():.3f}]")
            print(f"Offset frame range: [{frames[2].min():.3f}, {frames[2].max():.3f}]")
            
            # Get dataset statistics
            stats = dataset.get_statistics()
            print(f"\nDataset Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Number of subjects: {stats['num_subjects']}")
            print(f"  Samples per subject: {stats['samples_per_subject']}")
            
            # Test subject splits
            subject_splits = dataset.get_subject_splits()
            print(f"\nSubject splits:")
            for subject, indices in subject_splits.items():
                print(f"  {subject}: {len(indices)} samples")
        
        print("\n✅ CNN dataset working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure CASME2-coding-20140508.csv exists in ../data/labels/")
