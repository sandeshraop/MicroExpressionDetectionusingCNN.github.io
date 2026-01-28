#!/usr/bin/env python3
"""
Proper LOSO (Leave-One-Subject-Out) Evaluation
Integrates with existing training pipeline for publication-ready results
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from micro_expression_model import EnhancedHybridModel
from dataset_loader import CNNCASMEIIDataset
from config import EMOTION_LABELS, LABEL_TO_EMOTION, EMOTION_DISPLAY_ORDER
from train_augmented import AugmentedTrainer, AugmentedDataset


class ProperLOSOEvaluator:
    """Proper LOSO evaluation using existing training pipeline"""
    
    def __init__(self, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Proper LOSO Evaluator initialized on {self.device}")
    
    def get_all_subjects(self, dataset_path):
        """Get all unique subjects from dataset"""
        subjects = set()
        data_root = Path(dataset_path)
        
        for subject_dir in data_root.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
                subjects.add(subject_dir.name)
        
        return sorted(list(subjects))
    
    def create_loso_datasets(self, dataset_path, test_subject, labels_file):
        """Create LOSO train/test datasets"""
        print(f"🎯 Creating LOSO split - Test Subject: {test_subject}")
        
        # Load full dataset
        full_dataset = CNNCASMEIIDataset(dataset_path, labels_file)
        
        if len(full_dataset) == 0:
            print(f"❌ No samples loaded from dataset")
            return None, None
        
        # Split by subject
        train_samples = []
        test_samples = []
        
        for i in range(len(full_dataset)):
            frames, flows, label, metadata = full_dataset[i]
            
            # Extract subject from metadata or video path
            subject = metadata.get('subject', 'unknown')
            
            sample = (frames, flows, label, metadata)
            
            if subject == test_subject:
                test_samples.append(sample)
            else:
                train_samples.append(sample)
        
        print(f"📊 Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
        
        if len(test_samples) == 0:
            print(f"⚠️ No test samples for subject {test_subject}")
            return None, None
        
        return train_samples, test_samples
    
    def train_model_on_samples(self, train_samples):
        """Train model on training samples using existing pipeline"""
        print(f"🔄 Training model on {len(train_samples)} samples...")
        
        # Create augmented dataset from training samples
        augmented_dataset = AugmentedDataset(train_samples, augmentation_factor=3)
        
        if len(augmented_dataset) == 0:
            print(f"❌ No augmented samples created")
            return None
        
        # Train using existing trainer
        trainer = AugmentedTrainer(device=self.device)
        
        try:
            model, accuracy, report = trainer.train_augmented_model(
                augmented_dataset,
                epochs=12,
                learning_rate=0.001
            )
            print(f"✅ Model trained successfully")
            return model
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def evaluate_model_on_samples(self, model, test_samples):
        """Evaluate trained model on test samples"""
        print(f"🧪 Evaluating model on {len(test_samples)} samples...")
        
        predictions = []
        true_labels = []
        probabilities = []
        
        model.feature_extractor.to('cpu')
        
        for frames, flows, label, metadata in test_samples:
            try:
                # Prepare tensors for inference
                # Handle temporal frames
                if frames.dim() == 4:  # (T, C, H, W)
                    # Aggregate temporal features
                    temporal_features = []
                    for t in range(frames.shape[0]):
                        single_frame = frames[t:t+1].cpu()
                        single_flow = flows.unsqueeze(0).cpu()
                        
                        # Extract features
                        frame_features = model.extract_all_features(single_frame, single_flow)
                        temporal_features.append(frame_features)
                    
                    # Aggregate temporal features
                    features = np.mean(temporal_features, axis=0)
                else:
                    frames_cpu = frames.cpu()
                    flows_cpu = flows.cpu()
                    features = model.extract_all_features(frames_cpu, flows_cpu)
                
                # Predict
                pred_idx = model.pipeline.predict(features)[0]
                pred_proba = model.pipeline.predict_proba(features)[0]
                
                predictions.append(pred_idx)
                true_labels.append(label.item())
                probabilities.append(pred_proba)
                
            except Exception as e:
                print(f"⚠️ Error processing sample: {e}")
                continue
        
        if len(predictions) == 0:
            print(f"❌ No successful predictions")
            return None
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        probabilities = np.array(probabilities)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class recall
        per_class_recall = {}
        for i, emotion in enumerate(['happiness', 'surprise', 'disgust', 'repression']):
            mask = (true_labels == i)
            if mask.sum() > 0:
                recall = np.mean(predictions[mask] == i)
                per_class_recall[emotion] = recall
            else:
                per_class_recall[emotion] = 0.0
        
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'accuracy': accuracy,
            'per_class_recall': per_class_recall
        }
    
    def evaluate_single_subject(self, dataset_path, test_subject, labels_file):
        """Evaluate LOSO for single subject"""
        print(f"\n{'='*60}")
        print(f"🎯 LOSO Evaluation - Test Subject: {test_subject}")
        print(f"{'='*60}")
        
        # Create LOSO split
        train_samples, test_samples = self.create_loso_datasets(dataset_path, test_subject, labels_file)
        
        if train_samples is None or test_samples is None:
            return None
        
        # Train model
        model = self.train_model_on_samples(train_samples)
        if model is None:
            return None
        
        # Evaluate model
        results = self.evaluate_model_on_samples(model, test_samples)
        if results is None:
            return None
        
        results['test_subject'] = test_subject
        results['train_samples'] = len(train_samples)
        results['test_samples'] = len(test_samples)
        
        print(f"✅ {test_subject}: Accuracy = {results['accuracy']:.3f}")
        print(f"   Per-class recall: {results['per_class_recall']}")
        
        return results
    
    def run_full_loso_evaluation(self, dataset_path, labels_file):
        """Run complete LOSO evaluation"""
        print("🚀 Starting Proper LOSO Evaluation...")
        print(f"📁 Dataset: {dataset_path}")
        print(f"📄 Labels: {labels_file}")
        
        # Get all subjects
        subjects = self.get_all_subjects(dataset_path)
        print(f"👥 Found {len(subjects)} subjects: {subjects}")
        
        if len(subjects) < 2:
            print("❌ Need at least 2 subjects for LOSO evaluation")
            return None
        
        # Run LOSO for each subject
        all_results = []
        subject_results = {}
        
        for i, test_subject in enumerate(subjects, 1):
            print(f"\n📊 Progress: {i}/{len(subjects)}")
            
            result = self.evaluate_single_subject(dataset_path, test_subject, labels_file)
            if result:
                all_results.append(result)
                subject_results[test_subject] = result
            else:
                print(f"❌ {test_subject}: Evaluation failed")
        
        if not all_results:
            print("❌ No successful evaluations")
            return None
        
        # Aggregate results
        return self._aggregate_results(all_results, subject_results, subjects)
    
    def _aggregate_results(self, all_results, subject_results, subjects):
        """Aggregate LOSO results"""
        print(f"\n{'='*60}")
        print("📊 AGGREGATED LOSO RESULTS")
        print(f"{'='*60}")
        
        # Collect all predictions
        all_predictions = np.concatenate([r['predictions'] for r in all_results])
        all_true_labels = np.concatenate([r['true_labels'] for r in all_results])
        all_probabilities = np.concatenate([r['probabilities'] for r in all_results])
        
        # Overall accuracy
        overall_accuracy = np.mean(all_predictions == all_true_labels)
        
        # Per-class recall
        per_class_recall = {}
        per_class_predictions = {}
        
        for i, emotion in enumerate(['happiness', 'surprise', 'disgust', 'repression']):
            mask = (all_true_labels == i)
            if mask.sum() > 0:
                recall = np.mean(all_predictions[mask] == i)
                per_class_recall[emotion] = recall
                
                # Count predictions
                pred_mask = (all_predictions == i)
                per_class_predictions[emotion] = pred_mask.sum()
            else:
                per_class_recall[emotion] = 0.0
                per_class_predictions[emotion] = 0
        
        # Calculate UAR (Unweighted Average Recall)
        uar = np.mean(list(per_class_recall.values()))
        
        # Classification report
        class_report = classification_report(
            all_true_labels, all_predictions,
            target_names=['happiness', 'surprise', 'disgust', 'repression'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        
        # Print results
        print(f"\n🎯 FINAL LOSO PERFORMANCE METRICS:")
        print(f"   Overall Accuracy: {overall_accuracy:.3f}")
        print(f"   UAR: {uar:.3f}")
        print(f"\n📈 Per-Class Recall:")
        for emotion, recall in per_class_recall.items():
            print(f"   {emotion:10s}: {recall:.3f}")
        
        print(f"\n📊 Per-Class Predictions:")
        total_preds = sum(per_class_predictions.values())
        for emotion, count in per_class_predictions.items():
            percentage = (count / total_preds * 100) if total_preds > 0 else 0
            print(f"   {emotion:10s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n📋 Detailed Classification Report:")
        for emotion in ['happiness', 'surprise', 'disgust', 'repression']:
            if emotion in class_report:
                metrics = class_report[emotion]
                print(f"   {emotion:10s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print(f"   Predicted →")
        print(f"   Actual ↓  Happy  Surprise  Disgust  Repression")
        emotions = ['Happy', 'Surprise', 'Disgust', 'Repression']
        for i, emotion in enumerate(emotions):
            print(f"   {emotion:10s} {conf_matrix[i, 0]:6d} {conf_matrix[i, 1]:8d} {conf_matrix[i, 2]:7d} {conf_matrix[i, 3]:10d}")
        
        # Subject-wise results
        print(f"\n👥 Subject-wise Results:")
        print(f"   Subject    Accuracy  Samples")
        print(f"   -------    --------  -------")
        for subject in subjects:
            if subject in subject_results:
                result = subject_results[subject]
                print(f"   {subject:10s} {result['accuracy']:8.3f} {result['test_samples']:8d}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"../models/loso_results_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'dataset_path': str(dataset_path),
            'evaluation_type': 'LOSO',
            'num_subjects': len(subjects),
            'num_successful_evaluations': len(all_results),
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'uar': uar,
                'per_class_recall': per_class_recall,
                'per_class_predictions': per_class_predictions
            },
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'subject_results': subject_results,
            'total_samples': len(all_predictions)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        return results_data


def main():
    """Main function to run proper LOSO evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Proper LOSO Evaluation for Micro-Expression Recognition')
    parser.add_argument('--dataset', type=str, 
                       default='../data/casme2',
                       help='Path to CASME-II dataset')
    parser.add_argument('--labels', type=str,
                       default='../data/labels/casme2_labels.csv',
                       help='Path to labels file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Check dataset exists
    dataset_path = Path(args.dataset)
    labels_path = Path(args.labels)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        return
    
    # Run LOSO evaluation
    evaluator = ProperLOSOEvaluator(device=args.device)
    results = evaluator.run_full_loso_evaluation(dataset_path, labels_path)
    
    if results:
        print(f"\n🎉 LOSO Evaluation completed successfully!")
        print(f"📊 Overall Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        print(f"📊 UAR: {results['overall_metrics']['uar']:.3f}")
        
        # Summary for publication
        print(f"\n📋 PUBLICATION-READY SUMMARY:")
        print(f"   Method: Leave-One-Subject-Out (LOSO)")
        print(f"   Subjects: {results['num_subjects']}")
        print(f"   Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        print(f"   UAR: {results['overall_metrics']['uar']:.3f}")
        print(f"   Per-class Recall: {results['overall_metrics']['per_class_recall']}")
    else:
        print(f"\n❌ LOSO Evaluation failed!")


if __name__ == '__main__':
    main()
