#!/usr/bin/env python3
"""
Boosting Effect Logger - CRITICAL FIX #3

Logs pre/post boosting scores for validation and reviewer satisfaction.
This provides concrete evidence that AU boosting actually helps.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path


class BoostingEffectLogger:
    """
    Logs and analyzes the effect of AU soft boosting on predictions.
    
    CRITICAL for reviewer validation: "How do we know boosting helped?"
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize boosting effect logger.
        
        Args:
            log_file: Path to log file (auto-generated if None)
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"boosting_effects_{timestamp}.json"
        
        self.log_file = Path(log_file)
        self.session_logs = []
        self.session_start = datetime.now()
        
        print(f"📊 Boosting Effect Logger initialized")
        print(f"📁 Log file: {self.log_file}")
    
    def log_boosting_effect(self, sample_id: str, before_scores: np.ndarray, 
                           after_scores: np.ndarray, boosting_info: Dict,
                           true_label: Optional[int] = None,
                           predicted_label_before: Optional[int] = None,
                           predicted_label_after: Optional[int] = None) -> None:
        """
        Log the effect of boosting on a single sample.
        
        Args:
            sample_id: Unique sample identifier
            before_scores: Prediction scores before boosting
            after_scores: Prediction scores after boosting
            boosting_info: Boosting metadata
            true_label: Ground truth label (if available)
            predicted_label_before: Prediction before boosting
            predicted_label_after: Prediction after boosting
        """
        # Calculate effect metrics
        max_before = np.max(before_scores)
        max_after = np.max(after_scores)
        confidence_change = max_after - max_before
        
        # Prediction changes
        pred_before = np.argmax(before_scores) if predicted_label_before is None else predicted_label_before
        pred_after = np.argmax(after_scores) if predicted_label_after is None else predicted_label_after
        prediction_changed = pred_before != pred_after
        
        # Correctness (if ground truth available)
        correct_before = None
        correct_after = None
        improvement = None
        
        if true_label is not None:
            correct_before = (pred_before == true_label)
            correct_after = (pred_after == true_label)
            
            if correct_before and not correct_after:
                improvement = "worsened"
            elif not correct_before and correct_after:
                improvement = "improved"
            elif correct_before == correct_after:
                improvement = "unchanged"
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'boosting_applied': boosting_info.get('boosting_applied', False),
            'max_confidence_before': float(max_before),
            'max_confidence_after': float(max_after),
            'confidence_change': float(confidence_change),
            'prediction_before': int(pred_before),
            'prediction_after': int(pred_after),
            'prediction_changed': prediction_changed,
            'true_label': int(true_label) if true_label is not None else None,
            'correct_before': correct_before,
            'correct_after': correct_after,
            'improvement': improvement,
            'before_scores': before_scores.tolist(),
            'after_scores': after_scores.tolist(),
            'score_changes': (after_scores - before_scores).tolist(),
            'emotion_weights': boosting_info.get('emotion_weights', {}),
            'uncertainty_threshold': boosting_info.get('uncertainty_threshold', 0.6)
        }
        
        self.session_logs.append(log_entry)
    
    def analyze_session_effects(self) -> Dict:
        """
        Analyze the overall effects of boosting across the session.
        
        Returns:
            Analysis dictionary with key metrics
        """
        if not self.session_logs:
            return {'error': 'No logs to analyze'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.session_logs)
        
        # Basic statistics
        total_samples = len(df)
        boosted_samples = df['boosting_applied'].sum()
        prediction_changes = df['prediction_changed'].sum()
        
        # Confidence analysis
        avg_confidence_change = df['confidence_change'].mean()
        confidence_improvements = (df['confidence_change'] > 0).sum()
        confidence_decreases = (df['confidence_change'] < 0).sum()
        
        # Accuracy analysis (if ground truth available)
        accuracy_analysis = {}
        if 'correct_before' in df.columns and df['correct_before'].notna().any():
            before_accuracy = df['correct_before'].mean()
            after_accuracy = df['correct_after'].mean()
            accuracy_improvement = after_accuracy - before_accuracy
            
            improvements = df[df['improvement'] == 'improved']
            worsened = df[df['improvement'] == 'worsened']
            unchanged = df[df['improvement'] == 'unchanged']
            
            accuracy_analysis = {
                'accuracy_before': float(before_accuracy),
                'accuracy_after': float(after_accuracy),
                'accuracy_improvement': float(accuracy_improvement),
                'samples_improved': len(improvements),
                'samples_worsened': len(worsened),
                'samples_unchanged': len(unchanged),
                'improvement_rate': float(len(improvements) / len(df[df['improvement'].notna()])),
                'worsening_rate': float(len(worsened) / len(df[df['improvement'].notna()]))
            }
        
        # Emotion-specific analysis
        emotion_analysis = {}
        for emotion_idx in range(4):  # 4 emotions
            emotion_mask = df['prediction_before'] == emotion_idx
            if emotion_mask.any():
                emotion_df = df[emotion_mask]
                emotion_changes = emotion_df['prediction_changed'].sum()
                emotion_total = len(emotion_df)
                
                emotion_analysis[f'emotion_{emotion_idx}'] = {
                    'total_samples': emotion_total,
                    'prediction_changes': emotion_changes,
                    'change_rate': float(emotion_changes / emotion_total),
                    'avg_confidence_change': float(emotion_df['confidence_change'].mean())
                }
        
        analysis = {
            'session_summary': {
                'total_samples': total_samples,
                'boosted_samples': int(boosted_samples),
                'boosting_rate': float(boosted_samples / total_samples),
                'prediction_changes': int(prediction_changes),
                'prediction_change_rate': float(prediction_changes / total_samples),
                'avg_confidence_change': float(avg_confidence_change),
                'confidence_improvements': int(confidence_improvements),
                'confidence_decreases': int(confidence_decreases),
                'session_duration': str(datetime.now() - self.session_start)
            },
            'accuracy_analysis': accuracy_analysis,
            'emotion_analysis': emotion_analysis,
            'recommendation': self._generate_recommendation(df)
        }
        
        return analysis
    
    def _generate_recommendation(self, df: pd.DataFrame) -> str:
        """Generate recommendation based on analysis."""
        if len(df) == 0:
            return "No data to analyze"
        
        boosting_rate = df['boosting_applied'].mean()
        accuracy_improvement = 0.0
        
        if 'improvement' in df.columns:
            improved = len(df[df['improvement'] == 'improved'])
            total_with_ground_truth = len(df[df['improvement'].notna()])
            if total_with_ground_truth > 0:
                accuracy_improvement = improved / total_with_ground_truth
        
        if boosting_rate < 0.3:
            return "Low boosting rate - consider lowering uncertainty threshold"
        elif boosting_rate > 0.8:
            return "High boosting rate - consider increasing uncertainty threshold"
        elif accuracy_improvement > 0.1:
            return "Good improvement - boosting is effective"
        elif accuracy_improvement < -0.05:
            return "Negative impact - consider reducing lambda_boost"
        else:
            return "Moderate impact - fine-tune parameters"
    
    def save_logs(self) -> None:
        """Save session logs to file."""
        # Save detailed logs
        with open(self.log_file, 'w') as f:
            json.dump({
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_samples': len(self.session_logs)
                },
                'logs': self.session_logs,
                'analysis': self.analyze_session_effects()
            }, f, indent=2)
        
        print(f"💾 Logs saved to: {self.log_file}")
    
    def create_reviewer_table(self) -> pd.DataFrame:
        """
        Create reviewer-friendly table showing pre/post boosting effects.
        
        Returns:
            DataFrame with reviewer-friendly format
        """
        if not self.session_logs:
            return pd.DataFrame()
        
        # Create reviewer table
        reviewer_data = []
        for log in self.session_logs:
            reviewer_data.append({
                'Sample': log['sample_id'],
                'Before': f"{log['prediction_before']} ({log['max_confidence_before']:.3f})",
                'After': f"{log['prediction_after']} ({log['max_confidence_after']:.3f})",
                'Correct': log['improvement'] if log['improvement'] else 'N/A',
                'Boosted': '✅' if log['boosting_applied'] else '❌',
                'Change': f"{log['confidence_change']:+.3f}"
            })
        
        df = pd.DataFrame(reviewer_data)
        
        # Save reviewer table
        reviewer_file = self.log_file.parent / f"reviewer_table_{self.log_file.stem}.csv"
        df.to_csv(reviewer_file, index=False)
        
        print(f"📊 Reviewer table saved to: {reviewer_file}")
        return df
    
    def print_summary(self) -> None:
        """Print summary of boosting effects."""
        analysis = self.analyze_session_effects()
        
        if 'error' in analysis:
            print("❌ No data to analyze")
            return
        
        summary = analysis['session_summary']
        accuracy = analysis.get('accuracy_analysis', {})
        
        print("\n" + "="*60)
        print("📊 BOOSTING EFFECTS SUMMARY")
        print("="*60)
        print(f"📈 Total Samples: {summary['total_samples']}")
        print(f"🚀 Boosting Rate: {summary['boosting_rate']:.1%}")
        print(f"🔄 Prediction Changes: {summary['prediction_change_rate']:.1%}")
        print(f"📊 Avg Confidence Change: {summary['avg_confidence_change']:+.3f}")
        
        if accuracy:
            print(f"\n🎯 ACCURACY IMPACT:")
            print(f"   Before: {accuracy['accuracy_before']:.1%}")
            print(f"   After: {accuracy['accuracy_after']:.1%}")
            print(f"   Improvement: {accuracy['accuracy_improvement']:+.1%}")
            print(f"   Improved: {accuracy['samples_improved']} samples")
            print(f"   Worsened: {accuracy['samples_worsened']} samples")
        
        print(f"\n💡 Recommendation: {analysis['recommendation']}")
        print("="*60)


# Integration utility
def create_boosting_logger(log_dir: str = "results") -> BoostingEffectLogger:
    """
    Factory function to create boosting logger.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured BoostingEffectLogger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    return BoostingEffectLogger()


if __name__ == "__main__":
    # Test the logger
    print("🧪 Testing Boosting Effect Logger...")
    
    logger = create_boosting_logger()
    
    # Simulate some boosting effects
    for i in range(10):
        before_scores = np.array([0.3, 0.2, 0.4, 0.1])
        after_scores = np.array([0.35, 0.15, 0.45, 0.05])  # Boosted
        
        boosting_info = {
            'boosting_applied': True,
            'max_confidence': 0.4,
            'uncertainty_threshold': 0.6,
            'emotion_weights': {'happiness': 0.1, 'disgust': 0.05, 'surprise': 0.2, 'repression': 0.0}
        }
        
        logger.log_boosting_effect(
            sample_id=f"sample_{i}",
            before_scores=before_scores,
            after_scores=after_scores,
            boosting_info=boosting_info,
            true_label=2,  # surprise
            predicted_label_before=2,
            predicted_label_after=2
        )
    
    # Print summary
    logger.print_summary()
    
    # Save logs
    logger.save_logs()
    
    # Create reviewer table
    reviewer_df = logger.create_reviewer_table()
    print(f"\n📊 Reviewer table shape: {reviewer_df.shape}")
    
    print("🎉 Boosting Effect Logger test complete!")
