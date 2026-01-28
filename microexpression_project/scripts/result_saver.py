#!/usr/bin/env python3
"""
Result Saver for Web Demo Predictions
Stores web demo analysis results to results/ directory
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import uuid

class ResultSaver:
    """Save and manage web demo analysis results"""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize result saver
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "json").mkdir(exist_ok=True)
        (self.results_dir / "csv").mkdir(exist_ok=True)
        (self.results_dir / "summaries").mkdir(exist_ok=True)
        
        # Initialize CSV file if not exists
        self.csv_file = self.results_dir / "csv" / "analysis_results.csv"
        self._init_csv_file()
    
    def _init_csv_file(self):
        """Initialize CSV file with headers"""
        if not self.csv_file.exists():
            headers = [
                'timestamp', 'analysis_id', 'filename', 'file_size_mb',
                'predicted_emotion', 'confidence', 'happiness_prob',
                'surprise_prob', 'disgust_prob', 'repression_prob',
                'processing_time', 'frames_processed', 'faces_detected',
                'model_type', 'inference_mode'
            ]
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def save_analysis_result(self, result: Dict[str, Any], video_info: Dict[str, Any]) -> str:
        """
        Save a single analysis result
        
        Args:
            result: Analysis result from model
            video_info: Video file information
            
        Returns:
            Analysis ID
        """
        # Generate unique ID
        analysis_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Prepare result data
        analysis_data = {
            'analysis_id': analysis_id,
            'timestamp': timestamp,
            'video_info': video_info,
            'prediction': result.get('prediction', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'all_probabilities': result.get('all_probabilities', {}),
            'au_contribution': result.get('au_contribution', {}),
            'preprocessing': result.get('preprocessing', 'Unknown'),
            'frame_info': result.get('frame_info', {}),
            'model_info': result.get('model_info', {}),
            'disclaimer': result.get('disclaimer', ''),
            'processing_time': video_info.get('processing_time', '0.0s')
        }
        
        # Save as JSON
        json_file = self.results_dir / "json" / f"analysis_{analysis_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        # Add to CSV
        self._add_to_csv(analysis_data, video_info)
        
        print(f"💾 Saved analysis result: {analysis_id}")
        return analysis_id
    
    def _add_to_csv(self, analysis_data: Dict[str, Any], video_info: Dict[str, Any]):
        """Add result to CSV file"""
        probs = analysis_data['all_probabilities']
        
        csv_row = [
            analysis_data['timestamp'],
            analysis_data['analysis_id'],
            video_info.get('filename', 'unknown'),
            video_info.get('file_size_mb', 0),
            analysis_data['prediction'],
            analysis_data['confidence'],
            probs.get('happiness', 0.0),
            probs.get('surprise', 0.0),
            probs.get('disgust', 0.0),
            probs.get('repression', 0.0),
            analysis_data['processing_time'],
            analysis_data['frame_info'].get('frames_processed', 0),
            analysis_data['frame_info'].get('faces_detected', 0),
            analysis_data['model_info'].get('model_type', 'Unknown'),
            analysis_data['model_info'].get('inference_mode', 'Unknown')
        ]
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
    
    def get_analysis_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent analysis history
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent analyses
        """
        json_dir = self.results_dir / "json"
        
        if not json_dir.exists():
            return []
        
        # Get all JSON files
        json_files = list(json_dir.glob("analysis_*.json"))
        
        # Sort by timestamp (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Load and return recent results
        results = []
        for json_file in json_files[:limit]:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"⚠️ Could not load {json_file}: {e}")
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics of all analyses"""
        try:
            # Read CSV data
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return {'error': 'No analysis data found'}
            
            # Calculate statistics
            total_analyses = len(rows)
            emotion_counts = {}
            confidence_sum = 0.0
            processing_times = []
            
            for row in rows:
                emotion = row['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                confidence_sum += float(row['confidence'])
                
                # Extract processing time in seconds
                time_str = row['processing_time'].replace('s', '')
                try:
                    processing_times.append(float(time_str))
                except:
                    pass
            
            avg_confidence = confidence_sum / total_analyses
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            summary = {
                'total_analyses': total_analyses,
                'emotion_distribution': emotion_counts,
                'average_confidence': avg_confidence,
                'average_processing_time': avg_processing_time,
                'date_range': {
                    'first': rows[0]['timestamp'] if rows else None,
                    'last': rows[-1]['timestamp'] if rows else None
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Save summary
            summary_file = self.results_dir / "summaries" / "analysis_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            return summary
            
        except Exception as e:
            return {'error': f'Could not generate summary: {e}'}
    
    def export_to_csv(self, output_path: str = None) -> str:
        """
        Export all results to CSV file
        
        Args:
            output_path: Custom output path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.results_dir / f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Copy existing CSV file
        import shutil
        shutil.copy2(self.csv_file, output_path)
        
        print(f"📄 Exported results to: {output_path}")
        return str(output_path)
    
    def cleanup_old_results(self, days: int = 30):
        """
        Clean up old analysis results
        
        Args:
            days: Remove results older than this many days
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        json_dir = self.results_dir / "json"
        
        if not json_dir.exists():
            return
        
        removed_count = 0
        for json_file in json_dir.glob("analysis_*.json"):
            if json_file.stat().st_mtime < cutoff_time:
                json_file.unlink()
                removed_count += 1
        
        print(f"🗑️ Cleaned up {removed_count} old results (older than {days} days)")

# Global result saver instance
result_saver = ResultSaver()

def save_analysis_result(result: Dict[str, Any], video_info: Dict[str, Any]) -> str:
    """
    Convenience function to save analysis result
    
    Args:
        result: Analysis result from model
        video_info: Video file information
        
    Returns:
        Analysis ID
    """
    return result_saver.save_analysis_result(result, video_info)

def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience function to get analysis history"""
    return result_saver.get_analysis_history(limit)

def generate_summary_report() -> Dict[str, Any]:
    """Convenience function to generate summary report"""
    return result_saver.generate_summary_report()

if __name__ == '__main__':
    # Test the result saver
    print("🔧 Testing Result Saver...")
    
    # Create test result
    test_result = {
        'prediction': 'Happiness',
        'confidence': 0.75,
        'all_probabilities': {
            'happiness': 0.75,
            'surprise': 0.10,
            'disgust': 0.05,
            'repression': 0.10
        },
        'preprocessing': 'Face detection + optical flow + CNN-SVM',
        'frame_info': {
            'frames_processed': 10,
            'faces_detected': 1
        },
        'model_info': {
            'model_type': 'Enhanced Hybrid CNN-SVM',
            'inference_mode': 'DEMO_PIPELINE'
        }
    }
    
    test_video_info = {
        'filename': 'test_video.avi',
        'file_size_mb': 5.2,
        'processing_time': '1.5s'
    }
    
    # Save test result
    analysis_id = save_analysis_result(test_result, test_video_info)
    print(f"✅ Saved test analysis: {analysis_id}")
    
    # Get history
    history = get_analysis_history(5)
    print(f"📊 Recent analyses: {len(history)}")
    
    # Generate summary
    summary = generate_summary_report()
    print(f"📈 Summary: {summary.get('total_analyses', 0)} total analyses")
    
    print("🎉 Result saver test complete!")
