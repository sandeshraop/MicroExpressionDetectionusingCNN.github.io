#!/usr/bin/env python3
"""
Demo script for micro-expression inference pipeline.

This script demonstrates how to:
1. Load a trained model
2. Process video files
3. Predict emotions with AU contribution analysis
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from enhanced_inference_pipeline import EnhancedMicroExpressionInferencePipeline


def main():
    parser = argparse.ArgumentParser(description='Micro-Expression Inference Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file or directory with videos')
    parser.add_argument('--output', type=str, default='inference_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=== Micro-Expression Inference Demo ===")
    
    # Initialize pipeline
    print(f"Initializing pipeline on {args.device}...")
    pipeline = EnhancedMicroExpressionInferencePipeline(args.model)
    
    print(f"✅ Model loaded successfully from {args.model}")
    
    # Process video(s)
    video_path = Path(args.video)
    
    if video_path.is_file():
        # Single video
        print(f"Processing single video: {video_path}")
        result = pipeline.predict_emotion(str(video_path))
        
        if result['success']:
            print(f"\n✅ Prediction Results:")
            print(f"  Emotion: {result['predicted_emotion']}")
            print(f"  Confidence: {result['relative_probability']:.3f}")
            print(f"  Most Active AU: {result['au_contribution']['most_active_au']}")
            
            print(f"\n📊 All Probabilities:")
            for emotion, prob in result['all_probabilities'].items():
                print(f"  {emotion}: {prob:.3f}")
            
            print(f"\n🎯 AU Contribution Analysis:")
            for au, data in list(result['au_contribution']['au_rankings'].items())[:3]:
                print(f"  {au}: {data['activity_score']:.3f}")
            
            # Save result
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
            
        else:
            print(f"❌ Error: {result['error']}")
    
    elif video_path.is_dir():
        # Multiple videos
        print(f"Processing videos in directory: {video_path}")
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_path.glob(f'*{ext}'))
            video_files.extend(video_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No video files found in {video_path}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # Process all videos
        results = pipeline.batch_predict([str(vf) for vf in video_files])
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n✅ Processed {successful}/{len(results)} videos successfully")
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")
        
        # Show summary
        if successful > 0:
            successful_results = [r for r in results if r['success']]
            emotion_counts = {}
            for result in successful_results:
                emotion = result['predicted_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"\n📊 Emotion Distribution:")
            for emotion, count in emotion_counts.items():
                print(f"  {emotion}: {count}")
    
    else:
        print(f"Error: {args.video} is not a valid file or directory")


if __name__ == "__main__":
    main()
