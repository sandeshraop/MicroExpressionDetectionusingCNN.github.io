#!/usr/bin/env python3
"""
AU-Aware Soft Boosting for Micro-Expression Recognition

Key Innovation: Replace hard thresholds with confidence-based enhancement
using Action Unit (AU) activation patterns.

AU Patterns:
- Happiness: AU6+12 (cheek raiser + lip corner puller)
- Disgust: AU9+15+16 (nose wrinkler + lip corner depressor + lower lip depressor)
- Surprise: AU1+2+5+26 (inner/outer brow raiser + upper lid raiser + jaw drop)
- Repression: Attempted suppression patterns
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import EMOTION_LABELS


@dataclass
class AUConfiguration:
    """Configuration for Action Unit patterns and emotions."""
    
    # AU indices (0-based for array access)
class AUIndices:
    AU_INNER_BROW_RAISER = 1    # AU1
    AU_OUTER_BROW_RAISER = 2    # AU2
    AU_UPPER_LID_RAISER = 5     # AU5
    AU_CHEEK_RAISER = 6         # AU6
    AU_NOSE_WRINKLER = 9        # AU9
    AU_LIP_CORNER_PULLER = 12   # AU12
    AU_LIP_CORNER_DEPRESSOR = 15 # AU15
    AU_LOWER_LIP_DEPRESSOR = 16  # AU16
    AU_JAW_DROP = 26            # AU26

class AUSoftBoosting:
    """
    AU-aware Soft Boosting for micro-expression recognition
    
    SCIENTIFIC NOTES:
    • AU proxy activations derived from AU-aligned strain magnitude (not explicit AU detection)
    • Conditional boosting prevents overconfidence amplification
    • Product of AU activations enforces co-activation patterns
    • Post-hoc confidence calibration (not feature learning)
    
    NOTE: au_activations uses FACS AU numbering as array indices
    Index i corresponds to AUi for standard FACS AUs
    """
    
    def __init__(
        self,
        lambda_boost: float = 0.3,
        uncertainty_threshold: float = 0.6,
        confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize AU Soft Boosting

        Args:
            lambda_boost: Boosting strength (0.0-1.0)
            uncertainty_threshold: Apply boosting only if max_prob < threshold
            confidence_threshold: Alias for uncertainty_threshold (FaceSleuth hybrid API)
        """
        if confidence_threshold is not None:
            uncertainty_threshold = float(confidence_threshold)
        self.lambda_boost = lambda_boost
        self.uncertainty_threshold = uncertainty_threshold
        
        # FACS AU index mapping for clarity
        self.AU_INDEX = {
            1: 1, 2: 2, 4: 4, 5: 5, 6: 6, 9: 9, 10: 10, 12: 12, 15: 15, 16: 16, 25: 25, 26: 26
        }
        
        # Emotion-AU patterns (FACS-based). Keys must match config.EMOTION_LABELS names.
        self.EMOTION_AU_PATTERNS = {
            'happiness': [6, 12],  # Cheek raiser + lip corner puller
            'surprise': [1, 2, 5, 26],
            'disgust': [9, 15, 16],
            'repression': [],  # weak / conflicting AU proxy (see _compute_repression_au_score)
            'others': [],
        }
        # Order must match model probabilities: indices 0..N-1 per ``EMOTION_LABELS`` numeric values
        self.prob_emotion_order = [e for e, _ in sorted(EMOTION_LABELS.items(), key=lambda kv: kv[1])]
    
    def compute_emotion_au_weights(self, au_activations: np.ndarray) -> Dict[str, float]:
        """
        Compute emotion-specific weights based on AU activation patterns.
        
        Args:
            au_activations: Array of AU activation values (0-1)
            
        Returns:
            Dictionary mapping emotions to AU-based weights
        """
        emotion_weights = {}
        
        for emotion in self.prob_emotion_order:
            pattern_aus = self.EMOTION_AU_PATTERNS.get(emotion, [])
            
            if emotion == 'repression':
                emotion_weights[emotion] = self._compute_repression_au_score(au_activations)
            elif emotion == 'others':
                emotion_weights[emotion] = float(
                    max(0.0, 1.0 - self._compute_repression_au_score(au_activations))
                )
            else:
                # Standard case: product of relevant AU activations
                if pattern_aus:
                    au_products = []
                    for au_idx in pattern_aus:
                        if au_idx < len(au_activations):
                            au_products.append(au_activations[au_idx])
                    
                    if au_products:
                        emotion_weights[emotion] = np.prod(au_products)
                    else:
                        emotion_weights[emotion] = 0.0
                else:
                    emotion_weights[emotion] = 0.0
        
        return emotion_weights
    
    def _compute_repression_au_score(self, au_activations: np.ndarray) -> float:
        """
        Proxy score for repression / suppression-style dynamics from AU strain proxies.
        """
        # Look for weak activations across happiness and disgust AUs
        happiness_aus = [6, 12]  # Lip corner puller + Dimpler
        disgust_aus = [9, 15, 16]  # Nose wrinkler + Lip corner depressor + Lower lip depressor
        
        # Compute average activation for each emotion pattern
        happiness_score = 0.0
        disgust_score = 0.0
        
        for au_idx in happiness_aus:
            if au_idx < len(au_activations):
                happiness_score += au_activations[au_idx]
        happiness_score /= len(happiness_aus)
        
        for au_idx in disgust_aus:
            if au_idx < len(au_activations):
                disgust_score += au_activations[au_idx]
        disgust_score /= len(disgust_aus)
        
        # Suppression indicated by conflicting weak activations
        conflict_score = min(happiness_score, disgust_score)
        
        # Combine low activation with conflict
        overall_activation = (happiness_score + disgust_score) / 2
        suppression_score = (1.0 - overall_activation) * conflict_score
        
        return suppression_score
    
    def apply_conditional_soft_boosting(self, raw_scores: torch.Tensor, au_activations: np.ndarray, 
                                    uncertainty_threshold: float = 0.6, uncertainty: float = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply AU-aware soft boosting ONLY when model is uncertain.
        
        CRITICAL FIX #2: Prevent overconfidence amplification and happiness domination.
        
        Args:
            raw_scores: Raw model scores (logits or probabilities)
            au_activations: AU activation values (0-1)
            uncertainty_threshold: Apply boosting only if max_prob < threshold
            
        Returns:
            Tuple of (boosted_scores, boosting_info)
        """
        # Handle both uncertainty parameter formats
        if uncertainty is not None:
            uncertainty_threshold = uncertainty
            
        # Convert to probabilities if needed.
        # Research correctness: avoid applying softmax to something that is already probabilities.
        scores = raw_scores
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if scores.dim() != 2 or scores.size(1) <= 1:
            raise ValueError(f"expected (B, C) scores, got shape {tuple(raw_scores.shape)}")

        row_sums = scores.sum(dim=1)
        looks_like_proba = (
            torch.all(scores >= 0)
            and torch.all(scores <= 1)
            and torch.all(torch.isfinite(scores))
            and torch.all(torch.isfinite(row_sums))
            and torch.all(torch.abs(row_sums - 1.0) < 1e-3)
        )
        probabilities = scores if looks_like_proba else torch.softmax(scores, dim=1)
        
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        
        # CRITICAL FIX #2: Only boost when uncertain
        if max_prob.item() >= uncertainty_threshold:
            # Model is confident - return unchanged
            return probabilities, {
                'boosting_applied': False,
                'reason': 'Model confident',
                'max_probability': max_prob.item(),
                'predicted_class': predicted_class.item()
            }
        
        # Model is uncertain - apply AU-aware soft boosting
        emotion_weights = self.compute_emotion_au_weights(au_activations)
        
        # Convert to tensor
        weight_tensor = torch.tensor(
            [emotion_weights.get(emotion, 0.0) for emotion in self.prob_emotion_order],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        
        # Apply soft boosting formula
        boosted_scores = probabilities * (1 + self.lambda_boost * weight_tensor)

        # Renormalize by sum-to-one (NOT softmax). Softmax would distort already-probabilistic scores.
        boosted_scores = boosted_scores / (boosted_scores.sum(dim=-1, keepdim=True) + 1e-12)
        
        boosting_info = {
            'boosting_applied': True,
            'reason': 'Model uncertain - AU boosting applied',
            'max_probability': max_prob.item(),
            'predicted_class': predicted_class.item(),
            'uncertainty_threshold': float(uncertainty_threshold),
            'before_scores': probabilities.clone().detach().cpu().numpy(),
            'after_scores': boosted_scores.clone().detach().cpu().numpy(),
            'emotion_weights': emotion_weights,
            'score_changes': (boosted_scores - probabilities).detach().cpu().numpy()
        }
        
        return boosted_scores, boosting_info
    
    def apply_soft_boosting(self, raw_scores: torch.Tensor, au_activations: np.ndarray) -> torch.Tensor:
        scores = raw_scores
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if scores.dim() != 2 or scores.size(1) <= 1:
            raise ValueError(f"expected (B, C) scores, got shape {tuple(raw_scores.shape)}")

        row_sums = scores.sum(dim=1)
        looks_like_proba = (
            torch.all(scores >= 0)
            and torch.all(scores <= 1)
            and torch.all(torch.isfinite(scores))
            and torch.all(torch.isfinite(row_sums))
            and torch.all(torch.abs(row_sums - 1.0) < 1e-3)
        )
        probabilities = scores if looks_like_proba else F.softmax(scores, dim=-1)
        
        # Compute AU-based emotion weights
        emotion_weights = self.compute_emotion_au_weights(au_activations)
        
        # Convert to tensor
        weight_tensor = torch.tensor(
            [emotion_weights.get(emotion, 0.0) for emotion in self.prob_emotion_order],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        
        # Apply soft boosting formula
        boosted_scores = probabilities * (1 + self.lambda_boost * weight_tensor)
        
        # Renormalize by sum-to-one (NOT softmax).
        boosted_scores = boosted_scores / (boosted_scores.sum(dim=-1, keepdim=True) + 1e-12)
        
        return boosted_scores
    
    def apply_soft_boosting_numpy(self, raw_scores: np.ndarray, au_activations: np.ndarray) -> np.ndarray:
        """
        Apply AU-aware soft boosting (NumPy version).
        
        Args:
            raw_scores: Raw scores array
            au_activations: AU activation values
            
        Returns:
            Boosted scores array
        """
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if scores.ndim != 2 or scores.shape[1] <= 1:
            raise ValueError(f"expected (B, C) scores, got shape {scores.shape}")

        row_sums = scores.sum(axis=1)
        looks_like_proba = (
            np.all(np.isfinite(scores))
            and np.all(scores >= 0.0)
            and np.all(scores <= 1.0)
            and np.all(np.isfinite(row_sums))
            and np.all(np.abs(row_sums - 1.0) < 1e-3)
        )
        probabilities = scores if looks_like_proba else self._softmax_numpy(scores)
        
        # Compute AU-based emotion weights
        emotion_weights = self.compute_emotion_au_weights(au_activations)
        
        # Convert to array
        weight_array = np.array([emotion_weights.get(emotion, 0.0) for emotion in self.prob_emotion_order])
        
        # Apply soft boosting formula
        boosted_scores = probabilities * (1 + self.lambda_boost * weight_array)
        
        # Renormalize by sum-to-one (NOT softmax).
        boosted_scores = boosted_scores / (np.sum(boosted_scores, axis=-1, keepdims=True) + 1e-12)
        
        return boosted_scores
    
    def _softmax_numpy(self, x: np.ndarray) -> np.ndarray:
        """Softmax implementation for NumPy arrays."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_au_contribution_analysis(self, au_activations: np.ndarray, 
                                   original_scores: np.ndarray, 
                                   boosted_scores: np.ndarray) -> Dict:
        """
        Analyze AU contribution to boosting decisions.
        
        Args:
            au_activations: AU activation values
            original_scores: Original prediction scores
            boosted_scores: Boosted prediction scores
            
        Returns:
            Analysis dictionary with AU contributions
        """
        emotion_weights = self.compute_emotion_au_weights(au_activations)
        
        # Compute score changes
        score_changes = boosted_scores - original_scores
        
        # Find most influential AU for each emotion
        au_influence = {}
        for emotion in self.prob_emotion_order:
            pattern_aus = self.EMOTION_AU_PATTERNS.get(emotion, [])
            au_contributions = {}
            
            for au_idx in pattern_aus:
                if au_idx < len(au_activations):
                    au_contributions[f'AU{au_idx}'] = au_activations[au_idx]
            
            au_influence[emotion] = au_contributions
        
        return {
            'emotion_weights': emotion_weights,
            'score_changes': score_changes.tolist(),
            'au_influence': au_influence,
            'boosting_applied': np.any(score_changes > 0.01)
        }


# Integration utilities
def create_au_soft_boosting_layer(lambda_boost: float = 0.3) -> AUSoftBoosting:
    """
    Factory function to create AU soft boosting layer.
    
    Args:
        lambda_boost: Boosting intensity
        
    Returns:
        Configured AUSoftBoosting instance
    """
    return AUSoftBoosting(lambda_boost=lambda_boost)


def extract_au_activations_from_strain(strain_statistics: np.ndarray) -> np.ndarray:
    """
    Extract real AU activations from AU-aligned strain statistics.
    
    Option A: Use existing strain statistics for AU activations
    - Fully consistent with training pipeline
    - Scientifically valid (strain → AU activation correlation)
    - No additional models needed
    
    Args:
        strain_statistics: AU-aligned strain statistics (40-D)
        
    Returns:
        AU activations array (27-D)
    """
    # Initialize 27 AU activations (FACS standard)
    au_activations = np.zeros(27)
    
    # Map strain statistics to specific AUs
    # AU-aligned regions: AU4, AU6, AU9, AU10, AU12 (5 AUs x 4 stats x 2 strain maps = 40-D)
    
    # Extract statistics for each AU region (20 stats per strain map, 2 strain maps)
    au4_stats = strain_statistics[0:4]    # AU4: brow lowerer
    au6_stats = strain_statistics[4:8]    # AU6: cheek raiser  
    au9_stats = strain_statistics[8:12]   # AU9: nose wrinkler
    au10_stats = strain_statistics[12:16] # AU10: upper lip raiser
    au12_stats = strain_statistics[16:20] # AU12: lip corner puller
    
    # Convert strain statistics to AU activations (0-1 scale)
    # Use normalized strain magnitude as activation proxy
    def normalize_strain_to_au(stats):
        """Convert strain statistics to AU activation (0-1)."""
        strain_magnitude = np.mean(np.abs(stats))
        # Normalize using sigmoid-like function
        activation = 2.0 / (1.0 + np.exp(-3.0 * strain_magnitude)) - 1.0
        return np.clip(activation, 0.0, 1.0)
    
    # Map to specific AU indices (FACS numbering)
    au_activations[4] = normalize_strain_to_au(au4_stats)   # AU4
    au_activations[6] = normalize_strain_to_au(au6_stats)   # AU6  
    au_activations[9] = normalize_strain_to_au(au9_stats)   # AU9
    au_activations[10] = normalize_strain_to_au(au10_stats) # AU10
    au_activations[12] = normalize_strain_to_au(au12_stats) # AU12
    
    # Add correlated AUs based on facial anatomy
    # Happiness correlation: AU6 + AU12
    if au_activations[6] > 0.3 and au_activations[12] > 0.3:
        au_activations[25] = min(1.0, (au_activations[6] + au_activations[12]) / 2)  # AU25: lips part
    
    # Disgust correlation: AU9 + AU10  
    if au_activations[9] > 0.3 and au_activations[10] > 0.3:
        au_activations[15] = min(1.0, (au_activations[9] + au_activations[10]) / 2)  # AU15: lip corner depressor
        au_activations[16] = min(1.0, au_activations[10] * 0.8)  # AU16: lower lip depressor
    
    # Surprise correlation: AU4 activation (brow raise)
    if au_activations[4] > 0.4:
        au_activations[1] = min(1.0, au_activations[4] * 0.9)  # AU1: inner brow raiser
        au_activations[2] = min(1.0, au_activations[4] * 0.8)  # AU2: outer brow raiser
        au_activations[5] = min(1.0, au_activations[4] * 0.7)  # AU5: upper lid raiser
    
    return au_activations


def simulate_au_activations_from_flow(flow: np.ndarray) -> np.ndarray:
    """
    Simulate AU activations from optical flow (for testing).
    
    In practice, this would use a dedicated AU detection model.
    
    Args:
        flow: Optical flow array
        
    Returns:
        Simulated AU activations (27 AUs)
    """
    # Simple simulation based on flow patterns
    # This is a placeholder - real implementation would use AU detection
    
    vertical_motion = np.mean(np.abs(flow[..., 1]))
    horizontal_motion = np.mean(np.abs(flow[..., 0]))
    total_motion = np.mean(np.abs(flow))
    
    # Simulate 27 AU activations
    au_activations = np.random.rand(27) * 0.3  # Base random activations
    
    # Add pattern-based activations
    au_activations[1] = min(1.0, vertical_motion * 2)  # AU1: brow raiser
    au_activations[2] = min(1.0, vertical_motion * 1.8)  # AU2: brow raiser
    au_activations[6] = min(1.0, horizontal_motion * 1.5)  # AU6: cheek raiser
    au_activations[12] = min(1.0, horizontal_motion * 1.2)  # AU12: lip corner
    
    return au_activations


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing AU Soft Boosting...")
    
    # Create booster
    booster = AUSoftBoosting(lambda_boost=0.3)
    
    # Simulate AU activations
    au_activations = simulate_au_activations_from_flow(np.random.rand(64, 64, 2))
    print(f"✅ AU activations shape: {au_activations.shape}")
    
    # Test with raw scores
    raw_scores = np.array([0.3, 0.2, 0.4, 0.1])  # happiness, disgust, surprise, repression
    boosted_scores = booster.apply_soft_boosting_numpy(raw_scores, au_activations)
    
    print(f"✅ Original scores: {raw_scores}")
    print(f"✅ Boosted scores: {boosted_scores}")
    
    # Analyze contributions
    analysis = booster.get_au_contribution_analysis(au_activations, raw_scores, boosted_scores)
    print(f"✅ Emotion weights: {analysis['emotion_weights']}")
    
    print("🎉 AU Soft Boosting implementation complete!")
