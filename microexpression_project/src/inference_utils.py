"""
Shared helpers for hybrid CNN–SVM inference (web + offline scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np

from config import LABEL_TO_EMOTION, NUM_EMOTIONS


def try_load_enhanced_hybrid_raw(raw: Any) -> Tuple[Any, Optional[str]]:
    """
    Deserialize a joblib checkpoint into a fitted hybrid model (or pass through object
    that already exposes ``extract_all_features`` and ``pipeline``).

    Returns ``(model, None)`` on success, or ``(None, error_message)``.
    """
    from micro_expression_model import EnhancedHybridModel

    if hasattr(raw, "extract_all_features") and hasattr(raw, "pipeline"):
        return raw, None
    if (
        isinstance(raw, dict)
        and "feature_extractor_state" in raw
        and ("pipeline" in raw or "default_pipeline" in raw)
    ):
        em = EnhancedHybridModel(
            cnn_model=raw.get("cnn_model", "hybrid"),
            classifier_type=raw.get("classifier_type", "svm"),
            use_facesleuth=True,
            vertical_alpha=1.5,
            enable_boosting_logging=False,
        )
        em.feature_extractor.load_state_dict(raw["feature_extractor_state"])
        em.pipeline = raw.get("pipeline") or raw.get("default_pipeline")
        em.is_fitted = raw.get("is_fitted", True)
        # Optional: per-subject LOSO pipelines (for predict-folder evaluation parity)
        if "pipelines_by_subject" in raw and isinstance(raw["pipelines_by_subject"], dict):
            em.pipelines_by_subject = raw["pipelines_by_subject"]
            em.loso_checkpoint = True
        return em, None
    return None, f"unexpected checkpoint type {type(raw).__name__}"


def load_enhanced_hybrid_from_path(path: str | Path) -> Any:
    """Load hybrid checkpoint from disk; raises ``ValueError`` if the format is unknown."""
    path = Path(path)
    raw = joblib.load(path)
    model, err = try_load_enhanced_hybrid_raw(raw)
    if err:
        raise ValueError(f"{path}: {err}")
    return model


def proba_by_emotion_name(pipeline, proba_1d) -> dict:
    """Map sklearn predict_proba columns (pipeline.classes_ order) to emotion strings."""
    out = {}
    for cls, p in zip(pipeline.classes_, np.ravel(proba_1d)):
        ic = int(np.asarray(cls).item())
        name = LABEL_TO_EMOTION.get(ic, str(cls))
        out[name] = float(p)
    for i in range(NUM_EMOTIONS):
        out.setdefault(LABEL_TO_EMOTION[i], 0.0)
    return out


def hybrid_predict_from_features(model, features_2d: np.ndarray) -> dict:
    """
    Headline emotion = argmax(predict_proba) so it matches probability bars.
    SVC(probability=True) can disagree with predict(); we prefer proba for UX consistency.
    """
    features_2d = np.asarray(features_2d, dtype=np.float64)
    if features_2d.ndim == 1:
        features_2d = features_2d.reshape(1, -1)
    elif features_2d.ndim != 2:
        raise ValueError(f"features must be 1D or 2D array, got shape {features_2d.shape}")
    pl = model.pipeline
    pred_scalar = pl.predict(features_2d)[0]
    proba = np.ravel(pl.predict_proba(features_2d)[0])
    by_emotion = proba_by_emotion_name(pl, proba)
    best_emotion = max(by_emotion, key=by_emotion.get)
    pred_int = int(np.asarray(pred_scalar).item())
    pipeline_emotion = LABEL_TO_EMOTION.get(pred_int, str(pred_scalar))
    return {
        "by_emotion": by_emotion,
        "prediction_emotion": best_emotion,
        "confidence": float(by_emotion[best_emotion]),
        "pipeline_prediction": pipeline_emotion,
        "pipeline_agrees": pipeline_emotion == best_emotion,
    }


def hybrid_predict_from_features_with_pipeline(pipeline, features_2d: np.ndarray) -> dict:
    """
    Same output as hybrid_predict_from_features(), but using an explicit sklearn pipeline
    (useful for LOSO per-subject pipelines in deployment/evaluation parity).
    """
    features_2d = np.asarray(features_2d, dtype=np.float64)
    if features_2d.ndim == 1:
        features_2d = features_2d.reshape(1, -1)
    elif features_2d.ndim != 2:
        raise ValueError(f"features must be 1D or 2D array, got shape {features_2d.shape}")
    pred_scalar = pipeline.predict(features_2d)[0]
    proba = np.ravel(pipeline.predict_proba(features_2d)[0])
    by_emotion = proba_by_emotion_name(pipeline, proba)
    best_emotion = max(by_emotion, key=by_emotion.get)
    pred_int = int(np.asarray(pred_scalar).item())
    pipeline_emotion = LABEL_TO_EMOTION.get(pred_int, str(pred_scalar))
    return {
        "by_emotion": by_emotion,
        "prediction_emotion": best_emotion,
        "confidence": float(by_emotion[best_emotion]),
        "pipeline_prediction": pipeline_emotion,
        "pipeline_agrees": pipeline_emotion == best_emotion,
    }
