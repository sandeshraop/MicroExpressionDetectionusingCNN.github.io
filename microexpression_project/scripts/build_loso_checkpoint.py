#!/usr/bin/env python3
"""
Build a LOSO checkpoint bundle for web inference parity on `data/predict/`.

Why this exists:
- The deployable "real_data_model_*.pkl" is trained on all samples (default behavior).
- For reviewer-safe evaluation and predict-folder demos, you may want LOSO behavior:
  when predicting subject subXX, use a pipeline fitted on all other subjects.

This script:
- Loads an existing trained checkpoint (feature extractor + default pipeline).
- Re-extracts hybrid features for all CASME-II samples (CSV onset/apex/offset + reg_img).
- Fits one sklearn pipeline per held-out subject.
- Saves a new joblib checkpoint with:
  - feature_extractor_state (shared)
  - default_pipeline (fit on all data)
  - pipelines_by_subject (LOSO)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score

root = Path(__file__).resolve().parent.parent
for p in (root, root / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from config import EMOTION_LABELS, LABEL_TO_EMOTION, NUM_EMOTIONS
from preprocessing_pipeline import OnsetApexOffsetSelector
from optical_flow_utils import triplet_to_six_channel_flow
from inference_utils import load_enhanced_hybrid_from_path


def _extract_all_features_for_labels(model, labels_df: pd.DataFrame, data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selector = OnsetApexOffsetSelector(None)
    feats: list[np.ndarray] = []
    y: list[int] = []
    subjects: list[str] = []

    model.feature_extractor.eval()
    model.feature_extractor.to("cpu")

    with torch.no_grad():
        for _, row in labels_df.iterrows():
            emotion = str(row["emotion_label"]).strip()
            if emotion not in EMOTION_LABELS:
                continue

            subject = str(row["subject_id"]).strip().lower()
            episode = str(row["episode_id"]).strip()
            ep_dir = data_root / subject / episode
            if not ep_dir.is_dir():
                continue

            sample = {
                "video_path": str(ep_dir),
                "onset_frame": int(row["onset_frame"]),
                "apex_frame": int(row["apex_frame"]),
                "offset_frame": int(row["offset_frame"]),
            }
            triplet = selector.load_onset_apex_offset_rgb(sample)
            if triplet is None:
                continue
            o, a, off = triplet

            frames_tensor = torch.stack(
                [
                    torch.from_numpy(o).permute(2, 0, 1),
                    torch.from_numpy(a).permute(2, 0, 1),
                    torch.from_numpy(off).permute(2, 0, 1),
                ],
                dim=0,
            ).float()
            flow_np = triplet_to_six_channel_flow(o, a, off)
            flows_tensor = torch.from_numpy(flow_np).float()

            frames_b = frames_tensor.unsqueeze(0)
            flows_b = flows_tensor.unsqueeze(0)

            f = model.extract_all_features(frames_b, flows_b)
            f = np.asarray(f, dtype=np.float64)
            if f.ndim == 1:
                f = f.reshape(1, -1)

            feats.append(f[0])
            y.append(int(EMOTION_LABELS[emotion]))
            subjects.append(subject)

    if not feats:
        raise RuntimeError("No features extracted (check your data_root and labels CSV).")

    X = np.asarray(feats, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    g_arr = np.asarray(subjects, dtype=str)
    return X, y_arr, g_arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_checkpoint",
        type=str,
        default=str(root / "models" / "real_data_model_20260414_085233.pkl"),
        help="Path to existing trained checkpoint (real_data_model_*.pkl).",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=str(root / "data" / "predict"),
        help="CASME reg_img root containing subXX/EP... folders (default: data/predict).",
    )
    ap.add_argument(
        "--labels_csv",
        type=str,
        default=str(root / "data" / "labels" / "casme2_labels.csv"),
        help="Labels CSV with onset/apex/offset.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(root / "models"),
        help="Output directory for LOSO bundle.",
    )
    args = ap.parse_args()

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.is_file():
        raise FileNotFoundError(base_ckpt)

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    labels_csv = Path(args.labels_csv)
    if not labels_csv.is_file():
        raise FileNotFoundError(labels_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOSO] Loading base checkpoint: {base_ckpt}")
    model = load_enhanced_hybrid_from_path(base_ckpt)
    if not hasattr(model, "extract_all_features") or not hasattr(model, "pipeline"):
        raise RuntimeError("Base checkpoint did not load as EnhancedHybridModel-like object.")

    print(f"[LOSO] Loading labels: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)

    print(f"[LOSO] Extracting features from data_root={data_root}")
    X, y, groups = _extract_all_features_for_labels(model, labels_df, data_root)
    print(f"[LOSO] Features: X={X.shape}, y={y.shape}, subjects={len(np.unique(groups))}")

    # Fit default pipeline on all data (deployment baseline)
    default_pl = clone(model.pipeline)
    default_pl.fit(X, y)

    # Fit per-subject LOSO pipelines
    pipelines_by_subject: dict[str, object] = {}
    uniq = sorted(set(groups.tolist()))
    for subj in uniq:
        test_mask = groups == subj
        train_mask = ~test_mask
        if int(np.sum(train_mask)) < 5 or int(np.sum(test_mask)) < 1:
            continue
        pl = clone(model.pipeline)
        pl.fit(X[train_mask], y[train_mask])
        pipelines_by_subject[subj] = pl

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"loso_bundle_model_{ts}.pkl"
    payload = {
        "cnn_model": getattr(model, "cnn_model", "hybrid"),
        "classifier_type": getattr(model, "classifier_type", "svm"),
        "feature_extractor_state": model.feature_extractor.state_dict(),
        "default_pipeline": default_pl,
        "pipelines_by_subject": pipelines_by_subject,
        "is_fitted": True,
        "loso_checkpoint": True,
        "built_from": str(base_ckpt),
        "data_root": str(data_root),
        "labels_csv": str(labels_csv),
        "n_samples": int(X.shape[0]),
        "n_subjects": int(len(uniq)),
        "n_subject_pipelines": int(len(pipelines_by_subject)),
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(payload, out_path)
    print(f"[LOSO] Saved LOSO bundle: {out_path}")

    # Pooled LOSO predictions (each sample predicted by pipeline trained without its subject)
    y_pred = np.zeros(len(y), dtype=np.int64)
    for subj in uniq:
        if subj not in pipelines_by_subject:
            continue
        mask = groups == subj
        y_pred[mask] = pipelines_by_subject[subj].predict(X[mask])

    pooled_acc = float(accuracy_score(y, y_pred))
    labels_order = list(range(NUM_EMOTIONS))
    recalls = recall_score(y, y_pred, labels=labels_order, average=None, zero_division=0)
    uar = float(np.mean(recalls))
    per_class_recall = {LABEL_TO_EMOTION[i]: float(recalls[i]) for i in labels_order}

    meta_path = out_dir / f"loso_bundle_model_{ts}.json"
    meta = {
        "checkpoint_kind": "loso_bundle",
        "timestamp": ts,
        "model_type": "FaceSleuthEnhancedHybridModel",
        "training_data": "Real CASME-II cropped images",
        "evaluation_method": "LOSO SVM on hybrid features; bundle stores per-subject held-out pipelines for inference parity",
        "training_accuracy": pooled_acc,
        "loso_pooled_accuracy": pooled_acc,
        "uar": uar,
        "happiness_recall": float(per_class_recall.get("happiness", 0.0)),
        "per_class_recall": per_class_recall,
        "feature_dim": int(X.shape[1]),
        "training_samples": int(len(y)),
        "n_subjects": int(len(uniq)),
        "built_from": str(base_ckpt),
        "data_root": str(data_root),
        "labels_csv": str(labels_csv),
        "metadata_note": (
            "training_accuracy is pooled LOSO (concatenated held-out predictions). "
            "UAR is macro-averaged recall over classes on those predictions."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[LOSO] Saved bundle metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

