#!/usr/bin/env python3
"""
Evaluate every video under data/predict/subXX/*.avi.

For each file we always emit a prediction. Tensor source (best → fallback):
  1) data/casme2/<sub>/<ep>/reg_img + casme2_labels.csv onset/apex/offset (same as training)
  2) reg_img folder only → first / mid / last image triplet
  3) Decode the .avi with VideoPreprocessor

Ground truth for accuracy: rows where emotion_label is in config.EMOTION_LABELS (5 classes, includes others).

Writes data/predict/video_eval_results.json and data/predict/last_batch_predictions.json
(rows always include `pred`; `true` / `match` may be null when CSV emotion is missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC))

from casme_predict_bridge import (  # noqa: E402
    default_regimg_search_roots,
    episode_id_candidates,
    find_csv_row,
    get_clip_tensors,
)
from config import EMOTION_LABELS  # noqa: E402
from inference_utils import hybrid_predict_from_features  # noqa: E402
from micro_expression_model import EnhancedHybridModel  # noqa: E402
from preprocessing_pipeline import OnsetApexOffsetSelector, VideoPreprocessor  # noqa: E402


def _load_model(model_path: Path):
    import joblib

    raw = joblib.load(model_path)
    if hasattr(raw, "extract_all_features") and hasattr(raw, "pipeline"):
        return raw
    if isinstance(raw, dict) and "feature_extractor_state" in raw and "pipeline" in raw:
        em = EnhancedHybridModel(
            cnn_model=raw.get("cnn_model", "hybrid"),
            classifier_type=raw.get("classifier_type", "svm"),
            use_facesleuth=True,
            vertical_alpha=1.5,
            enable_boosting_logging=False,
        )
        em.feature_extractor.load_state_dict(raw["feature_extractor_state"])
        em.pipeline = raw["pipeline"]
        em.is_fitted = raw.get("is_fitted", True)
        return em
    raise TypeError(f"Unsupported checkpoint: {type(raw)}")


def _find_any_csv_row(df: pd.DataFrame, subject: str, stem: str):
    for eid in episode_id_candidates(stem):
        r = find_csv_row(df, subject, eid)
        if r is not None:
            return r, eid
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_dir", type=Path, default=_PROJECT_ROOT / "data" / "predict")
    ap.add_argument("--labels", type=Path, default=_PROJECT_ROOT / "data" / "labels" / "casme2_labels.csv")
    ap.add_argument("--casme2", type=Path, default=_PROJECT_ROOT / "data" / "casme2")
    ap.add_argument("--model", type=Path, default=None)
    args = ap.parse_args()

    predict_dir = args.predict_dir.expanduser().resolve()
    vids = sorted(predict_dir.glob("sub*/**/*.[aA][vV][iI]")) + sorted(
        predict_dir.glob("sub*/**/*.[mM][pP]4")
    )
    vids = sorted({p.resolve() for p in vids if p.is_file()})
    if not vids:
        print(f"No videos under {predict_dir}")
        sys.exit(1)

    model_path = args.model
    if model_path is None:
        cands = sorted(
            (_PROJECT_ROOT / "models").glob("real_data_model_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            print("No real_data_model_*.pkl found.")
            sys.exit(1)
        model_path = cands[0]

    labels_df = pd.read_csv(args.labels)
    model = _load_model(model_path)
    selector = OnsetApexOffsetSelector(str(args.labels))
    video_pre = VideoPreprocessor()

    rows = []
    correct = 0
    evaluated = 0
    errors = 0

    for vp in vids:
        rel = vp.relative_to(predict_dir).as_posix()
        subject = vp.parent.name.lower()
        stem = vp.stem

        row_any, ep_hit = _find_any_csv_row(labels_df, subject, stem)
        raw_emo = str(row_any["emotion_label"]).strip().lower() if row_any is not None else None
        true_lbl = raw_emo if raw_emo in EMOTION_LABELS else None

        try:
            roots = default_regimg_search_roots(_PROJECT_ROOT, subject)
            primary = roots[0] if roots else args.casme2
            extras = roots[1:] if len(roots) > 1 else []
            ft, fl, source = get_clip_tensors(
                subject=subject,
                filename_stem=stem,
                video_path=vp,
                casme2_root=primary,
                labels_df=labels_df,
                selector=selector,
                video_pre=video_pre,
                max_video_frames=64,
                extra_regimg_roots=extras,
            )
            frames_b = ft.unsqueeze(0)
            flows_b = fl.unsqueeze(0)
            with torch.no_grad():
                feats = model.extract_all_features(frames_b, flows_b)
            arr = feats.detach().cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            hp = hybrid_predict_from_features(model, arr.astype(np.float64, copy=False))
            pred = hp["prediction_emotion"]
            ok = None
            if true_lbl is not None:
                ok = pred == true_lbl
                if ok:
                    correct += 1
                evaluated += 1

            rows.append(
                {
                    "file": rel,
                    "subject": subject,
                    "episode_stem": stem,
                    "episode_csv": ep_hit,
                    "tensor_source": source,
                    "csv_emotion": raw_emo,
                    "ground_truth": true_lbl,
                    "true_4class": true_lbl,
                    "pred": pred,
                    "match": ok,
                    "confidence": hp["confidence"],
                    "status": "ok",
                }
            )
        except Exception as e:
            errors += 1
            rows.append(
                {
                    "file": rel,
                    "subject": subject,
                    "episode_stem": stem,
                    "tensor_source": None,
                    "csv_emotion": raw_emo,
                    "ground_truth": true_lbl,
                    "true_4class": true_lbl,
                    "pred": None,
                    "match": None,
                    "status": "error",
                    "error": str(e),
                }
            )

    acc = correct / evaluated if evaluated else 0.0
    out = {
        "model": str(model_path),
        "predict_dir": str(predict_dir),
        "labels_csv": str(args.labels),
        "casme2_root": str(args.casme2),
        "pipeline": "casme_predict_bridge: CSV+regimg → naive regimg → VideoPreprocessor; hybrid_predict_from_features",
        "videos_total": len(vids),
        "predictions_emitted": len(vids) - errors,
        "errors": errors,
        "with_labeled_ground_truth": evaluated,
        "correct_labeled": correct,
        "accuracy_on_labeled_subset": acc,
        "with_4class_ground_truth": evaluated,
        "correct_on_4class": correct,
        "accuracy_on_4class_subset": acc,
        "rows": rows,
    }

    out_path = predict_dir / "video_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    legacy_rows = []
    for r in rows:
        legacy_rows.append(
            {
                "file": r["file"],
                "true": r.get("ground_truth", r.get("true_4class")),
                "pred": r.get("pred"),
                "match": r.get("match"),
                "confidence": r.get("confidence"),
                "tensor_source": r.get("tensor_source"),
                "status": r.get("status"),
                "error": r.get("error"),
            }
        )
    last_path = predict_dir / "last_batch_predictions.json"
    with open(last_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "n": evaluated,
                "correct": correct,
                "errors": errors,
                "predictions_for_all_files": len(vids) - errors,
                "source": "evaluate_predict_videos.py (unified CASME regimg + video fallback)",
                "rows": legacy_rows,
            },
            f,
            indent=2,
        )

    print(f"Videos scanned: {len(vids)}")
    print(f"Predictions emitted: {len(vids) - errors} (errors: {errors})")
    print(f"With labeled ground truth: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Accuracy (labeled subset): {acc:.3f}")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {last_path}")


if __name__ == "__main__":
    main()
