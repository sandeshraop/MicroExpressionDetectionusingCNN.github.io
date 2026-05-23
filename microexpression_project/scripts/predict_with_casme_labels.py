#!/usr/bin/env python3
"""
Run the trained hybrid model on CASME-II-style episode folders using the same
onset / apex / offset protocol as training (matches CASME-II metadata in the labels CSV).

Use this when evaluating under `data/casme2` or a copy such as `data/predict`:
random frame sampling or generic video uploads are not comparable to training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC))

from config import LABEL_TO_EMOTION  # noqa: E402
from inference_utils import proba_by_emotion_name  # noqa: E402
from micro_expression_model import EnhancedHybridModel  # noqa: E402
from optical_flow_utils import triplet_to_six_channel_flow  # noqa: E402
from preprocessing_pipeline import OnsetApexOffsetSelector  # noqa: E402


def _load_model(model_path: Path):
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
    raise TypeError(f"Unsupported checkpoint type: {type(raw)}")


def main():
    ap = argparse.ArgumentParser(description="Predict on CASME-II folders using labels CSV O/A/O.")
    ap.add_argument(
        "--data_root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "casme2",
        help="Root with subXX/EP_.../reg_img*.jpg (or use data/predict with same layout).",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=_PROJECT_ROOT / "data" / "labels" / "casme2_labels.csv",
        help="CSV with subject_id, episode_id, emotion_label, onset_frame, apex_frame, offset_frame.",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to real_data_model_*.pkl (default: newest under models/).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max episodes (0 = all).")
    ap.add_argument("--subject", default=None, help="Run only this subject_id (e.g. sub01).")
    ap.add_argument("--episode", default=None, help="Run only this episode_id (e.g. EP02_01f); EP02_01 matches EP02_01f.")
    args = ap.parse_args()

    if not args.labels.is_file():
        print(f"Labels file not found: {args.labels}")
        sys.exit(1)
    if not args.data_root.is_dir():
        print(f"Data root not found: {args.data_root}")
        sys.exit(1)

    model_path = args.model
    if model_path is None:
        candidates = sorted(
            (_PROJECT_ROOT / "models").glob("real_data_model_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No real_data_model_*.pkl under models/. Train first or pass --model.")
            sys.exit(1)
        model_path = candidates[0]
    print(f"Checkpoint: {model_path}")

    model = _load_model(model_path)
    selector = OnsetApexOffsetSelector(str(args.labels))
    samples = selector.get_all_samples(str(args.data_root))
    if args.subject:
        sid = args.subject.strip().lower()
        samples = [s for s in samples if s["subject"].lower() == sid]
    if args.episode:
        w = args.episode.strip().lower()
        cand = [s for s in samples if s["episode"].lower() == w]
        if not cand and not w.endswith("f"):
            cand = [s for s in samples if s["episode"].lower() == w + "f"]
        if not cand:
            cand = [s for s in samples if s["episode"].lower().startswith(w)]
        samples = cand
    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        print("No labeled episodes found under data_root (check folder names match CSV).")
        sys.exit(1)

    correct = 0
    rows = []
    for sample in samples:
        loaded = selector.load_onset_apex_offset_rgb(sample)
        if loaded is None:
            print(f"SKIP (no images): {sample['subject']}/{sample['episode']}")
            continue
        onset, apex, offset = loaded
        flow_np = triplet_to_six_channel_flow(onset, apex, offset)
        frames_t = torch.stack(
            [
                torch.from_numpy(onset).permute(2, 0, 1),
                torch.from_numpy(apex).permute(2, 0, 1),
                torch.from_numpy(offset).permute(2, 0, 1),
            ],
            dim=0,
        ).float()
        flows_t = torch.from_numpy(flow_np).float()
        frames_b = frames_t.unsqueeze(0)
        flows_b = flows_t.unsqueeze(0)

        feats = model.extract_all_features(frames_b, flows_b)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        pred_raw = model.pipeline.predict(feats)[0]
        pred_int = int(np.asarray(pred_raw).item())
        pred_emotion = LABEL_TO_EMOTION.get(pred_int, str(pred_raw))
        true_emotion = sample["emotion"]
        ok = pred_emotion == true_emotion
        correct += int(ok)
        proba = model.pipeline.predict_proba(feats)[0]
        rows.append(
            {
                "subject": sample["subject"],
                "episode": sample["episode"],
                "true": true_emotion,
                "pred": pred_emotion,
                "match": ok,
                "proba": proba_by_emotion_name(model.pipeline, proba),
            }
        )
        mark = "OK" if ok else "XX"
        print(f"[{mark}] {sample['subject']}/{sample['episode']}: true={true_emotion} pred={pred_emotion}")

    acc = correct / max(len(rows), 1)
    print(f"\nAccuracy (protocol = CSV O/A/O + reg images): {acc:.3f} ({correct}/{len(rows)})")
    out_json = _PROJECT_ROOT / "data" / "predict" / "last_batch_predictions.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "n": len(rows), "rows": rows}, f, indent=2)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
