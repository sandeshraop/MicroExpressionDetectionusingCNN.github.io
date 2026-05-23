#!/usr/bin/env python3
"""
Generate true LOSO (leave-one-subject-out) predictions and save to JSON.

Why: Confusion matrices built from in-sample predictions (model fit on all data)
can look "too perfect". This script produces *held-out* predictions only.

Output format matches data/predict/last_batch_predictions.json so it can be fed into:
  scripts/generate_confusion_matrix.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from config import LABEL_TO_EMOTION, NUM_EMOTIONS  # noqa: E402
from dataset_loader import CNNCASMEIIDataset  # noqa: E402
from inference_utils import load_enhanced_hybrid_from_path, proba_by_emotion_name  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "real_data_model_20260414_085233.pkl",
        help="Saved hybrid checkpoint (.pkl)",
    )
    ap.add_argument(
        "--data_root",
        type=Path,
        default=root / "data" / "casme2",
        help="CASME-II reg_img root (subXX/EP.../reg_img*.jpg)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=root / "data" / "labels" / "casme2_labels.csv",
        help="Labels CSV (subject_id, episode_id, emotion_label, onset/apex/offset frames)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "data" / "predict" / "loso_predictions.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    if not args.model.is_file():
        print("ERROR: model not found:", args.model)
        return 1
    if not args.data_root.is_dir():
        print("ERROR: data_root not found:", args.data_root)
        return 1
    if not args.labels.is_file():
        print("ERROR: labels CSV not found:", args.labels)
        return 1

    print("Loading checkpoint:", args.model)
    model = load_enhanced_hybrid_from_path(args.model)
    model.feature_extractor.eval()

    print("Building tensors from CNNCASMEIIDataset...")
    ds = CNNCASMEIIDataset(str(args.data_root), str(args.labels))

    frames_list, flows_list, labels_list, groups, metas = [], [], [], [], []
    for i in range(len(ds)):
        frames, flows, label, metadata = ds[i]
        frames_list.append(frames)
        flows_list.append(flows)
        labels_list.append(int(label.item()))
        g = metadata.get("subject") or metadata.get("subject_id") or f"sub_{i}"
        groups.append(str(g))
        metas.append(metadata)

    frames_tensor = torch.stack(frames_list)
    flows_tensor = torch.stack(flows_list)
    y = np.asarray(labels_list, dtype=np.int64)
    groups_np = np.asarray(groups)

    print("Extracting hybrid features (frozen CNN)...")
    feats = []
    with torch.no_grad():
        for i in range(len(frames_tensor)):
            f = model.extract_all_features(frames_tensor[i : i + 1], flows_tensor[i : i + 1])
            arr = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
            feats.append(arr.reshape(1, -1))
    X = np.vstack(feats)
    print("  X shape:", X.shape, " y shape:", y.shape)

    logo = LeaveOneGroupOut()
    rows = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups_np)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Fit scaler+classifier on training fold only (no leakage).
        model.pipeline.fit(X_tr, y_tr)

        pred = model.pipeline.predict(X_te)
        proba = model.pipeline.predict_proba(X_te)

        for j, idx in enumerate(test_idx):
            meta = metas[int(idx)]
            true_i = int(y_te[j])
            pred_i = int(np.asarray(pred[j]).item())
            rows.append(
                {
                    "subject": meta.get("subject") or meta.get("subject_id"),
                    "episode": meta.get("episode") or meta.get("episode_id"),
                    "true": LABEL_TO_EMOTION.get(true_i, str(true_i)),
                    "pred": LABEL_TO_EMOTION.get(pred_i, str(pred_i)),
                    "match": bool(pred_i == true_i),
                    "proba": proba_by_emotion_name(model.pipeline, proba[j]),
                    "fold": int(fold),
                }
            )

        y_true_all.append(y_te.astype(np.int64).ravel())
        y_pred_all.append(np.asarray(pred, dtype=np.int64).ravel())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    acc = float(np.mean(y_true_all == y_pred_all))

    out = {
        "protocol": "LOSO (held-out subjects only); fold pipeline fit on train split each fold",
        "model": str(args.model),
        "data_root": str(args.data_root),
        "labels_csv": str(args.labels),
        "n": int(len(rows)),
        "accuracy": acc,
        "rows": rows,
        "labels_order": [LABEL_TO_EMOTION[i] for i in range(NUM_EMOTIONS)],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {args.out}")
    print(f"Accuracy (LOSO pooled): {acc:.6f} ({int(np.sum(y_true_all == y_pred_all))}/{len(y_true_all)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

