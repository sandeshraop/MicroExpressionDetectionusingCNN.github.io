#!/usr/bin/env python3
"""
Recompute LOSO accuracy / UAR / per-class recall from a saved real_data_model pickle
and optional JSON metadata, to confirm reported numbers match sklearn definitions.

Run from repo: python scripts/verify_model_metrics.py [--model path] [--meta path]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from config import LABEL_TO_EMOTION, NUM_EMOTIONS
from dataset_loader import CNNCASMEIIDataset
from inference_utils import load_enhanced_hybrid_from_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "real_data_model_20260414_085233.pkl",
        help="Saved hybrid checkpoint (.pkl)",
    )
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional metadata JSON to compare (default: sibling real_data_model_metadata_*.json)",
    )
    args = ap.parse_args()

    model_path = args.model
    if not model_path.is_file():
        print("ERROR: model not found:", model_path)
        return 1

    meta_path = args.meta
    if meta_path is None:
        stem = model_path.stem.replace("real_data_model_", "", 1)
        cand = model_path.parent / f"real_data_model_metadata_{stem}.json"
        meta_path = cand if cand.is_file() else None

    meta = {}
    if meta_path and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    labels_csv = root / "data" / "labels" / "casme2_labels.csv"
    data_root = root / "data" / "casme2"
    if not labels_csv.is_file() or not data_root.is_dir():
        print("ERROR: need data/labels/casme2_labels.csv and data/casme2")
        return 1

    print("Loading checkpoint:", model_path.name)
    model = load_enhanced_hybrid_from_path(model_path)
    model.feature_extractor.eval()

    print("Building tensors from CNNCASMEIIDataset...")
    ds = CNNCASMEIIDataset(str(data_root), str(labels_csv))
    frames_list, flows_list, labels_list, groups = [], [], [], []
    for i in range(len(ds)):
        frames, flows, label, metadata = ds[i]
        frames_list.append(frames)
        flows_list.append(flows)
        labels_list.append(int(label.item()))
        groups.append(metadata.get("subject") or metadata.get("subject_id") or f"sub_{i}")

    frames_tensor = torch.stack(frames_list)
    flows_tensor = torch.stack(flows_list)
    labels_np = np.asarray(labels_list, dtype=np.int64)
    groups_np = np.asarray(groups)

    print("Extracting hybrid features (frozen CNN)...")
    all_features = []
    with torch.no_grad():
        for i in range(len(frames_tensor)):
            f = model.extract_all_features(frames_tensor[i : i + 1], flows_tensor[i : i + 1])
            arr = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
            all_features.append(arr.reshape(1, -1))
    X = np.vstack(all_features)
    print("  X shape:", X.shape, " y shape:", labels_np.shape)

    logo = LeaveOneGroupOut()
    fold_accs: list[float] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, labels_np, groups=groups_np)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = labels_np[train_idx], labels_np[test_idx]
        try:
            model.pipeline.fit(X_tr, y_tr)
            pred = model.pipeline.predict(X_te)
            fold_accs.append(float(np.mean(pred == y_te)))
            y_true_parts.append(y_te.astype(np.int64).ravel())
            y_pred_parts.append(np.asarray(pred, dtype=np.int64).ravel())
        except Exception as e:
            print(f"  Fold {fold + 1} FAILED:", e)
            fold_accs.append(float("nan"))

    mean_fold = float(np.nanmean(fold_accs))
    std_fold = float(np.nanstd(fold_accs))
    y_true_all = np.concatenate(y_true_parts)
    y_pred_all = np.concatenate(y_pred_parts)
    pooled_acc = float(accuracy_score(y_true_all, y_pred_all))

    labels_order = list(range(NUM_EMOTIONS))
    recalls = recall_score(
        y_true_all, y_pred_all, labels=labels_order, average=None, zero_division=0
    )
    uar = float(np.mean(recalls))
    per_class = {LABEL_TO_EMOTION[i]: float(recalls[i]) for i in labels_order}

    print()
    print("=== Recomputed (from checkpoint + current CASME-II data) ===")
    print(f"  Mean LOSO fold accuracy: {mean_fold:.6f}  (std {std_fold:.6f})")
    print(f"  Pooled LOSO accuracy:    {pooled_acc:.6f}  (single accuracy on all held-out preds)")
    print(f"  UAR (macro recall):       {uar:.6f}")
    print(f"  Per-class recall:         {per_class}")

    if meta:
        print()
        print("=== Metadata file ===", meta_path)
        ta = meta.get("training_accuracy")
        ts = meta.get("loso_accuracy_std")
        m_uar = meta.get("uar")
        print(f"  training_accuracy (stored): {ta}")
        print(f"  loso_accuracy_std (stored): {ts}")
        print(f"  uar (stored):               {m_uar}")
        print(f"  per_class_recall (stored):  {meta.get('per_class_recall')}")

        tol = 1e-5
        ok_mean = ta is not None and abs(float(ta) - mean_fold) < tol
        ok_std = ts is not None and abs(float(ts) - std_fold) < tol
        ok_uar = m_uar is not None and abs(float(m_uar) - uar) < tol
        print()
        if ok_mean and ok_std and ok_uar:
            print("OK: mean LOSO accuracy, std, and UAR match metadata within tolerance.")
        else:
            print("MISMATCH detail:")
            if not ok_mean:
                print(f"  mean fold acc: meta={ta} vs recomputed={mean_fold} diff={float(ta) - mean_fold}")
            if not ok_std:
                print(f"  std:           meta={ts} vs recomputed={std_fold}")
            if not ok_uar:
                print(f"  uar:           meta={m_uar} vs recomputed={uar}")

        # Note naming
        print()
        print(
            "NOTE: JSON key 'training_accuracy' holds mean LOSO fold accuracy (not CNN epoch acc)."
        )
        fd = meta.get("feature_dim")
        br = meta.get("feature_breakdown", {}).get("total_corrected")
        if fd and br and int(fd) != int(br):
            print(
                f"NOTE: metadata feature_dim={fd} vs feature_breakdown.total_corrected={br} (doc inconsistency; actual X columns = {X.shape[1]})."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
