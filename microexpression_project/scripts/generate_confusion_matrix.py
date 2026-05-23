#!/usr/bin/env python3
"""
Generate a confusion matrix from saved prediction JSON.

Supported inputs:
- data/predict/last_batch_predictions.json (from scripts/predict_with_casme_labels.py)
- data/predict/video_eval_results.json (from scripts/evaluate_predict_videos.py)

Outputs:
- <out_dir>/confusion_matrix.csv
- <out_dir>/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import EMOTION_DISPLAY_NAMES, EMOTION_LABELS  # noqa: E402


def _coerce_emotion_name(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s == "nan" or s == "none":
        return None
    return s


def _iter_rows(payload: dict) -> list[dict]:
    # last_batch_predictions.json: {accuracy, n, rows:[{true,pred,...}]}
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    # video_eval_results.json: { ... rows:[{ground_truth,pred,...}] }
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    return []


def _extract_true_pred(rows: list[dict]) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for r in rows:
        t = _coerce_emotion_name(
            r.get("true")
            or r.get("ground_truth")
            or r.get("true_4class")
            or r.get("label")
            or r.get("emotion")
        )
        p = _coerce_emotion_name(r.get("pred") or r.get("prediction"))
        if t is None or p is None:
            continue
        if t not in EMOTION_LABELS:
            continue
        if p not in EMOTION_LABELS:
            continue
        y_true.append(t)
        y_pred.append(p)
    return y_true, y_pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions_json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "predict" / "last_batch_predictions.json",
        help="Path to prediction JSON produced by project scripts.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_PROJECT_ROOT / "results" / "confusion_matrix",
        help="Output directory for CSV/PNG.",
    )
    ap.add_argument(
        "--normalize",
        choices=["none", "true", "pred", "all"],
        default="none",
        help="Confusion matrix normalization mode.",
    )
    args = ap.parse_args()

    pj = args.predictions_json.expanduser().resolve()
    if not pj.is_file():
        print(f"ERROR: predictions JSON not found: {pj}")
        print("Generate it first with one of:")
        print("  python microexpression_project/scripts/predict_with_casme_labels.py")
        print("  python microexpression_project/scripts/evaluate_predict_videos.py")
        return 1

    with open(pj, encoding="utf-8") as f:
        payload = json.load(f)

    rows = _iter_rows(payload)
    y_true, y_pred = _extract_true_pred(rows)
    if not y_true:
        print("ERROR: no labeled rows with both true and pred in supported emotion set.")
        return 2

    # Fixed label order (matches training indices).
    labels = sorted(EMOTION_LABELS.keys(), key=lambda k: EMOTION_LABELS[k])
    display_labels = [EMOTION_DISPLAY_NAMES.get(k, k) for k in labels]

    normalize = None if args.normalize == "none" else args.normalize
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "confusion_matrix.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(display_labels) + "\n")
        for i, row in enumerate(cm):
            f.write(display_labels[i] + "," + ",".join(f"{x:.6f}" if normalize else str(int(x)) for x in row) + "\n")

    # Save PNG
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")
    ax.set_title(f"Confusion Matrix ({'raw counts' if normalize is None else 'normalized=' + normalize})")
    fig.tight_layout()
    png_path = out_dir / "confusion_matrix.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Loaded: {pj}")
    print(f"Rows used: {len(y_true)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

