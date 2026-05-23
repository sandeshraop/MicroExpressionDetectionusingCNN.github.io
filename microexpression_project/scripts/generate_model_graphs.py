#!/usr/bin/env python3
"""
Generate as many evaluation graphs as possible for the "current" model.

Preferred input (best fidelity):
- Held-out LOSO predictions JSON produced by scripts/generate_loso_predictions.py
  (includes per-sample probabilities for ROC/PR/calibration curves).

Fallback inputs (when dataset/predictions JSON is not available):
- Existing confusion matrix CSVs under results/confusion_matrix*/confusion_matrix.csv
  (typically normalized; we can still plot heatmaps and extract per-class recall if row-normalized).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import EMOTION_DISPLAY_NAMES, EMOTION_LABELS  # noqa: E402


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _emotion_keys_ordered() -> list[str]:
    return sorted(EMOTION_LABELS.keys(), key=lambda k: EMOTION_LABELS[k])


def _display_names_ordered() -> list[str]:
    keys = _emotion_keys_ordered()
    return [EMOTION_DISPLAY_NAMES.get(k, k) for k in keys]


def _find_latest_model(models_dir: Path) -> Optional[Path]:
    cands = sorted(models_dir.glob("real_data_model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _try_load_json(path: Path) -> Optional[dict]:
    if not path or not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


def _plot_heatmap(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    fmt: str,
    cmap: str = "Blues",
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=170)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = np.nanmax(cm) * 0.6 if np.size(cm) else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            s = "" if (v is None or (isinstance(v, float) and not math.isfinite(v))) else format(v, fmt)
            ax.text(
                j,
                i,
                s,
                ha="center",
                va="center",
                color="white" if (_safe_float(v) is not None and float(v) > thresh) else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _bar_plot(values: dict[str, float], title: str, ylabel: str, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    keys = _emotion_keys_ordered()
    names = [EMOTION_DISPLAY_NAMES.get(k, k) for k in keys]
    ys = [float(values.get(k, 0.0)) for k in keys]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=170)
    bars = ax.bar(names, ys, color="#2b6cb0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _read_confusion_csv(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    # expected shape: header row+col with display names
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        nums = []
        for x in parts[1:]:
            xf = _safe_float(x)
            if xf is None:
                return None
            nums.append(xf)
        rows.append(nums)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


@dataclass
class PredictionsPack:
    y_true: np.ndarray  # shape (n,)
    y_pred: np.ndarray  # shape (n,)
    proba: Optional[np.ndarray]  # shape (n, C)
    labels: list[str]  # ordered keys (canonical)
    rows: list[dict]


def _load_predictions_pack(pred_json: Path) -> Optional[PredictionsPack]:
    payload = _try_load_json(pred_json)
    if not payload:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None

    labels = payload.get("labels_order")
    if isinstance(labels, list) and labels:
        labels_keys = [str(x).strip().lower() for x in labels]
    else:
        labels_keys = _emotion_keys_ordered()

    key_to_idx = {k: i for i, k in enumerate(labels_keys)}
    y_true, y_pred, probas = [], [], []
    has_proba = True

    for r in rows:
        t = str(r.get("true") or r.get("ground_truth") or "").strip().lower()
        p = str(r.get("pred") or r.get("prediction") or "").strip().lower()
        if t not in key_to_idx or p not in key_to_idx:
            continue
        y_true.append(key_to_idx[t])
        y_pred.append(key_to_idx[p])

        pr = r.get("proba")
        if isinstance(pr, dict):
            rowp = np.zeros((len(labels_keys),), dtype=np.float64)
            for k, v in pr.items():
                kk = str(k).strip().lower()
                if kk in key_to_idx:
                    fv = _safe_float(v)
                    rowp[key_to_idx[kk]] = 0.0 if fv is None else float(fv)
            s = float(np.sum(rowp))
            if s > 0:
                rowp = rowp / s
            probas.append(rowp)
        else:
            has_proba = False

    if not y_true:
        return None

    proba_arr = np.vstack(probas) if (has_proba and probas) else None
    return PredictionsPack(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        proba=proba_arr,
        labels=labels_keys,
        rows=rows,
    )


def _metrics_from_predictions(pack: PredictionsPack) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        precision_recall_fscore_support,
        confusion_matrix,
        log_loss,
    )

    y_true = pack.y_true
    y_pred = pack.y_pred
    labels_idx = list(range(len(pack.labels)))

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_true_norm = confusion_matrix(y_true, y_pred, labels=labels_idx, normalize="true")

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0
    )

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "per_class": {},
        "sklearn_classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=[EMOTION_DISPLAY_NAMES.get(k, k) for k in pack.labels],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix_counts": cm_counts.tolist(),
        "confusion_matrix_true_normalized": cm_true_norm.tolist(),
    }

    for i, k in enumerate(pack.labels):
        out["per_class"][k] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }

    if pack.proba is not None and pack.proba.shape[0] == len(y_true):
        try:
            out["log_loss"] = float(log_loss(y_true, pack.proba, labels=labels_idx))
        except Exception:
            out["log_loss"] = None
    else:
        out["log_loss"] = None
    return out


def _plot_from_predictions(pack: PredictionsPack, out_dir: Path, title_prefix: str) -> dict:
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        PrecisionRecallDisplay,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels_keys = pack.labels
    display_labels = [EMOTION_DISPLAY_NAMES.get(k, k) for k in labels_keys]
    labels_idx = list(range(len(labels_keys)))

    # Confusion matrices
    cm_counts = confusion_matrix(pack.y_true, pack.y_pred, labels=labels_idx, normalize=None)
    cm_true = confusion_matrix(pack.y_true, pack.y_pred, labels=labels_idx, normalize="true")
    cm_pred = confusion_matrix(pack.y_true, pack.y_pred, labels=labels_idx, normalize="pred")

    _plot_heatmap(
        cm_counts.astype(np.float64),
        display_labels,
        f"{title_prefix} Confusion Matrix (counts)",
        out_dir / "confusion_matrix_counts.png",
        fmt=".0f",
    )
    _plot_heatmap(
        cm_true.astype(np.float64),
        display_labels,
        f"{title_prefix} Confusion Matrix (row-normalized)",
        out_dir / "confusion_matrix_row_normalized.png",
        fmt=".3f",
    )
    _plot_heatmap(
        cm_pred.astype(np.float64),
        display_labels,
        f"{title_prefix} Confusion Matrix (col-normalized)",
        out_dir / "confusion_matrix_col_normalized.png",
        fmt=".3f",
    )

    # Per-class bars
    metrics = _metrics_from_predictions(pack)
    per = metrics["per_class"]
    _bar_plot({k: float(per[k]["precision"]) for k in labels_keys}, f"{title_prefix} Per-class Precision", "Precision", out_dir / "per_class_precision.png")
    _bar_plot({k: float(per[k]["recall"]) for k in labels_keys}, f"{title_prefix} Per-class Recall", "Recall", out_dir / "per_class_recall.png")
    _bar_plot({k: float(per[k]["f1"]) for k in labels_keys}, f"{title_prefix} Per-class F1", "F1", out_dir / "per_class_f1.png")

    # Subject-level accuracy (if present)
    subj = {}
    for r in pack.rows:
        s = (r.get("subject") or r.get("subject_id") or "").strip().lower()
        t = str(r.get("true") or "").strip().lower()
        p = str(r.get("pred") or "").strip().lower()
        if not s or t not in set(labels_keys) or p not in set(labels_keys):
            continue
        subj.setdefault(s, {"n": 0, "correct": 0})
        subj[s]["n"] += 1
        subj[s]["correct"] += int(t == p)
    if subj:
        _ensure_matplotlib()
        import matplotlib.pyplot as plt

        items = sorted(((k, v["correct"] / v["n"], v["n"]) for k, v in subj.items()), key=lambda x: x[0])
        names = [k for k, _, _ in items]
        accs = [a for _, a, _ in items]
        ns = [n for _, _, n in items]
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=170)
        bars = ax.bar(names, accs, color="#2f855a")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_prefix} Per-subject accuracy (LOSO held-out)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        for b, a, n in zip(bars, accs, ns):
            ax.text(b.get_x() + b.get_width() / 2, min(0.98, a + 0.02), f"{a:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "per_subject_accuracy.png")
        plt.close(fig)

    # Curves requiring probabilities
    curve_stats: dict[str, Any] = {}
    if pack.proba is not None and pack.proba.shape[0] == len(pack.y_true):
        y_bin = label_binarize(pack.y_true, classes=labels_idx)

        # ROC (one-vs-rest)
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=170)
        for i, name in enumerate(display_labels):
            try:
                RocCurveDisplay.from_predictions(y_bin[:, i], pack.proba[:, i], ax=ax, name=name)
            except Exception:
                continue
        ax.set_title(f"{title_prefix} ROC curves (OvR)")
        fig.tight_layout()
        fig.savefig(out_dir / "roc_ovr.png")
        plt.close(fig)
        try:
            curve_stats["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, pack.proba, average="macro", multi_class="ovr"))
        except Exception:
            curve_stats["roc_auc_macro_ovr"] = None

        # PR (one-vs-rest)
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=170)
        for i, name in enumerate(display_labels):
            try:
                PrecisionRecallDisplay.from_predictions(y_bin[:, i], pack.proba[:, i], ax=ax, name=name)
            except Exception:
                continue
        ax.set_title(f"{title_prefix} Precision–Recall curves (OvR)")
        fig.tight_layout()
        fig.savefig(out_dir / "pr_ovr.png")
        plt.close(fig)
        try:
            curve_stats["average_precision_macro_ovr"] = float(average_precision_score(y_bin, pack.proba, average="macro"))
        except Exception:
            curve_stats["average_precision_macro_ovr"] = None

        # Calibration (reliability diagram) - plot only for top-1 confidence
        try:
            from sklearn.calibration import calibration_curve

            conf = np.max(pack.proba, axis=1)
            correct = (pack.y_true == np.argmax(pack.proba, axis=1)).astype(np.int32)
            prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
            fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=170)
            ax.plot(prob_pred, prob_true, marker="o", label="Model")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
            ax.set_xlabel("Predicted confidence (top-1)")
            ax.set_ylabel("Empirical accuracy")
            ax.set_title(f"{title_prefix} Calibration (top-1 confidence)")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "calibration_top1.png")
            plt.close(fig)
        except Exception:
            pass

    return {"curve_stats": curve_stats, "metrics": metrics}


def _plot_from_confusion_csv_only(cm_csv: Path, out_dir: Path, title_prefix: str) -> dict:
    cm = _read_confusion_csv(cm_csv)
    if cm is None:
        raise ValueError(f"Could not parse confusion matrix CSV: {cm_csv}")
    labels = _display_names_ordered()

    # If matrix is normalized (floats), infer row-normalized if rows approx sum to 1.
    row_sums = np.sum(cm, axis=1)
    row_normalized = np.all(np.isfinite(row_sums)) and bool(np.all(np.abs(row_sums - 1.0) < 1e-2))
    fmt = ".3f" if np.nanmax(cm) <= 1.0 + 1e-6 else ".0f"

    _plot_heatmap(cm, labels, f"{title_prefix} Confusion Matrix (from CSV)", out_dir / "confusion_matrix_from_csv.png", fmt=fmt)

    out: dict[str, Any] = {
        "source_confusion_csv": str(cm_csv),
        "row_normalized_detected": bool(row_normalized),
        "per_class_recall": None,
        "macro_recall": None,
    }
    if row_normalized:
        recalls = {k: float(cm[i, i]) for i, k in enumerate(_emotion_keys_ordered())}
        out["per_class_recall"] = recalls
        out["macro_recall"] = float(np.mean(list(recalls.values()))) if recalls else None
        _bar_plot(recalls, f"{title_prefix} Per-class Recall (diag of row-norm CM)", "Recall", out_dir / "per_class_recall_from_cm.png")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None, help="Path to model .pkl (default: latest real_data_model_*.pkl)")
    ap.add_argument(
        "--predictions_json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "predict" / "loso_predictions.json",
        help="LOSO predictions JSON (preferred).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_PROJECT_ROOT / "results" / "model_report",
        help="Base output directory. Script will create a timestamped subfolder.",
    )
    ap.add_argument(
        "--fallback_confusion_csv",
        type=Path,
        default=None,
        help="Optional confusion matrix CSV to use when predictions_json is missing.",
    )
    ap.add_argument(
        "--also_use_existing_confusion_matrices",
        action="store_true",
        help="If set, also render plots for results/confusion_matrix*/confusion_matrix.csv when present.",
    )
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = _find_latest_model(_PROJECT_ROOT / "models")
    if model_path is None:
        print("ERROR: could not find a model (real_data_model_*.pkl).")
        return 1

    stamp_dir = args.out_dir.expanduser().resolve() / f"{_now_stamp()}_graphs"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model": str(model_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outputs_dir": str(stamp_dir),
        "inputs": {
            "predictions_json": str(args.predictions_json),
            "fallback_confusion_csv": str(args.fallback_confusion_csv) if args.fallback_confusion_csv else None,
        },
        "mode": None,
        "notes": [],
    }

    pack = _load_predictions_pack(args.predictions_json)
    if pack is not None:
        report["mode"] = "from_predictions_json"
        title_prefix = "Current model (LOSO held-out)"
        out = _plot_from_predictions(pack, stamp_dir, title_prefix=title_prefix)
        report.update(out)
        _write_json(stamp_dir / "metrics_and_plots_summary.json", report)
        print("Wrote report to:", stamp_dir)
        return 0

    # Fallback: use confusion matrices already in results/
    report["mode"] = "from_confusion_csv_only"
    used_any = False

    # Prefer explicit fallback first
    if args.fallback_confusion_csv:
        info = _plot_from_confusion_csv_only(args.fallback_confusion_csv, stamp_dir, "Current model")
        report["fallback_primary"] = info
        used_any = True

    if args.also_use_existing_confusion_matrices:
        existing = [
            _PROJECT_ROOT / "results" / "confusion_matrix" / "confusion_matrix.csv",
            _PROJECT_ROOT / "results" / "confusion_matrix_loso" / "confusion_matrix.csv",
        ]
        infos = []
        for p in existing:
            if p.is_file():
                sub = stamp_dir / f"from_{p.parent.name}"
                sub.mkdir(parents=True, exist_ok=True)
                infos.append(_plot_from_confusion_csv_only(p, sub, f"Existing {p.parent.name}"))
                used_any = True
        report["existing_confusion_matrices"] = infos

    if not used_any:
        report["notes"].append(
            "No predictions JSON found, and no confusion matrix CSV provided/found. "
            "To generate full graphs (ROC/PR/calibration + accurate per-class metrics), "
            "first run scripts/generate_loso_predictions.py to create data/predict/loso_predictions.json."
        )
        _write_json(stamp_dir / "metrics_and_plots_summary.json", report)
        print("Wrote empty report (no inputs) to:", stamp_dir)
        return 2

    _write_json(stamp_dir / "metrics_and_plots_summary.json", report)
    print("Wrote report to:", stamp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
