"""Validate real_data_model checkpoints and paired metadata (run from project root)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from inference_utils import hybrid_predict_from_features, try_load_enhanced_hybrid_raw  # noqa: E402

REQUIRED_KEYS = {"cnn_model", "classifier_type", "feature_extractor_state", "pipeline", "is_fitted"}


def load_hybrid(path: Path):
    raw = joblib.load(path)
    if isinstance(raw, dict):
        missing = REQUIRED_KEYS - raw.keys()
        if missing:
            return None, f"dict missing keys: {missing}"
    model, err = try_load_enhanced_hybrid_raw(raw)
    if err:
        return None, err
    kind = "object" if not isinstance(raw, dict) else "dict->EnhancedHybridModel"
    return model, kind


def check_meta(pkl_path: Path):
    stem = pkl_path.stem.replace("real_data_model_", "")
    meta = root / "models" / f"real_data_model_metadata_{stem}.json"
    if not meta.is_file():
        return None, f"no metadata file for {stem}"
    try:
        with open(meta, encoding="utf-8") as f:
            d = json.load(f)
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"
    return d, meta.name


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    print("=== Checkpoint scan ===")
    pkls = sorted(root.glob("models/real_data_model_*.pkl"))
    if not pkls:
        print("No models/real_data_model_*.pkl found.")
        return 1

    for pkl in pkls:
        print(f"\n-- {pkl.name}")
        m, kind = load_hybrid(pkl)
        if m is None:
            errors.append(f"{pkl.name}: {kind}")
            print("  ERROR:", kind)
            continue
        print("  load:", kind)
        if not getattr(m, "is_fitted", False):
            warnings.append(f"{pkl.name}: is_fitted is False")

        try:
            fr = torch.randn(1, 3, 3, 64, 64)
            fl = torch.randn(1, 6, 64, 64)
            with torch.no_grad():
                feat = m.extract_all_features(fr, fl)
            arr = feat.detach().cpu().numpy() if torch.is_tensor(feat) else np.asarray(feat)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            hp = hybrid_predict_from_features(m, arr.astype(np.float64))
            print(
                "  smoke infer:",
                hp["prediction_emotion"],
                f"conf={hp['confidence']:.3f}",
            )
        except Exception as e:
            errors.append(f"{pkl.name} infer: {e}")
            print("  ERROR infer:", e)

        md, mres = check_meta(pkl)
        if md is None:
            warnings.append(str(mres))
            print("  WARN meta:", mres)
        else:
            print("  metadata:", mres)
            ta = md.get("training_accuracy")
            if ta is not None and not (0.0 <= float(ta) <= 1.0):
                warnings.append(f"{mres}: training_accuracy out of [0,1]")
            for k in ("uar", "happiness_recall"):
                v = md.get(k)
                if v is not None and not (0.0 <= float(v) <= 1.0):
                    warnings.append(f"{mres}: {k} out of [0,1]")

    for meta in root.glob("models/real_data_model_metadata_*.json"):
        ts = meta.stem.replace("real_data_model_metadata_", "")
        buddy = root / "models" / f"real_data_model_{ts}.pkl"
        if not buddy.is_file():
            warnings.append(f"Orphan metadata (no matching pkl): {meta.name}")

    print("\n=== Summary ===")
    print("Errors:", len(errors))
    for e in errors:
        print(" ", e)
    print("Warnings:", len(warnings))
    for w in warnings:
        print(" ", w)
    if errors:
        return 1
    print("OK: all checkpoints load and infer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
