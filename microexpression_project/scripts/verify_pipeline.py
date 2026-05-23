"""Smoke test: CNNCASMEIIDataset → DataLoader → model → hybrid_predict (run from project root)."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loader import CNNCASMEIIDataset
from inference_utils import hybrid_predict_from_features, load_enhanced_hybrid_from_path


def main() -> None:
    labels = root / "data" / "labels" / "casme2_labels.csv"
    data_root = root / "data" / "casme2"

    print("=== 1. Dataset (CNNCASMEIIDataset) ===")
    if not labels.is_file():
        print("SKIP: missing", labels)
        return
    ds = CNNCASMEIIDataset(str(data_root), str(labels))
    print("samples:", len(ds))
    if len(ds) == 0:
        print("No samples; skipping model/inference.")
        return

    f, fl, y, _meta = ds[0]
    print("shapes: frames", tuple(f.shape), "flows", tuple(fl.shape), "label", int(y))
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    bf, bfl, _by, _ = next(iter(dl))
    print("DataLoader batch: frames", tuple(bf.shape), "flows", tuple(bfl.shape))

    print()
    print("=== 2. Model load ===")
    models_dir = root / "models"
    ckpts = sorted(models_dir.glob("real_data_model_*.pkl"), key=lambda p: -p.stat().st_mtime)
    if not ckpts:
        print("No real_data_model_*.pkl in", models_dir)
        return
    mp = ckpts[0]
    model = load_enhanced_hybrid_from_path(mp)
    print("loaded:", mp.name, type(model).__name__, "fitted:", getattr(model, "is_fitted", "?"))

    print()
    print("=== 3. Inference (extract_all_features + hybrid_predict) ===")
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        feat = model.extract_all_features(bf.float(), bfl.float())
    if torch.is_tensor(feat):
        arr = feat.detach().cpu().numpy()
    else:
        arr = np.asarray(feat)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for i in range(min(2, arr.shape[0])):
        out = hybrid_predict_from_features(model, arr[i : i + 1])
        print(f"  sample {i} -> {out['prediction_emotion']} ({out['confidence']:.3f})")
    print("OK")


if __name__ == "__main__":
    main()
