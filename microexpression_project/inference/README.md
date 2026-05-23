# Micro-expression inference

Offline and API-style inference use the **enhanced hybrid** checkpoint format (CNN feature extractor + sklearn pipeline) loaded via `src/inference_utils.load_enhanced_hybrid_from_path`.

## Files

- `enhanced_inference_pipeline.py` — `EnhancedMicroExpressionInferencePipeline`: video preprocessing, `extract_all_features`, and `hybrid_predict_from_features`.
- `demo_inference.py` — small CLI/demo entry points.
- `requirements.txt` — Python dependencies for this folder (keep in sync with `web/requirements.txt` / `deployment/requirements.txt` as needed).

## Quick start

From the `inference` directory (parents add `src` to `PYTHONPATH` inside the scripts):

```bash
cd inference
python demo_inference.py --model ../models/real_data_model_20260414_075750.pkl --video path/to/video.mp4
```

Use a trained hybrid pickle under `../models/real_data_model_*.pkl` (or legacy augmented pickles in the same object/dict format).

## Python API

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference_utils import load_enhanced_hybrid_from_path, hybrid_predict_from_features
from preprocessing_pipeline import VideoPreprocessor
import torch

model = load_enhanced_hybrid_from_path("../models/real_data_model_latest.pkl")
model.feature_extractor.eval()
prep = VideoPreprocessor()
frames, flows = prep.preprocess_video("path/to/video.mp4", verbose=False)
frames_b = frames.unsqueeze(0).float()
flows_b = flows.unsqueeze(0).float()
with torch.no_grad():
    feat = model.extract_all_features(frames_b, flows_b)
import numpy as np
arr = feat.detach().cpu().numpy() if torch.is_tensor(feat) else np.asarray(feat)
if arr.ndim == 1:
    arr = arr.reshape(1, -1)
print(hybrid_predict_from_features(model, arr.astype(np.float64)))
```

The legacy `MicroExpressionInferencePipeline` / `inference_pipeline.py` module has been removed; use `EnhancedMicroExpressionInferencePipeline` or the helpers above.
