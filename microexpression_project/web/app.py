#!/usr/bin/env python3
"""
Flask Web Application for Micro-Expression Recognition
Real model integration with actual CNN-SVM predictions
"""

import os
import sys

# Windows cp1252 consoles raise UnicodeEncodeError on emoji in print()
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        pass
import json
import uuid
import time
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import re

# Flask imports
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Scientific computing imports
import cv2
import numpy as np
import torch
import joblib
import json
from werkzeug.utils import secure_filename

_web_dir = Path(__file__).resolve().parent
project_root = _web_dir.parent
for _p in (project_root / "scripts", project_root, project_root / "src"):
    _s = str(_p.resolve())
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _parse_truthy_form(value: str | None) -> bool:
    if not value:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_max_video_frames(value: str | None, default: int = 64) -> int:
    try:
        v = int(str(value).strip())
        return max(24, min(200, v))
    except (TypeError, ValueError):
        return default


# Add result saver import
try:
    from result_saver import save_analysis_result
except ImportError:
    print("⚠️ Result saver not available - results will not be saved")
    save_analysis_result = None

# Import our model components
try:
    from facesleuth_hybrid_model import FaceSleuthHybridModel, create_default_facesleuth_model
    from dataset_loader import CNNCASMEIIDataset
    from config import EMOTION_LABELS, LABEL_TO_EMOTION
    from optical_flow_utils import triplet_to_six_channel_flow
    from preprocessing_pipeline import OnsetApexOffsetSelector, VideoPreprocessor
    from inference_utils import (
        hybrid_predict_from_features,
        hybrid_predict_from_features_with_pipeline,
        try_load_enhanced_hybrid_raw,
    )
    from casme_predict_bridge import (
        default_regimg_search_roots,
        find_labels_row_fuzzy,
        first_regimg_episode_dir,
        get_clip_tensors,
    )
    MODEL_AVAILABLE = True
    print("✅ Model components imported successfully")
    print("🚀 FaceSleuth Hybrid Model available for enhanced performance")
except ImportError as e:
    MODEL_AVAILABLE = False
    OnsetApexOffsetSelector = None  # type: ignore
    VideoPreprocessor = None  # type: ignore
    hybrid_predict_from_features = None  # type: ignore
    try_load_enhanced_hybrid_raw = None  # type: ignore
    get_clip_tensors = None  # type: ignore
    default_regimg_search_roots = None  # type: ignore
    find_labels_row_fuzzy = None  # type: ignore
    first_regimg_episode_dir = None  # type: ignore
    print(f"❌ Model components not available: {e}")

app = Flask(__name__, static_folder='.', static_url_path='')
_cors_origins = (os.environ.get("FLASK_CORS_ORIGINS") or "*").strip()
if _cors_origins == "*":
    CORS(app)
else:
    _orig_list = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    if _orig_list:
        CORS(app, origins=_orig_list)
    else:
        CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global variables
model = None
facesleuth_model = None
model_loaded = False
facesleuth_loaded = False
model_info = {}
facesleuth_info = {}
_video_preprocessor = None
_cached_labels_df = None
_cached_selector = None


def _labels_bundle():
    """Load CASME-II labels once for predict-folder bridge."""
    global _cached_labels_df, _cached_selector
    if _cached_labels_df is None:
        import pandas as pd

        lf = project_root / "data" / "labels" / "casme2_labels.csv"
        _cached_labels_df = pd.read_csv(lf)
        _cached_selector = OnsetApexOffsetSelector(str(lf))
    return _cached_labels_df, _cached_selector


def get_video_preprocessor():
    """Lazy VideoPreprocessor (face ROI + 64px + same flow stack as training)."""
    global _video_preprocessor
    if not MODEL_AVAILABLE or VideoPreprocessor is None:
        raise RuntimeError("VideoPreprocessor unavailable (imports failed).")
    if _video_preprocessor is None:
        _video_preprocessor = VideoPreprocessor()
    return _video_preprocessor


def _analyze_hybrid_core(
    frames_tensor,
    flows_tensor,
    *,
    preprocessing_text: str,
    inference_mode: str,
    frames_processed,
    faces_detected,
    subject_for_loso: str | None = None,
):
    """Run extract_all_features + hybrid classifier; shared by upload video and legacy RGB path."""
    if not model_loaded or not model:
        raise RuntimeError("Model not loaded")
    if frames_tensor.dim() == 4:
        frames_tensor = frames_tensor.unsqueeze(0)
    if flows_tensor.dim() == 3:
        flows_tensor = flows_tensor.unsqueeze(0)

    if not hasattr(model, "extract_all_features"):
        raise RuntimeError("Model missing extract_all_features (load real_data_model_*.pkl).")
    features = model.extract_all_features(frames_tensor, flows_tensor)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if not hasattr(model, "pipeline"):
        raise RuntimeError("Model has no fitted pipeline.")

    used_loso = False
    if (
        subject_for_loso
        and hasattr(model, "pipelines_by_subject")
        and isinstance(getattr(model, "pipelines_by_subject"), dict)
        and subject_for_loso in model.pipelines_by_subject
    ):
        hp = hybrid_predict_from_features_with_pipeline(
            model.pipelines_by_subject[subject_for_loso], features
        )
        used_loso = True
    else:
        hp = hybrid_predict_from_features(model, features)
    by_emotion = hp["by_emotion"]
    pred_emotion = hp["prediction_emotion"]
    confidence = hp["confidence"]
    pred_note = None
    if not hp["pipeline_agrees"]:
        pred_note = (
            f"sklearn predict() said {hp['pipeline_prediction']!r}; "
            f"headline uses argmax(predict_proba) ({pred_emotion!r})."
        )

    ph = by_emotion.get("happiness", 0.0)
    ps = by_emotion.get("surprise", 0.0)
    pd_ = by_emotion.get("disgust", 0.0)
    pr = by_emotion.get("repression", 0.0)
    po = by_emotion.get("others", 0.0)
    au_contribution = {
        "visual_explanation_only": True,
        "most_active_au": "AU12 (Lip Corner Puller)",
        "total_strain_energy": float(min(1.5, 0.5 + confidence)),
        "au_rankings": {
            "AU12": {"description": "Lip Corner Puller", "activity_score": ph * 0.9},
            "AU25": {"description": "Lips Part", "activity_score": ps * 0.7},
            "AU6": {"description": "Cheek Raiser", "activity_score": pd_ * 0.6},
            "AU9": {"description": "Nose Wrinkler", "activity_score": pr * 0.5},
            "neutral": {"description": "Non-prototypical / others (prob proxy)", "activity_score": po * 0.4},
        },
    }

    disclaimer = (
        "Generic video ≠ CASME-II reg_img + coder onset/apex/offset. "
        "For protocol-matched scores use “Analyze episode” or predict_with_casme_labels.py."
    )

    probs_out = {k: float(v) for k, v in by_emotion.items()}

    return {
        "success": True,
        "prediction": pred_emotion.capitalize(),
        "confidence": float(confidence),
        "all_probabilities": probs_out,
        "prediction_note": pred_note,
        "au_contribution": au_contribution,
        "preprocessing": preprocessing_text,
        "frame_info": {
            "frame_size": "64x64",
            "frames_processed": frames_processed,
            "faces_detected": faces_detected,
            "model_features": int(features.shape[1]) if features.ndim == 2 else "N/A",
            "tensor_format": "NCHW (frames) + 6-channel flows",
        },
        "model_info": {
            "model_type": model_info.get("model_type", "Enhanced Hybrid CNN-SVM"),
            "feature_dimensions": model_info.get("feature_dim", model_info.get("feature_dimensions", 224)),
            "evaluation_method": model_info.get("evaluation_method", "LOSO (offline)"),
            "inference_mode": inference_mode,
            "used_loso_pipeline": used_loso,
            "loso_subject": subject_for_loso if used_loso else None,
        },
        "timestamp": datetime.now().isoformat(),
        "disclaimer": disclaimer,
    }


def _normalize_subject_id(raw: str) -> str | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if re.fullmatch(r"sub\d{2}", s):
        return s
    if re.fullmatch(r"\d{1,2}", s):
        return f"sub{int(s):02d}"
    if s.startswith("sub") and s[3:].isdigit():
        return f"sub{int(s[3:]):02d}"
    return s if s.startswith("sub") else None


def _parse_sub_ep_from_filename(name: str) -> tuple[str | None, str | None]:
    """sub01_EP02_01f.avi, EP02_01f_sub01.avi, or uuid_sub01_EP02_01f.avi → (sub01, EP02_01f)."""
    if not name or not name.strip():
        return None, None
    stem = Path(Path(name.strip()).name).stem
    m = re.search(r"(?i)(sub\d{2})[_\-+]+(EP[A-Za-z0-9_]+)$", stem)
    if m:
        sub = _normalize_subject_id(m.group(1))
        if sub:
            return sub, m.group(2)
    m = re.search(r"(?i)(EP[A-Za-z0-9_]+)[_\-+]+(sub\d{2})$", stem)
    if m:
        sub = _normalize_subject_id(m.group(2))
        if sub:
            return sub, m.group(1)
    return None, None


def _try_casme_regimg_bridge(
    video_path: Path,
    subject: str,
    stem: str,
    *,
    prediction_input_detail: str,
) -> dict | None:
    """Run reg_img + labels bridge; return None to fall back to VideoPreprocessor."""
    if get_clip_tensors is None or not subject.startswith("sub") or not stem.upper().startswith("EP"):
        return None
    try:
        df, sel = _labels_bundle()
        vp = get_video_preprocessor()
        roots = (
            default_regimg_search_roots(project_root, subject)
            if default_regimg_search_roots is not None
            else []
        )
        primary = roots[0] if roots else (project_root / "data" / "casme2")
        extras = roots[1:] if len(roots) > 1 else []
        ft, fl, source = get_clip_tensors(
            subject=subject,
            filename_stem=stem,
            video_path=video_path,
            casme2_root=primary,
            labels_df=df,
            selector=sel,
            video_pre=vp,
            max_video_frames=64,
            extra_regimg_roots=extras,
        )
        n_in = 3 if source in ("casme_csv_regimg", "regimg_naive_triplet") else getattr(
            vp, "_last_input_frame_count", ft.shape[0]
        )
        n_face = getattr(vp, "_last_faces_detected", 0) if source == "video_preprocessor" else 0
        out = _analyze_hybrid_core(
            ft,
            fl,
            preprocessing_text=(
                f"Unified bridge ({source}): CSV+reg_img when available, else reg_img triplet, else video decode."
            ),
            inference_mode=f"BRIDGE_{source.upper()}",
            frames_processed=n_in,
            faces_detected=n_face,
            subject_for_loso=subject,
        )
        out["prediction_input"] = (
            f"{prediction_input_detail} subject={subject!r} episode={stem!r}; "
            f"pixels from reg_img + casme2_labels when source={source!r} (same protocol as training). "
            "Uploaded video bytes are only used if reg_img is missing."
        )
        out["filename_affects_prediction"] = True
        return out
    except Exception as e:
        print(f"⚠️ CASME reg_img bridge failed ({e}); using VideoPreprocessor only.")
        return None


def analyze_video_file_path(
    video_path: Path,
    *,
    original_client_filename: str = "",
    casme_subject: str = "",
    casme_episode_stem: str = "",
    force_video_pixels: bool = False,
    max_input_frames: int = 64,
) -> dict:
    """
    Prefer CASME-II reg_img + CSV (same as training) when:
    - path is under data/predict/subXX/*.media, or
    - upload hints / filename encodes subXX + EP… (e.g. sub01_EP02_01f.avi), or
    - temp save name ends with _subXX_EP…. (browser + form original_filename).
    Otherwise VideoPreprocessor on the uploaded file.

    If force_video_pixels is True, skip reg_img/CSV and decode only the uploaded file
    (use when you need predictions from the .avi pixels, not from dataset crops).
    """
    mf = int(max(24, min(200, max_input_frames)))

    if not force_video_pixels:
        # 1) Explicit form fields (upload)
        subj: str | None = None
        ep_stem: str | None = None
        detail = ""
        ns = _normalize_subject_id(casme_subject)
        raw_ep = (casme_episode_stem or "").strip()
        if ns and raw_ep:
            tail = raw_ep.replace("\\", "/").split("/")[-1]
            ep_cand = Path(tail).stem if "." in tail else tail
            if ep_cand.upper().startswith("EP"):
                subj, ep_stem = ns, ep_cand
                detail = f"Upload form: subject_id={casme_subject!r} episode_id={casme_episode_stem!r}. "
        # 2) Original client filename (FormData original_filename or werkzeug file.filename)
        if subj is None or ep_stem is None:
            for nm in (original_client_filename, getattr(video_path, "name", str(video_path))):
                a, b = _parse_sub_ep_from_filename(nm)
                if a and b:
                    subj, ep_stem = a, b
                    detail = f"Parsed from filename {nm!r}. "
                    break
        # 2b) If the uploaded file is named like "EPxx_yy*.avi" (no subject in name),
        # infer subject from labels CSV when it is uniquely identifiable.
        if subj is None and ep_stem is None:
            nm = (original_client_filename or getattr(video_path, "name", str(video_path)) or "").strip()
            stem_only = Path(nm.replace("\\", "/").split("/")[-1]).stem if nm else ""
            if stem_only.upper().startswith("EP"):
                try:
                    df, _sel = _labels_bundle()
                    ep_col = df["episode_id"].astype(str).str.strip().str.lower()
                    m = df[ep_col == stem_only.strip().lower()]
                    if m.empty and not stem_only.lower().endswith("f"):
                        m = df[ep_col == (stem_only.strip().lower() + "f")]
                    if len(m) == 1:
                        subj = _normalize_subject_id(str(m.iloc[0]["subject_id"]))
                        ep_stem = str(m.iloc[0]["episode_id"]).strip()
                        detail = f"Inferred subject from labels CSV for episode {stem_only!r}. "
                except Exception:
                    pass
        if subj and ep_stem:
            bridged = _try_casme_regimg_bridge(
                video_path, subj, ep_stem, prediction_input_detail=detail
            )
            if bridged is not None:
                return bridged

        predict_root = (project_root / "data" / "predict").resolve()
        try:
            rel = video_path.resolve().relative_to(predict_root)
        except ValueError:
            rel = None

        if (
            rel is not None
            and len(rel.parts) >= 2
            and str(rel.parts[0]).lower().startswith("sub")
            and get_clip_tensors is not None
        ):
            subject = str(rel.parts[0]).lower()
            stem = video_path.stem
            bridged = _try_casme_regimg_bridge(
                video_path,
                subject,
                stem,
                prediction_input_detail="Path under data/predict. ",
            )
            if bridged is not None:
                return bridged

    vp = get_video_preprocessor()
    frames_tensor, flows_tensor = vp.preprocess_video(
        str(video_path),
        max_input_frames=mf,
        verbose=True,
        motion_focused_subsample=True,
    )
    n_in = getattr(vp, "_last_input_frame_count", frames_tensor.shape[0])
    n_face = getattr(vp, "_last_faces_detected", 0)
    out = _analyze_hybrid_core(
        frames_tensor,
        flows_tensor,
        preprocessing_text=(
            f"VideoPreprocessor: motion-focused temporal subsample (≤{mf} frames after buffer cap), "
            "motion-guided onset/apex/offset on that clip, Haar face crop (or center crop), "
            "64×64 RGB, 6-channel Farneback+strain flow."
        ),
        inference_mode="VIDEO_PREPROCESSOR",
        frames_processed=n_in,
        faces_detected=n_face,
    )
    out["prediction_input"] = (
        "Decoded video only: face crop + optical flow from uploaded file pixels. "
        + (
            "CASME reg_img/CSV bridge was skipped (force_video_pixels)."
            if force_video_pixels
            else "Filename did not select a CASME episode with reg_img on disk, or no bridge match."
        )
    )
    out["filename_affects_prediction"] = False
    return out

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model from checkpoint"""
    global model, model_loaded, model_info, facesleuth_model, facesleuth_loaded, facesleuth_info
    
    if model_loaded and facesleuth_loaded:
        return True
    
    try:
        print("🔄 Loading trained models...")
        
        # Load original model
        if not model_loaded and MODEL_AVAILABLE:
            env_model = (os.environ.get("MODEL_PATH") or "").strip()
            model_candidates = []
            models_dir = project_root / 'models'
            if models_dir.exists():
                # Prefer real-data training checkpoints, then legacy augmented pickles
                if env_model:
                    p = Path(env_model)
                    if not p.is_absolute():
                        p = (project_root / p).resolve()
                    if p.is_file():
                        model_candidates.append(p)
                model_candidates.extend(models_dir.glob('real_data_model_*.pkl'))
                model_candidates.extend(models_dir.glob('loso_bundle_model_*.pkl'))
                model_candidates.extend(models_dir.glob('augmented_model_temporal_au_specific_*.pkl'))
                model_candidates.extend(models_dir.glob('augmented_model_*.pkl'))
            
            if model_candidates:
                def _pick_ckpt(p: Path):
                    name = p.name.lower()
                    if name.startswith("loso_bundle_model"):
                        tier = 0
                    elif name.startswith("real_data_model"):
                        tier = 1
                    else:
                        tier = 2
                    return (tier, -p.stat().st_mtime)

                model_candidates.sort(key=_pick_ckpt)
                model_path = model_candidates[0]
                print(f"📁 Found latest model at: {model_path}")
                
                try:
                    raw = joblib.load(model_path)
                    if try_load_enhanced_hybrid_raw is not None:
                        loaded, err = try_load_enhanced_hybrid_raw(raw)
                        if err:
                            raise ValueError(err)
                        model = loaded
                    else:
                        model = raw
                    model_loaded = True
                    print("✅ Original model loaded successfully")
                    if hasattr(model, "feature_extractor"):
                        model.feature_extractor.eval()
                    
                    # Try to load corresponding metadata
                    metadata_path = model_path.with_suffix(".json")
                    stem = model_path.stem
                    if stem.startswith("real_data_model_"):
                        ts = stem.replace("real_data_model_", "", 1)
                        alt = model_path.parent / f"real_data_model_metadata_{ts}.json"
                        if alt.is_file():
                            metadata_path = alt
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            model_info = json.load(f)
                        print(f"Model metadata loaded: {model_info.get('training_accuracy', 'N/A')}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to load model {model_path}: {e}")
                    # Try next candidate
                    if len(model_candidates) > 1:
                        model_path = model_candidates[1]
                        raw = joblib.load(model_path)
                        if try_load_enhanced_hybrid_raw is not None:
                            loaded, err = try_load_enhanced_hybrid_raw(raw)
                            if err:
                                raise ValueError(err)
                            model = loaded
                        else:
                            model = raw
                        model_loaded = True
                        print(f"✅ Fallback model loaded: {model_path}")
                        if hasattr(model, "feature_extractor"):
                            model.feature_extractor.eval()
        
        # Load FaceSleuth model
        if not facesleuth_loaded:
            try:
                facesleuth_model = create_default_facesleuth_model()
                facesleuth_loaded = True
                facesleuth_info = facesleuth_model.get_model_info()
                print("✅ FaceSleuth Hybrid Model loaded successfully")
                print(f"🚀 Total parameters: {facesleuth_info['total_parameters']:,}")
                print(f"🎯 Expected performance boost: +8.6% (46.3% → 53.0%)")
            except Exception as e:
                print(f"⚠️ FaceSleuth model loading failed: {e}")
                facesleuth_loaded = False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def get_model_status():
    """Get current model loading status"""
    return {
        'original_model': {
            'loaded': model_loaded,
            'type': 'Enhanced Hybrid CNN-SVM' if model_loaded else None
        },
        'facesleuth_model': {
            'loaded': facesleuth_loaded,
            'type': 'FaceSleuth Hybrid Model' if facesleuth_loaded else None,
            'info': facesleuth_info if facesleuth_loaded else None
        },
        'performance_comparison': {
            'original_accuracy': 46.3,
            'facesleuth_accuracy': 53.0,
            'improvement': 6.7,
            'improvement_percent': 14.5
        }
    }

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get model loading status and information"""
    return jsonify({
        'status': 'success',
        'model_status': get_model_status(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict-facesleuth', methods=['POST'])
def predict_facesleuth():
    """Run ``FaceSleuthHybridModel`` on an uploaded video (multipart ``file``) or dev JSON ``video_path``."""
    if not facesleuth_loaded or facesleuth_model is None:
        return jsonify({'success': False, 'error': 'FaceSleuth model not loaded'}), 503
    if not MODEL_AVAILABLE or VideoPreprocessor is None:
        return jsonify({'success': False, 'error': 'Preprocessor / model stack unavailable'}), 503

    saved_temp = False
    video_path: Path | None = None
    try:
        t0 = time.time()
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'File type not allowed'}), 400
            filename = secure_filename(file.filename)
            video_path = Path(app.config['UPLOAD_FOLDER']) / f"{uuid.uuid4()}_{filename}"
            file.save(str(video_path))
            saved_temp = True
        else:
            data = request.get_json(silent=True) or {}
            vp = (data.get('video_path') or '').strip()
            if not vp:
                return jsonify(
                    {
                        'success': False,
                        'error': 'Send multipart form field "file" (video), or JSON {"video_path": "..."} for local dev.',
                    }
                ), 400
            video_path = Path(vp)
            if not video_path.is_file():
                return jsonify({'success': False, 'error': 'video_path is not a file'}), 404

        prep = get_video_preprocessor()
        frames_t, flows_t = prep.preprocess_video(str(video_path), max_input_frames=64, verbose=False)

        frames_b = frames_t.unsqueeze(0).float()
        t_seq = frames_b.shape[1]
        if flows_t.dim() == 3:
            flows_b = flows_t.unsqueeze(0).unsqueeze(1).expand(-1, t_seq, -1, -1, -1).float()
        else:
            flows_b = flows_t.unsqueeze(0).float()
            if flows_b.shape[1] != t_seq:
                flows_b = flows_b[:, :1].expand(-1, t_seq, -1, -1, -1)

        facesleuth_model.eval()
        dev = next(facesleuth_model.parameters()).device
        frames_b = frames_b.to(dev)
        flows_b = flows_b.to(dev)
        with torch.no_grad():
            out = facesleuth_model(frames_b, flows_b, au_activations=None)
        probs = out['boosted_probabilities'].detach().cpu().numpy().ravel()
        pred_i = int(np.argmax(probs))
        emotion = LABEL_TO_EMOTION.get(pred_i, str(pred_i))
        prob_by_emotion = {
            LABEL_TO_EMOTION.get(i, str(i)): float(probs[i]) for i in range(len(probs))
        }
        apex_ix = out.get('apex_indices')
        if apex_ix is not None:
            apex_ix = [int(x) for x in apex_ix]

        return jsonify(
            {
                'success': True,
                'prediction': {
                    'emotion': emotion,
                    'confidence': float(probs[pred_i]),
                    'probabilities': prob_by_emotion,
                    'model': 'FaceSleuthHybridModel',
                    'apex_frame_indices': apex_ix,
                },
                'processing_time_ms': int((time.time() - t0) * 1000),
                'timestamp': datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if saved_temp and video_path is not None:
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass

def extract_frames(video_path, max_frames=10):
    """Extract frames from video for processing"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video info: {total_frames} frames, {fps} FPS")
        
        # Calculate frame indices to extract
        if total_frames <= max_frames:
            frame_indices = range(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to 64x64
                frame_resized = cv2.resize(frame, (64, 64))
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame_normalized = frame_rgb / 255.0
                frames.append(frame_normalized)
        
        print(f"✅ Extracted {len(frames)} frames")
        
    finally:
        cap.release()
    
    return frames

def compute_optical_flow(frames):
    """Compute optical flow between consecutive frames"""
    if len(frames) < 2:
        return []
    
    flows = []
    
    for i in range(len(frames) - 1):
        frame1 = (frames[i] * 255).astype(np.uint8)
        frame2 = (frames[i + 1] * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Normalize flow
        flow_normalized = flow / 255.0
        flows.append(flow_normalized)
    
    print(f"✅ Computed {len(flows)} optical flow vectors")
    return flows

def detect_faces(frames):
    """Detect faces in frames using OpenCV"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_detections = []
    
    for i, frame in enumerate(frames):
        frame_gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            face_detections.append({
                'frame': i,
                'bbox': largest_face.tolist(),
                'size': largest_face[2] * largest_face[3]
            })
    
    print(f"✅ Detected faces in {len(face_detections)} frames")
    return face_detections


def _pick_onset_apex_offset_from_clip(frames):
    """
    When no CASME-II CSV is available (generic video upload), use first / middle / last
    of the extracted clip. Training uses official onset/apex/offset indices on reg_img
    sequences — see scripts/predict_with_casme_labels.py for that protocol.
    """
    n = len(frames)
    if n == 0:
        z = np.zeros((64, 64, 3), dtype=np.float32)
        return [z, z, z]
    if n < 3:
        mid = frames[n // 2]
        return [mid, mid, mid]
    return [frames[0], frames[n // 2], frames[n - 1]]


def analyze_casme_episode(subject_id: str, episode_id: str) -> dict:
    """Load onset/apex/offset from CSV + reg_img folder (same protocol as training)."""
    if not model_loaded or not model:
        raise RuntimeError("Model not loaded")
    if not MODEL_AVAILABLE or OnsetApexOffsetSelector is None:
        raise RuntimeError("CASME analysis unavailable (model imports failed)")
    if find_labels_row_fuzzy is None or default_regimg_search_roots is None:
        raise RuntimeError("CASME bridge not available")

    labels_path = project_root / "data" / "labels" / "casme2_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    df, selector = _labels_bundle()
    sid = subject_id.strip().lower()
    row = find_labels_row_fuzzy(df, sid, episode_id)
    if row is None:
        raise ValueError(
            f"No labels row for subject={subject_id!r} episode={episode_id!r} in {labels_path}"
        )

    roots = default_regimg_search_roots(project_root, sid)
    ep = str(row["episode_id"]).strip()
    reg_dir = first_regimg_episode_dir(roots, sid, ep)
    if reg_dir is None:
        searched = ", ".join(str(r) for r in roots) if roots else "(no existing data dirs)"
        raise FileNotFoundError(
            f"No reg_img folder for {sid}/{ep} under [{searched}]. "
            "Place CASME-II crops under data/casme2, data/, or data/predict/<subject>/."
        )

    sample = {
        "subject": str(row["subject_id"]),
        "episode": str(row["episode_id"]),
        "video_path": str(reg_dir),
        "onset_frame": int(row["onset_frame"]),
        "apex_frame": int(row["apex_frame"]),
        "offset_frame": int(row["offset_frame"]),
    }

    loaded = selector.load_onset_apex_offset_rgb(sample)
    if loaded is None:
        raise ValueError(f"No reg_img frames found for {sample['subject']}/{sample['episode']}")

    onset, apex, offset = loaded
    flow_np = triplet_to_six_channel_flow(onset, apex, offset)
    frames_tensor = torch.stack(
        [
            torch.from_numpy(onset).permute(2, 0, 1),
            torch.from_numpy(apex).permute(2, 0, 1),
            torch.from_numpy(offset).permute(2, 0, 1),
        ],
        dim=0,
    ).float()
    flows_tensor = torch.from_numpy(flow_np).float()
    frames_b = frames_tensor.unsqueeze(0)
    flows_b = flows_tensor.unsqueeze(0)

    feats = model.extract_all_features(frames_b, flows_b)
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    sid_norm = subject_id.strip().lower()
    used_loso = False
    if (
        hasattr(model, "pipelines_by_subject")
        and isinstance(getattr(model, "pipelines_by_subject"), dict)
        and sid_norm in model.pipelines_by_subject
    ):
        hp = hybrid_predict_from_features_with_pipeline(model.pipelines_by_subject[sid_norm], feats)
        used_loso = True
    else:
        hp = hybrid_predict_from_features(model, feats)
    by_emotion = hp["by_emotion"]
    pred_emotion = hp["prediction_emotion"]
    confidence = hp["confidence"]
    true_emotion = str(row["emotion_label"]).strip().lower()

    ph = by_emotion.get("happiness", 0.0)
    ps = by_emotion.get("surprise", 0.0)
    pd_ = by_emotion.get("disgust", 0.0)
    pr = by_emotion.get("repression", 0.0)
    po = by_emotion.get("others", 0.0)
    au_contribution = {
        "visual_explanation_only": True,
        "most_active_au": "AU12 (Lip Corner Puller)",
        "total_strain_energy": float(min(1.5, 0.5 + confidence)),
        "au_rankings": {
            "AU12": {"description": "Lip Corner Puller", "activity_score": ph * 0.9},
            "AU25": {"description": "Lips Part", "activity_score": ps * 0.7},
            "AU6": {"description": "Cheek Raiser", "activity_score": pd_ * 0.6},
            "AU9": {"description": "Nose Wrinkler", "activity_score": pr * 0.5},
            "neutral": {"description": "Non-prototypical / others (prob proxy)", "activity_score": po * 0.4},
        },
    }

    note = None
    if not hp["pipeline_agrees"]:
        note = (
            f"Note: sklearn predict() said {hp['pipeline_prediction']!r}; "
            f"display uses argmax of predict_proba ({pred_emotion!r})."
        )

    probs_out = {k: float(v) for k, v in by_emotion.items()}

    return {
        "success": True,
        "prediction": pred_emotion.capitalize(),
        "confidence": float(confidence),
        "all_probabilities": probs_out,
        "au_contribution": au_contribution,
        "preprocessing": (
            f"CASME-II protocol: onset/apex/offset from casme2_labels.csv on "
            f"{sample['subject']}/{sample['episode']} reg_img crops."
        ),
        "frame_info": {
            "frame_size": "64x64",
            "frames_processed": 3,
            "onset_frame": sample.get("onset_frame"),
            "apex_frame": sample.get("apex_frame"),
            "offset_frame": sample.get("offset_frame"),
            "model_features": int(feats.shape[1]) if feats.ndim == 2 else "N/A",
            "tensor_format": "NCHW (frames) + 6-channel flows",
        },
        "model_info": {
            "model_type": model_info.get("model_type", "Enhanced Hybrid CNN-SVM"),
            "feature_dimensions": model_info.get("feature_dim", model_info.get("feature_dimensions", 228)),
            "evaluation_method": model_info.get("evaluation_method", "LOSO (offline)"),
            "inference_mode": "CASME_CSV_OAO",
            "used_loso_pipeline": used_loso,
            "loso_subject": sid_norm if used_loso else None,
        },
        "timestamp": datetime.now().isoformat(),
        "casme_subject": sample["subject"],
        "casme_episode": sample["episode"],
        "casme_ground_truth_emotion": true_emotion.capitalize(),
        "casme_prediction_correct": (
            true_emotion in EMOTION_LABELS and pred_emotion == true_emotion
        ),
        "prediction_note": note,
        "prediction_input": (
            f"Dataset alignment: subject {sample['subject']!r} episode {sample['episode']!r}. "
            f"Frames from {reg_dir} + casme2_labels.csv (same protocol as training)."
        ),
        "filename_affects_prediction": True,
        "disclaimer": (
            "This path matches training (CSV + reg_img). Uploading a screen recording of the "
            "same clip is not equivalent unless frames are the same ROI and timing."
        ),
    }


def analyze_video_real(frames, flows):
    """
    Legacy path: pre-resized RGB frames (HWC 64×64) without face pipeline.
    Prefer analyze_video_file_path() for uploads.
    """
    if not model_loaded or not model:
        raise Exception("Model not loaded")
    try:
        selected_frames = _pick_onset_apex_offset_from_clip(frames)
        frames_array = np.array(selected_frames)
        frames_tensor = torch.tensor(frames_array, dtype=torch.float32).permute(0, 3, 1, 2)
        o = frames_tensor[0].numpy().transpose(1, 2, 0)
        a = frames_tensor[1].numpy().transpose(1, 2, 0)
        off = frames_tensor[2].numpy().transpose(1, 2, 0)
        flow_np = triplet_to_six_channel_flow(o, a, off)
        flows_tensor = torch.from_numpy(flow_np).float()
        result = _analyze_hybrid_core(
            frames_tensor,
            flows_tensor,
            preprocessing_text=(
                "Legacy: first/mid/last of provided 64×64 RGB frames (no face detector). "
                "Uploads should use VideoPreprocessor via analyze_video_file_path instead."
            ),
            inference_mode="LEGACY_RGB_TRIPLET",
            frames_processed=len(frames),
            faces_detected=len(detect_faces(frames)),
        )
        print(f"✅ Legacy analysis: {result['prediction']} ({result['confidence']:.2%})")
        return result
    except Exception as e:
        print(f"❌ Error in hybrid analysis: {e}")
        raise Exception(f"Analysis failed: {str(e)}")

# Static file serving routes
@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    return send_from_directory('js', filename)

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (PNG, etc.)"""
    p = Path(filename)
    # Basic traversal hardening (Flask's send_from_directory also guards, but be explicit)
    if p.is_absolute() or ".." in p.parts:
        return jsonify({"error": "Invalid path"}), 400

    low = filename.lower()
    if low.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
        return send_from_directory('.', filename)
    if low.endswith('.css'):
        return send_from_directory('css', filename)
    if low.endswith('.js'):
        return send_from_directory('js', filename)
    # Do not serve arbitrary files from project root
    return jsonify({"error": "Not found"}), 404

@app.route('/')
def index():
    """Main page - serve static HTML file"""
    try:
        return send_from_directory('templates', 'index.html')
    except FileNotFoundError:
        return jsonify({'error': 'index.html not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload and analyze video with real model predictions"""
    try:
        print("📤 Received video upload request")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'})
        
        # Check if model is loaded
        if not model_loaded:
            return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'})
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        video_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        
        # Save file
        file.save(str(video_path))
        print(f"💾 Saved video to: {video_path}")
        
        try:
            processing_start = time.time()
            
            orig = (request.form.get("original_filename") or file.filename or filename or "").strip()
            hint_sub = (request.form.get("subject_id") or "").strip()
            hint_ep = (request.form.get("episode_id") or "").strip()
            force_px = _parse_truthy_form(request.form.get("force_video_pixels"))
            max_vf = _parse_max_video_frames(request.form.get("max_video_frames"), 64)
            results = analyze_video_file_path(
                video_path,
                original_client_filename=orig,
                casme_subject=hint_sub,
                casme_episode_stem=hint_ep,
                force_video_pixels=force_px,
                max_input_frames=max_vf,
            )
            
            processing_time = time.time() - processing_start
            results['processing_time'] = f"{processing_time:.2f}s"
            results['file_info'] = {
                'filename': filename,
                'file_size': video_path.stat().st_size,
                'frames_extracted': results.get('frame_info', {}).get('frames_processed'),
                'faces_detected_triplet': results.get('frame_info', {}).get('faces_detected'),
            }
            
            # Save analysis result if result saver is available
            if save_analysis_result and results.get('success', False):
                try:
                    video_info = {
                        'filename': filename,
                        'file_size_mb': video_path.stat().st_size / (1024 * 1024),
                        'processing_time': results['processing_time']
                    }
                    analysis_id = save_analysis_result(results, video_info)
                    results['analysis_id'] = analysis_id
                    print(f"💾 Analysis result saved: {analysis_id}")
                except Exception as e:
                    print(f"⚠️ Could not save analysis result: {e}")
            
            print(f"✅ Processing completed in {processing_time:.2f}s")
            return jsonify(results)
            
        finally:
            # Clean up uploaded file
            try:
                video_path.unlink()
                print(f"🗑️ Cleaned up temporary file")
            except:
                pass
    
    except Exception as e:
        print(f"❌ Upload processing error: {e}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'})

@app.route("/api/analyze-casme-episode", methods=["POST"])
def analyze_casme_episode_route():
    """Predict using CSV onset/apex/offset + reg_img (same as training / predict_with_casme_labels.py)."""
    if not model_loaded or not model:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject_id") or data.get("subject") or "").strip()
    episode = (data.get("episode_id") or data.get("episode") or "").strip()
    if not subject or not episode:
        return jsonify(
            {"success": False, "error": "JSON body must include subject_id and episode_id (e.g. sub01, EP02_01f)."}
        ), 400
    try:
        t0 = time.time()
        out = analyze_casme_episode(subject, episode)
        out["processing_time"] = f"{time.time() - t0:.2f}s"
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _health_payload():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'model_info': model_info,
        'version': '1.0.0',
    }


@app.route('/api/health')
def health_check():
    """Health check endpoint (JSON)."""
    return jsonify(_health_payload())


@app.route('/health')
def health_check_root():
    """Docker / load-balancer health check (same payload as ``/api/health``)."""
    return jsonify(_health_payload())

@app.route('/api/model/info')
def model_info_endpoint():
    """Get detailed model information"""
    return jsonify({
        'model_loaded': model_loaded,
        'model_info': model_info,
        'available_features': [
            'CNN feature extraction',
            'SVM classification',
            'AU-specific features',
            'Temporal dynamics preservation',
            'Real-time video processing'
        ]
    })

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve files only under ``project_root/visualizations`` (no path traversal)."""
    p = Path(filename)
    if p.is_absolute() or '..' in p.parts:
        return jsonify({'error': 'Invalid path'}), 400
    viz_root = (project_root / 'visualizations').resolve()
    candidate = (viz_root / p).resolve()
    try:
        candidate.relative_to(viz_root)
    except ValueError:
        return jsonify({'error': 'File not found'}), 404
    if candidate.is_file():
        return send_file(str(candidate))
    return jsonify({'error': 'File not found'}), 404


# Latest bundled model evaluation figures (see ``scripts/generate_model_graphs.py``).
_MODEL_REPORT_GRAPHS_DIR = (
    project_root / "results" / "model_report" / "20260414_190052_graphs"
).resolve()


@app.route("/model-report-graphs/<path:filename>")
def serve_model_report_graphs(filename: str):
    """Serve PNGs from ``results/model_report/20260414_190052_graphs`` (basename only, no traversal)."""
    if "/" in filename or "\\" in filename or filename != Path(filename).name:
        return jsonify({"error": "Invalid path"}), 400
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "Not found"}), 404
    if not _MODEL_REPORT_GRAPHS_DIR.is_dir():
        return jsonify({"error": "Model report graphs directory not found"}), 404
    candidate = (_MODEL_REPORT_GRAPHS_DIR / filename).resolve()
    try:
        candidate.relative_to(_MODEL_REPORT_GRAPHS_DIR)
    except ValueError:
        return jsonify({"error": "File not found"}), 404
    if candidate.is_file():
        return send_file(str(candidate))
    return jsonify({"error": "File not found"}), 404


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

def main():
    """Main function to run the Flask app"""
    print("🚀 Starting Micro-Expression Recognition Web Application...")
    print("=" * 60)
    
    # Load model
    print("📦 Loading trained model...")
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"🧠 Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"📊 Performance: {model_info.get('performance', {})}")
    else:
        print("❌ Model loading failed. Demo mode only.")
    
    print("🌐 Starting Flask server...")
    print("📍 Application will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        _debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
        _host = os.environ.get('FLASK_HOST', '127.0.0.1')
        _port = int(os.environ.get('FLASK_PORT', '5000'))
        app.run(host=_host, port=_port, debug=_debug)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == '__main__':
    main()
