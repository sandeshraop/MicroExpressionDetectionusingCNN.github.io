"""
Unified clip → (frames, flows) tensors for the hybrid model.

Priority (matches training when possible):
1) CASME-II reg_img folder + onset/apex/offset from labels CSV
2) reg_img folder only → first / middle / last image (training-like crops, heuristic timing)
3) Generic video file → VideoPreprocessor
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from optical_flow_utils import triplet_to_six_channel_flow
from preprocessing_pipeline import OnsetApexOffsetSelector, VideoPreprocessor


def episode_id_candidates(stem: str) -> list[str]:
    """Filename stem → possible CASME episode_id strings."""
    out = [stem]
    if stem.endswith("_f") and not re.search(r"\d+f$", stem):
        out.append(re.sub(r"_f$", "f", stem))
    if not stem.endswith("f") and re.match(r"^EP\d+_\d+$", stem):
        out.append(stem + "f")
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _load_rgb_path(p: Path, size: int = 64) -> np.ndarray:
    bgr = cv2.imread(str(p))
    if bgr is None:
        return np.zeros((size, size, 3), dtype=np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def load_naive_regimg_triplet(episode_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    paths = OnsetApexOffsetSelector.natural_image_paths(episode_dir)
    if len(paths) < 3:
        return None
    i0, i1, i2 = 0, len(paths) // 2, len(paths) - 1
    return _load_rgb_path(paths[i0]), _load_rgb_path(paths[i1]), _load_rgb_path(paths[i2])


def rgb_triplet_to_tensors(
    onset: np.ndarray, apex: np.ndarray, offset: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    frames_tensor = torch.stack(
        [
            torch.from_numpy(onset).permute(2, 0, 1),
            torch.from_numpy(apex).permute(2, 0, 1),
            torch.from_numpy(offset).permute(2, 0, 1),
        ],
        dim=0,
    ).float()
    flow_np = triplet_to_six_channel_flow(onset, apex, offset)
    flows_tensor = torch.from_numpy(flow_np).float()
    return frames_tensor, flows_tensor


def _dedupe_existing_dirs(paths: Sequence[Optional[Path]]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p is None:
            continue
        r = Path(p).resolve()
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def default_regimg_search_roots(project_root: Path, subject_id: Optional[str] = None) -> list[Path]:
    """Search roots for ``<subject>/<episode>`` reg_img-style folders (jpgs/pngs).

    Order: official CASME tree → small hand-picked mirrors (e.g. triplets on disk) →
    per-subject predict copies → flat ``data/subXX/EP`` under ``data/``.

    ``data/raw_selected`` holds layouts like ``raw_selected/sub17/EP01_06/img81.jpg`` that
    must align with ``casme2_labels.csv`` onset/apex/offset; without this root the bridge
    misses the folder and uploads fall back to VideoPreprocessor (wrong predictions).
    """
    paths: list[Optional[Path]] = [
        project_root / "data" / "casme2",
        project_root / "data" / "raw_selected",
        project_root / "data",
    ]
    if subject_id:
        paths.insert(2, project_root / "data" / "predict" / str(subject_id).strip().lower())
    return _dedupe_existing_dirs(paths)


def first_regimg_episode_dir(roots: Sequence[Path], subject: str, episode_id: str) -> Optional[Path]:
    """First existing directory under roots/<subject>/<episode_id> that contains reg images."""
    sid = subject.strip().lower()
    ep = str(episode_id).strip()
    for root in roots:
        d = Path(root) / sid / ep
        if d.is_dir() and OnsetApexOffsetSelector.natural_image_paths(d):
            return d
    return None


def resolve_regimg_dir(casme2_root: Path, subject: str, stem: str) -> Tuple[Optional[Path], Optional[str]]:
    roots = _dedupe_existing_dirs([casme2_root])
    return resolve_regimg_dir_multi(roots, subject, stem)


def resolve_regimg_dir_multi(roots: Sequence[Path], subject: str, stem: str) -> Tuple[Optional[Path], Optional[str]]:
    if not roots:
        return None, None
    for eid in episode_id_candidates(stem):
        d = first_regimg_episode_dir(roots, subject, eid)
        if d is not None:
            return d, eid
    return None, None


def find_csv_row(df: pd.DataFrame, subject: str, episode_id: str) -> Optional[pd.Series]:
    sub = df[df["subject_id"].astype(str).str.strip().str.lower() == subject.lower()]
    if sub.empty:
        return None
    m = sub[sub["episode_id"].astype(str).str.strip() == episode_id]
    if not m.empty:
        return m.iloc[0]
    m = sub[sub["episode_id"].astype(str).str.strip().str.lower() == episode_id.lower()]
    if not m.empty:
        return m.iloc[0]
    return None


def find_labels_row_fuzzy(df: pd.DataFrame, subject_id: str, episode_hint: str) -> Optional[pd.Series]:
    """
    Match one labels row for a subject + episode (handles EPxx_yy vs EPxx_yyf).
    """
    sid = subject_id.strip().lower()
    sub = df[df["subject_id"].astype(str).str.strip().str.lower() == sid]
    if sub.empty:
        return None
    w = episode_hint.strip().lower()
    ep_col = sub["episode_id"].astype(str).str.strip().str.lower()
    m = sub[ep_col == w]
    if not m.empty:
        return m.iloc[0]
    if not w.endswith("f"):
        m = sub[ep_col == w + "f"]
        if not m.empty:
            return m.iloc[0]
    m = sub[ep_col.str.startswith(w)]
    if not m.empty:
        return m.iloc[0]
    return None


def get_clip_tensors(
    *,
    subject: str,
    filename_stem: str,
    video_path: Optional[Path],
    casme2_root: Path,
    labels_df: pd.DataFrame,
    selector: OnsetApexOffsetSelector,
    video_pre: VideoPreprocessor,
    max_video_frames: int = 64,
    extra_regimg_roots: Optional[Sequence[Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Returns (frames (3,3,64,64), flows (6,64,64), source_tag).

    CSV + reg_img: tries ``casme2_root`` then ``extra_regimg_roots`` (e.g. ``data/`` flat layout,
    ``data/predict/subXX/`` copies) so training-time paths resolve the same as Flask ``data/predict``.
    """
    roots = _dedupe_existing_dirs([casme2_root, *(extra_regimg_roots or ())])

    # 1) Prefer labels CSV + official onset/apex/offset on whichever root has reg_img.
    for eid in episode_id_candidates(filename_stem):
        row = find_csv_row(labels_df, subject, eid)
        if row is None:
            continue
        ep = str(row["episode_id"]).strip()
        reg_dir = first_regimg_episode_dir(roots, subject, ep)
        if reg_dir is None:
            continue
        sample = {
            "subject": str(row["subject_id"]),
            "episode": str(row["episode_id"]),
            "video_path": str(reg_dir),
            "onset_frame": int(row["onset_frame"]),
            "apex_frame": int(row["apex_frame"]),
            "offset_frame": int(row["offset_frame"]),
        }
        loaded = selector.load_onset_apex_offset_rgb(sample)
        if loaded is not None:
            o, a, off = loaded
            ft, fl = rgb_triplet_to_tensors(o, a, off)
            return ft, fl, "casme_csv_regimg"

    # 2) reg_img only (no CSV row): first matching folder under any root.
    reg_dir, _ = resolve_regimg_dir_multi(roots, subject, filename_stem)
    if reg_dir is not None:
        naive = load_naive_regimg_triplet(reg_dir)
        if naive is not None:
            o, a, off = naive
            ft, fl = rgb_triplet_to_tensors(o, a, off)
            return ft, fl, "regimg_naive_triplet"

    if video_path is not None and video_path.is_file():
        ft, fl = video_pre.preprocess_video(
            str(video_path), max_input_frames=max_video_frames, verbose=False
        )
        return ft, fl, "video_preprocessor"

    root_hint = ", ".join(str(r) for r in roots) if roots else str(casme2_root)
    raise FileNotFoundError(
        f"No reg_img for {subject}/{filename_stem} under [{root_hint}] and no video: {video_path}"
    )
