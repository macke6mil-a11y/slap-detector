#!/usr/bin/env python3
"""
Door Detector — Détecteur d'ouvertures/fermetures de portes
=============================================================
Pipeline 3 passes (audio-first, comme best_detector.py) :
  Passe 1 – Audio pre-filter : onset detection sur signatures sonores de porte
                               (claquement basse-fréq + grincement haute-fréq)
  Passe 2 – Motion gate      : vérifie le mouvement autour du candidat audio
  Passe 3 – CLIP confirm     : zero-shot "porte qui s'ouvre/ferme" (GPU)

Usage :
  python door_detector.py video.mkv
  python door_detector.py video.mkv --output portes.json --export-clips ./clips_portes/
  python door_detector.py video.mkv --sensitivity high
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = Console()

try:
    import torch
    import clip as openai_clip
    CLIP_AVAILABLE = True
except (ImportError, AttributeError):
    CLIP_AVAILABLE = False
    console.print("[yellow]CLIP non disponible — mode audio+motion uniquement[/yellow]" if RICH else
                  "WARNING: CLIP non disponible — mode audio+motion uniquement")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS CLIP
# ─────────────────────────────────────────────────────────────────────────────

DOOR_PROMPTS_POS = [
    "a door opening",
    "a door closing",
    "a person opening a door",
    "a person closing a door",
    "a person walking through a doorway",
    "an open door",
    "a door being pushed open",
    "a door slamming shut",
    "someone entering a room",
    "someone leaving through a door",
]

DOOR_PROMPTS_NEG = [
    "a window",
    "a wall",
    "two people fighting",
    "a person sitting still",
    "an outdoor landscape",
    "a car driving",
    "a table with objects",
    "a person eating",
]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_SR   = 22050
HOP_LENGTH = 512

@dataclass
class DoorConfig:
    # Passe 1 — Audio
    onset_delta: float          # sensibilité onset
    min_rms_db: float           # niveau minimum
    audio_score_threshold: float

    # Passe 2 — Motion
    motion_pixels: int          # pixels actifs minimum
    motion_window_s: float      # fenêtre autour du candidat

    # Passe 3 — CLIP
    clip_fps: float
    clip_window_s: float
    min_clip_score: float

    # Post-processing
    merge_gap_s: float


PRESETS = {
    "low": DoorConfig(
        onset_delta=0.35, min_rms_db=-22.0, audio_score_threshold=0.62,
        motion_pixels=800, motion_window_s=1.0,
        clip_fps=3.0, clip_window_s=1.5, min_clip_score=0.28,
        merge_gap_s=2.0,
    ),
    "medium": DoorConfig(
        onset_delta=0.25, min_rms_db=-30.0, audio_score_threshold=0.52,
        motion_pixels=400, motion_window_s=1.2,
        clip_fps=4.0, clip_window_s=2.0, min_clip_score=0.22,
        merge_gap_s=2.0,
    ),
    "high": DoorConfig(
        onset_delta=0.15, min_rms_db=-38.0, audio_score_threshold=0.42,
        motion_pixels=200, motion_window_s=1.5,
        clip_fps=5.0, clip_window_s=2.5, min_clip_score=0.16,
        merge_gap_s=3.0,
    ),
}


@dataclass
class AudioCandidate:
    t: float
    score: float
    features: dict = field(default_factory=dict)


@dataclass
class DoorEvent:
    t_start: float
    t_end: float
    t_peak: float
    final_score: float
    audio_score: float
    clip_score: float
    event_type: str
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str, sr: int = AUDIO_SR) -> tuple:
    tmp = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr),
           "-vn", tmp, "-loglevel", "error"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
    y, sr_out = sf.read(tmp)
    os.unlink(tmp)
    y = y.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * 0.95
    return y, sr_out


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or 25.0


def nms_merge(candidates: list, gap_s: float) -> list:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x.t)
    merged = [candidates[0]]
    for c in candidates[1:]:
        if c.t - merged[-1].t < gap_s:
            if c.score > merged[-1].score:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def nms_events(events: list, gap_s: float) -> list:
    if not events:
        return []
    events = sorted(events, key=lambda e: e.t_peak)
    merged = [events[0]]
    for e in events[1:]:
        if e.t_peak - merged[-1].t_peak < gap_s:
            if e.final_score > merged[-1].final_score:
                merged[-1] = e
        else:
            merged.append(e)
    return merged


def export_clip(video_path: str, t_start: float, t_end: float, out_path: str):
    duration = t_end - t_start
    cmd = ["ffmpeg", "-y", "-ss", str(max(0.0, t_start - 0.5)),
           "-i", video_path, "-t", str(duration + 1.0),
           "-c", "copy", out_path, "-loglevel", "error"]
    subprocess.run(cmd, capture_output=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 1 — AUDIO PRE-FILTER
# ─────────────────────────────────────────────────────────────────────────────

def compute_door_audio_features(y: np.ndarray, sr: int, peak_idx: int, hop: int) -> dict:
    """
    Features audio spécifiques aux sons de porte :
    - Claquement  : transitoire brusque, énergie basse fréquence, crest factor élevé
    - Grincement  : durée plus longue, contenu harmonique dans 300-3000Hz
    - Loquet/clic : très court, haute fréquence, fort crest factor
    """
    n = len(y)
    peak_s    = peak_idx * hop / sr
    i_center  = min(max(int(peak_s * sr), 0), n - 1)

    ctx_s   = int(1.5 * sr)
    hit_s   = int(0.12 * sr)
    long_s  = int(0.5 * sr)   # fenêtre longue pour grincements

    ctx_start  = max(0, i_center - ctx_s)
    ctx_end    = min(n, i_center + ctx_s)
    hit_start  = max(0, i_center - hit_s)
    hit_end    = min(n, i_center + hit_s)
    long_start = max(0, i_center - long_s)
    long_end   = min(n, i_center + long_s)
    pre_start  = max(0, i_center - ctx_s)
    pre_end    = max(0, i_center - int(0.1 * sr))

    y_ctx  = y[ctx_start:ctx_end]  if ctx_end  > ctx_start  else np.zeros(1)
    y_hit  = y[hit_start:hit_end]  if hit_end  > hit_start  else np.zeros(1)
    y_long = y[long_start:long_end] if long_end > long_start else np.zeros(1)
    y_pre  = y[pre_start:pre_end]  if pre_end  > pre_start  else np.zeros(1)

    rms_ctx  = np.sqrt(np.mean(y_ctx**2))  + 1e-10
    rms_hit  = np.sqrt(np.mean(y_hit**2))
    rms_long = np.sqrt(np.mean(y_long**2)) + 1e-10
    rms_pre  = np.sqrt(np.mean(y_pre**2))  + 1e-10

    # 1. Impact ratio : énergie pic / contexte
    impact_ratio = np.clip(rms_hit / rms_ctx, 0, 8) / 8

    # 2. Contrast ratio : rupture par rapport au silence précédent
    contrast_ratio = np.clip(rms_hit / rms_pre, 0, 10) / 10

    # 3. Crest factor : transitoire nette (claquement)
    crest = np.max(np.abs(y_hit)) / (rms_hit + 1e-10)
    crest_score = np.clip((crest - 1) / 12, 0, 1)

    # 4. Contenu basse fréquence (50-500Hz) — caractéristique claquement/résonance porte
    spec = np.abs(np.fft.rfft(y_long * np.hanning(len(y_long))))
    freqs = np.fft.rfftfreq(len(y_long), d=1/sr)
    total_energy = np.sum(spec) + 1e-10
    low_mask  = (freqs >= 50)  & (freqs <= 500)
    mid_mask  = (freqs >= 300) & (freqs <= 3000)
    low_ratio  = np.sum(spec[low_mask])  / total_energy
    mid_ratio  = np.sum(spec[mid_mask])  / total_energy

    # Score basse fréq : une porte résonne dans les basses
    low_score = np.clip(low_ratio * 3, 0, 1)
    # Score mid : grincement/frottement dans les médiums
    mid_score = np.clip(mid_ratio * 2, 0, 1)

    # 5. Durée de l'événement : les portes ont une durée caractéristique (0.1-2s)
    # On mesure la durée où le signal dépasse 20% du peak local
    threshold_level = np.max(np.abs(y_long)) * 0.20
    active = np.abs(y_long) > threshold_level
    duration_score = np.clip(np.sum(active) / sr / 1.5, 0, 1)

    # 6. Spectral flatness : un claquement de porte est "bruit large bande"
    spec_flat = np.mean(spec) / (np.max(spec) + 1e-10)
    flatness_score = np.clip(spec_flat * 4, 0, 1)

    rms_db = 20 * np.log10(rms_hit + 1e-10)

    return {
        "impact_ratio":   float(impact_ratio),
        "contrast_ratio": float(contrast_ratio),
        "crest_factor":   float(crest_score),
        "low_freq":       float(low_score),
        "mid_freq":       float(mid_score),
        "duration":       float(duration_score),
        "flatness":       float(flatness_score),
        "rms_db":         float(rms_db),
    }


def score_door_audio(feats: dict) -> float:
    """
    Score composite pour sons de porte.
    Différent d'une gifle : plus de basses fréquences, durée plus longue,
    moins de contenu haute fréquence pur.
    """
    # Impact/transitoire (40%)
    impact = (
        feats["impact_ratio"]   * 0.15 +
        feats["contrast_ratio"] * 0.15 +
        feats["crest_factor"]   * 0.10
    )
    # Signature fréquentielle porte (40%)
    freq = (
        feats["low_freq"]  * 0.20 +
        feats["mid_freq"]  * 0.10 +
        feats["flatness"]  * 0.10
    )
    # Durée (20%)
    dur = feats["duration"] * 0.20

    return float(np.clip(impact + freq + dur, 0, 1))


def pass1_audio(y: np.ndarray, sr: int, cfg: DoorConfig, verbose: bool = True) -> list:
    """Passe 1 : détection des candidats par analyse audio."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    peaks, _ = find_peaks(
        onset_env,
        height=np.mean(onset_env) + cfg.onset_delta * np.std(onset_env),
        distance=int(0.5 * sr / HOP_LENGTH),
        prominence=cfg.onset_delta * 0.4,
    )

    candidates = []
    for peak_idx in peaks:
        t = peak_idx * HOP_LENGTH / sr
        feats = compute_door_audio_features(y, sr, peak_idx, HOP_LENGTH)
        if feats["rms_db"] < cfg.min_rms_db:
            continue
        score = score_door_audio(feats)
        if score < cfg.audio_score_threshold:
            continue
        candidates.append(AudioCandidate(t=t, score=score, features=feats))

    candidates = nms_merge(candidates, gap_s=0.8)

    if verbose:
        console.print(f"  Passe 1 : {len(candidates)} candidats audio détectés")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 2 — MOTION GATE
# ─────────────────────────────────────────────────────────────────────────────

def pass2_motion(video_path: str, candidates: list, cfg: DoorConfig,
                 fps: float, verbose: bool = True) -> list:
    """Passe 2 : vérifie le mouvement visuel autour du candidat audio."""
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confirmed = []

    for cand in candidates:
        win_start = max(0, int((cand.t - cfg.motion_window_s) * fps))
        win_end   = min(total_frames - 1, int((cand.t + cfg.motion_window_s) * fps))
        step = max(1, int(fps / 4))

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
        f = win_start
        while f <= win_end:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            f += step
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)

        if len(frames) < 2:
            confirmed.append(cand)
            continue

        motion_scores = [np.sum(cv2.absdiff(frames[i-1], frames[i]) > 20)
                         for i in range(1, len(frames))]
        max_motion = max(motion_scores) if motion_scores else 0

        if max_motion >= cfg.motion_pixels:
            cand.features["motion_norm"] = float(np.clip(max_motion / (320 * 180 * 0.10), 0, 1))
            confirmed.append(cand)

    cap.release()

    if verbose:
        console.print(f"  Passe 2 : {len(confirmed)}/{len(candidates)} candidats après motion gate")

    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 3 — CLIP CONFIRM
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_model():
    if not CLIP_AVAILABLE:
        return None, None, "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = openai_clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def clip_score_frames(frames_bgr: list, model, preprocess, device,
                      pos_features, neg_features) -> float:
    """Score CLIP max sur une liste de frames."""
    if not frames_bgr or model is None:
        return 0.0
    from PIL import Image
    scores = []
    for frame in frames_bgr:
        img = preprocess(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            pos_sim = float((feat @ pos_features.T).max())
            neg_sim = float((feat @ neg_features.T).max())
            scores.append(pos_sim - neg_sim * 0.4)
    return float(np.max(scores)) if scores else 0.0


def infer_event_type(frames_bgr: list, model, preprocess, device) -> str:
    """Ouverture vs fermeture."""
    if model is None or len(frames_bgr) < 3:
        return "unknown"
    from PIL import Image
    open_tok  = openai_clip.tokenize(["a door opening", "an open doorway"]).to(device)
    close_tok = openai_clip.tokenize(["a door closing", "a closed door"]).to(device)
    with torch.no_grad():
        fo = model.encode_text(open_tok);  fo = fo / fo.norm(dim=-1, keepdim=True)
        fc = model.encode_text(close_tok); fc = fc / fc.norm(dim=-1, keepdim=True)
    open_s, close_s = [], []
    for frame in frames_bgr:
        img = preprocess(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            open_s.append(float((feat @ fo.T).max()))
            close_s.append(float((feat @ fc.T).max()))
    return "opening" if np.mean(open_s) >= np.mean(close_s) else "closing"


def pass3_clip(video_path: str, candidates: list, cfg: DoorConfig, fps: float,
               model, preprocess, device, pos_features, neg_features,
               verbose: bool = True) -> list:
    """Passe 3 : confirmation CLIP zero-shot."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / cfg.clip_fps))
    events = []

    for cand in candidates:
        f_start = max(0, int((cand.t - cfg.clip_window_s) * fps))
        f_end   = min(total_frames - 1, int((cand.t + cfg.clip_window_s) * fps))

        frames_bgr = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        f = f_start
        while f <= f_end:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
            f += step
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)

        if not frames_bgr:
            continue

        if model is not None:
            clip_s = clip_score_frames(frames_bgr, model, preprocess, device,
                                       pos_features, neg_features)
        else:
            # Sans CLIP : utiliser uniquement audio + motion
            clip_s = cand.score * 0.8

        if clip_s < cfg.min_clip_score:
            continue

        event_type = infer_event_type(frames_bgr, model, preprocess, device) if model else "unknown"

        # Score final : pondération audio + CLIP
        final_score = 0.40 * cand.score + 0.60 * clip_s

        events.append(DoorEvent(
            t_start=cand.t - cfg.clip_window_s,
            t_end=cand.t + cfg.clip_window_s,
            t_peak=cand.t,
            final_score=float(final_score),
            audio_score=float(cand.score),
            clip_score=float(clip_s),
            event_type=event_type,
            details={**cand.features},
        ))

    cap.release()

    if verbose:
        console.print(f"  Passe 3 : {len(events)}/{len(candidates)} portes confirmées par CLIP")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(
    video_path: str,
    sensitivity: str = "medium",
    output_json: Optional[str] = None,
    export_clips_dir: Optional[str] = None,
    verbose: bool = True,
) -> list:

    cfg = PRESETS[sensitivity]
    video_path = str(Path(video_path).resolve())

    console.rule(f"[bold cyan]Door Detector[/bold cyan] — sensibilité : [yellow]{sensitivity}[/yellow]" if RICH else
                 f"Door Detector — sensibilité : {sensitivity}")
    console.print(f"Vidéo : {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    console.print(f"  {w}×{h} @ {fps:.2f}fps")

    # Charger CLIP
    model, preprocess, device = None, None, "cpu"
    pos_features, neg_features = None, None
    if CLIP_AVAILABLE:
        console.print("  Chargement CLIP ViT-B/32...")
        model, preprocess, device = load_clip_model()
        if model is not None:
            tpos = openai_clip.tokenize(DOOR_PROMPTS_POS).to(device)
            tneg = openai_clip.tokenize(DOOR_PROMPTS_NEG).to(device)
            with torch.no_grad():
                pos_features = model.encode_text(tpos); pos_features /= pos_features.norm(dim=-1, keepdim=True)
                neg_features = model.encode_text(tneg); neg_features /= neg_features.norm(dim=-1, keepdim=True)
            console.print(f"  CLIP chargé ({device})")

    # ── Passe 1 : Audio ──────────────────────────────────────────────────────
    console.print("\n[bold]Passe 1[/bold] — Analyse audio..." if RICH else "\nPasse 1 — Analyse audio...")
    y, sr = extract_audio(video_path, AUDIO_SR)
    duration_s = len(y) / sr
    console.print(f"  Durée : {duration_s/60:.1f} min")
    candidates = pass1_audio(y, sr, cfg, verbose=verbose)

    if not candidates:
        console.print("  Aucun candidat audio.")
        return []

    # ── Passe 2 : Motion gate ─────────────────────────────────────────────────
    console.print("\n[bold]Passe 2[/bold] — Motion gate..." if RICH else "\nPasse 2 — Motion gate...")
    candidates = pass2_motion(video_path, candidates, cfg, fps, verbose=verbose)

    if not candidates:
        console.print("  Aucun candidat après motion gate.")
        return []

    # ── Passe 3 : CLIP ────────────────────────────────────────────────────────
    console.print("\n[bold]Passe 3[/bold] — Confirmation CLIP..." if RICH else "\nPasse 3 — Confirmation CLIP...")
    events = pass3_clip(video_path, candidates, cfg, fps,
                        model, preprocess, device,
                        pos_features, neg_features, verbose=verbose)

    events = nms_events(events, gap_s=cfg.merge_gap_s)

    # ── Résultats ─────────────────────────────────────────────────────────────
    console.rule("[bold green]Résultats[/bold green]" if RICH else "Résultats")

    if not events:
        console.print("  Aucune porte détectée.")
    else:
        if RICH:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#",       width=4)
            table.add_column("Timestamp", width=10)
            table.add_column("Type",    width=10)
            table.add_column("Score",   width=8)
            table.add_column("Audio",   width=8)
            table.add_column("CLIP",    width=8)
            for i, e in enumerate(events, 1):
                t = e.t_peak
                m, s = int(t // 60), t % 60
                table.add_row(str(i), f"{m:02d}:{s:05.2f}", e.event_type,
                              f"{e.final_score:.3f}", f"{e.audio_score:.3f}",
                              f"{e.clip_score:.3f}")
            console.print(table)
        else:
            for i, e in enumerate(events, 1):
                t = e.t_peak
                m, s = int(t // 60), t % 60
                print(f"  #{i:03d}  {m:02d}:{s:05.2f}  {e.event_type:<10}  score={e.final_score:.3f}  audio={e.audio_score:.3f}  clip={e.clip_score:.3f}")

    # ── Export JSON ───────────────────────────────────────────────────────────
    results = {
        "video": video_path,
        "sensitivity": sensitivity,
        "total_duration_s": round(duration_s, 2),
        "n_events": len(events),
        "events": [
            {"t_start": round(e.t_start, 3), "t_end": round(e.t_end, 3),
             "t_peak": round(e.t_peak, 3), "type": e.event_type,
             "score": round(e.final_score, 4), "audio_score": round(e.audio_score, 4),
             "clip_score": round(e.clip_score, 4), "details": e.details}
            for e in events
        ],
    }
    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"\nRésultats écrits : {output_json}")

    # ── Export clips ──────────────────────────────────────────────────────────
    if export_clips_dir and events:
        Path(export_clips_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        console.print(f"\nExport clips → {export_clips_dir}")
        for i, e in enumerate(events, 1):
            t = e.t_peak
            m, s = int(t // 60), t % 60
            name = f"{stem}_porte_{i:03d}_{e.event_type}_{m:02d}m{s:04.1f}s.mp4"
            path = str(Path(export_clips_dir) / name)
            export_clip(video_path, e.t_start, e.t_end, path)
            console.print(f"  [{i}] {name}")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Door Detector — pipeline 3 passes audio + motion + CLIP")
    parser.add_argument("video")
    parser.add_argument("--sensitivity", "-s", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--export-clips", "-c", default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Erreur : fichier introuvable : {args.video}")
        sys.exit(1)

    run_detection(
        video_path=args.video,
        sensitivity=args.sensitivity,
        output_json=args.output,
        export_clips_dir=args.export_clips,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
