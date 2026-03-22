#!/usr/bin/env python3
"""
Best Slap Detector — Pipeline 3 passes audio-visuel
====================================================
Combine le meilleur de tous les détecteurs existants :

  Passe 1 – Audio pre-filter  : onset detection + 8 features spectrales (rapide, CPU)
  Passe 2 – Motion gate       : détection de mouvement par diff d'images (léger, CPU)
  Passe 3 – Pose geo confirm  : YOLOv8x-pose + trajectoire main→visage (précis, GPU)

Usage :
  python best_detector.py video.mp4
  python best_detector.py video.mp4 --sensitivity high
  python best_detector.py video.mp4 --sensitivity low --output results.json
  python best_detector.py video.mp4 --export-clips ./clips/
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

# ── Progress UI ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = Console()

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    console.print("[yellow]ultralytics non installé — Passe 3 désactivée[/yellow]" if RICH else
                  "WARNING: ultralytics non installé — Passe 3 désactivée")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensitivityConfig:
    # Audio
    onset_delta: float          # sensibilité détection d'onset
    min_rms_db: float           # niveau sonore minimum (dB)
    audio_score_threshold: float

    # Motion gate
    motion_pixels: int          # nombre de pixels actifs minimum
    motion_window_s: float      # fenêtre autour du candidat (secondes)

    # Pose geometry
    wrist_velocity_threshold: float     # vitesse minimale du poignet (px/frame normalisé)
    hand_face_proximity: float          # distance max main-visage (ratio hauteur)
    pose_confidence_min: float          # confiance minimale YOLO

    # Fusion
    final_threshold: float
    audio_weight: float
    pose_weight: float


PRESETS = {
    "low": SensitivityConfig(
        onset_delta=0.18, min_rms_db=-22.0, audio_score_threshold=0.50,
        motion_pixels=35, motion_window_s=0.8,
        wrist_velocity_threshold=0.28, hand_face_proximity=0.28, pose_confidence_min=0.45,
        final_threshold=0.68, audio_weight=0.45, pose_weight=0.55,
    ),
    "medium": SensitivityConfig(
        onset_delta=0.12, min_rms_db=-30.0, audio_score_threshold=0.38,
        motion_pixels=25, motion_window_s=1.0,
        wrist_velocity_threshold=0.20, hand_face_proximity=0.35, pose_confidence_min=0.35,
        final_threshold=0.55, audio_weight=0.45, pose_weight=0.55,
    ),
    "high": SensitivityConfig(
        onset_delta=0.07, min_rms_db=-38.0, audio_score_threshold=0.28,
        motion_pixels=15, motion_window_s=1.2,
        wrist_velocity_threshold=0.13, hand_face_proximity=0.42, pose_confidence_min=0.25,
        final_threshold=0.42, audio_weight=0.50, pose_weight=0.50,
    ),
}

# Paramètres fixes
AUDIO_SR        = 22050
HOP_LENGTH      = 512
MIN_GAP_S       = 0.5       # gap minimum entre deux gifles (secondes)
CLUSTER_GAP_S   = 1.5       # fusion des détections proches
FRAMES_PER_WIN  = 12        # frames analysées pour la pose (±0.6s)
YOLO_MODEL      = "yolov8x-pose.pt"   # changer en yolov8n-pose.pt pour CPU

# Keypoints MediaPipe / YOLO (indices COCO)
KP_NOSE         = 0
KP_LEFT_EYE     = 1
KP_RIGHT_EYE    = 2
KP_LEFT_EAR     = 3
KP_RIGHT_EAR    = 4
KP_LEFT_WRIST   = 9
KP_RIGHT_WRIST  = 10


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioCandidate:
    t: float            # timestamp (secondes)
    score: float        # score audio 0-1
    features: dict = field(default_factory=dict)


@dataclass
class SlapDetection:
    t_start: float
    t_end: float
    t_peak: float
    final_score: float
    audio_score: float
    pose_score: float
    motion_score: float
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str, sr: int = AUDIO_SR) -> tuple[np.ndarray, int]:
    """Extrait l'audio d'une vidéo via ffmpeg vers fichier temporaire."""
    tmp = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr),
        "-vn", tmp,
        "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr.decode()}")
    y, sr_out = sf.read(tmp)
    os.unlink(tmp)
    y = y.astype(np.float32)
    # Normalisation peak : rend le pipeline indépendant du codec (DTS, AAC, PCM...)
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
    """Non-maximum suppression : fusionne les candidats proches, garde le meilleur."""
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


def nms_detections(detections: list[SlapDetection], gap_s: float) -> list[SlapDetection]:
    """Fusionne les détections finales proches."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.t_peak)
    merged = [detections[0]]
    for d in detections[1:]:
        if d.t_peak - merged[-1].t_peak < gap_s:
            if d.final_score > merged[-1].final_score:
                merged[-1] = d
        else:
            merged.append(d)
    return merged


def export_clip(video_path: str, t_start: float, t_end: float, out_path: str):
    """Exporte un clip vidéo avec ffmpeg."""
    duration = t_end - t_start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, t_start - 0.5)),
        "-i", video_path,
        "-t", str(duration + 1.0),
        "-c", "copy",
        out_path,
        "-loglevel", "error"
    ]
    subprocess.run(cmd, capture_output=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 1 — AUDIO PRE-FILTER
# ─────────────────────────────────────────────────────────────────────────────

def compute_audio_features(y: np.ndarray, sr: int, peak_idx: int, hop: int) -> dict:
    """
    Calcule 8 features pour caractériser un son de type 'gifle' autour d'un onset peak.
    Inspiré de audio_detector.py + slap_detector.py
    """
    n = len(y)
    # Contexte : ±1 seconde autour du pic, ±0.1s pour l'impact immédiat
    ctx_samples  = int(1.0 * sr)
    hit_samples  = int(0.1 * sr)
    pre_samples  = int(0.05 * sr)

    peak_s  = peak_idx * hop / sr
    i_center = min(max(int(peak_s * sr), 0), n - 1)

    ctx_start  = max(0, i_center - ctx_samples)
    ctx_end    = min(n, i_center + ctx_samples)
    hit_start  = max(0, i_center - hit_samples)
    hit_end    = min(n, i_center + hit_samples)
    pre_start  = max(0, i_center - ctx_samples)
    pre_end    = max(0, i_center - pre_samples)

    y_ctx  = y[ctx_start:ctx_end]  if ctx_end > ctx_start  else np.zeros(1)
    y_hit  = y[hit_start:hit_end]  if hit_end > hit_start  else np.zeros(1)
    y_pre  = y[pre_start:pre_end]  if pre_end > pre_start  else np.zeros(1)

    # 1. RMS ratio : énergie pic / énergie contexte
    rms_ctx = np.sqrt(np.mean(y_ctx**2)) + 1e-10
    rms_hit = np.sqrt(np.mean(y_hit**2))
    rms_ratio = np.clip(rms_hit / rms_ctx, 0, 5) / 5

    # 2. Contrast ratio : pic vs silence précédent
    rms_pre = np.sqrt(np.mean(y_pre**2)) + 1e-10
    contrast_ratio = np.clip(rms_hit / rms_pre, 0, 8) / 8

    # 3. Attack ratio : première moitié vs seconde moitié de la fenêtre hit
    mid = len(y_hit) // 2
    rms_first = np.sqrt(np.mean(y_hit[:mid]**2)) + 1e-10
    rms_second = np.sqrt(np.mean(y_hit[mid:]**2)) + 1e-10
    attack_ratio = np.clip(rms_first / rms_second, 0, 5) / 5

    # 4. Crest factor : amplitude crête / RMS
    crest = np.max(np.abs(y_hit)) / (rms_hit + 1e-10)
    crest_score = np.clip((crest - 1) / 9, 0, 1)   # score 0 si crest=1, 1 si crest=10

    # 5. Spectral centroid : fréquence dominante
    spec = np.abs(np.fft.rfft(y_hit * np.hanning(len(y_hit))))
    freqs = np.fft.rfftfreq(len(y_hit), d=1/sr)
    centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)
    # Une gifle est riche en 1-6kHz
    centroid_score = np.clip((centroid - 500) / (5500), 0, 1)

    # 6. Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * spec) / (np.sum(spec) + 1e-10))
    bandwidth_score = np.clip(bandwidth / 4000, 0, 1)

    # 7. Spectral flatness : proche du bruit = gifle, tonal = voix/musique
    spec_flat = np.mean(spec) / (np.max(spec) + 1e-10)
    flatness_score = np.clip(spec_flat * 5, 0, 1)   # plus c'est plat, plus c'est une gifle

    # 8. Zero Crossing Rate : contenu haute fréquence
    zcr = np.mean(np.abs(np.diff(np.sign(y_hit)))) / 2
    zcr_score = np.clip(zcr / 0.3, 0, 1)

    # RMS en dB
    rms_db = 20 * np.log10(rms_hit + 1e-10)

    return {
        "rms_ratio":      float(rms_ratio),
        "contrast_ratio": float(contrast_ratio),
        "attack_ratio":   float(attack_ratio),
        "crest_factor":   float(crest_score),
        "centroid":       float(centroid_score),
        "bandwidth":      float(bandwidth_score),
        "flatness":       float(flatness_score),
        "zcr":            float(zcr_score),
        "rms_db":         float(rms_db),
    }


def score_audio_features(feats: dict) -> float:
    """
    Score composite pondéré pour identifier une gifle dans les features audio.
    Poids calibrés empiriquement sur les approches existantes.
    """
    # Impact soudain (45%)
    impact = (
        feats["rms_ratio"]    * 0.15 +
        feats["contrast_ratio"] * 0.15 +
        feats["crest_factor"] * 0.15
    )
    # Caractère transitoire (20%)
    transitoire = feats["attack_ratio"] * 0.20

    # Spectre (35%) : centroid + flatness + zcr caractérisent le "clap"
    spectre = (
        feats["centroid"]   * 0.10 +
        feats["bandwidth"]  * 0.05 +
        feats["flatness"]   * 0.12 +
        feats["zcr"]        * 0.08
    )

    return float(np.clip(impact + transitoire + spectre, 0, 1))


def pass1_audio(y: np.ndarray, sr: int, cfg: SensitivityConfig, verbose: bool = False
                ) -> list[AudioCandidate]:
    """
    Passe 1 : détection des candidats par analyse audio.
    Retourne une liste d'AudioCandidate triée par timestamp.
    """
    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    # Trouver les pics avec prominence adaptée
    peaks, props = find_peaks(
        onset_env,
        height=np.mean(onset_env) + cfg.onset_delta * np.std(onset_env),
        distance=int(MIN_GAP_S * sr / HOP_LENGTH),
        prominence=cfg.onset_delta * 0.5,
    )

    candidates = []
    for peak_idx in peaks:
        t = peak_idx * HOP_LENGTH / sr

        feats = compute_audio_features(y, sr, peak_idx, HOP_LENGTH)

        # Filtre RMS minimum
        if feats["rms_db"] < cfg.min_rms_db:
            continue

        score = score_audio_features(feats)
        if score < cfg.audio_score_threshold:
            continue

        candidates.append(AudioCandidate(t=t, score=score, features=feats))

    # NMS
    candidates = nms_merge(candidates, gap_s=MIN_GAP_S)

    if verbose:
        console.print(f"  Passe 1 : {len(candidates)} candidats audio détectés")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 2 — MOTION GATE
# ─────────────────────────────────────────────────────────────────────────────

def pass2_motion(video_path: str, candidates: list[AudioCandidate],
                 cfg: SensitivityConfig, fps: float, verbose: bool = False
                 ) -> list[AudioCandidate]:
    """
    Passe 2 : vérifie qu'il y a du mouvement dans la fenêtre autour du candidat.
    Élimine ~70% des faux positifs audio (voix fortes, chocs, etc.)
    """
    if not candidates:
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    confirmed = []

    for cand in candidates:
        win_start = max(0, int((cand.t - cfg.motion_window_s) * fps))
        win_end   = min(total_frames - 1, int((cand.t + cfg.motion_window_s) * fps))

        # Extraction des frames à 4fps
        step = max(1, int(fps / 4))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)
        f = win_start
        while f <= win_end:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            frames.append(gray)
            f += step
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)

        if len(frames) < 2:
            # Pas assez de frames → on garde le candidat (bénéfice du doute)
            confirmed.append(cand)
            continue

        # Calcul du mouvement par diff absolue
        motion_scores = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            motion_scores.append(np.sum(diff > 20))

        max_motion = max(motion_scores) if motion_scores else 0
        motion_norm = np.clip(max_motion / (320 * 180 * 0.15), 0, 1)   # normalisé 0-1

        if max_motion >= cfg.motion_pixels:
            # Mettre à jour le score audio avec le bonus mouvement
            cand.features["motion_norm"] = float(motion_norm)
            confirmed.append(cand)

    cap.release()

    if verbose:
        console.print(f"  Passe 2 : {len(confirmed)}/{len(candidates)} candidats après motion gate")

    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# PASSE 3 — POSE GEOMETRY CONFIRM
# ─────────────────────────────────────────────────────────────────────────────

def get_face_center(kps: np.ndarray) -> Optional[np.ndarray]:
    """Centre du visage à partir des keypoints COCO (nez + yeux + oreilles)."""
    face_kps = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
    valid = []
    for ki in face_kps:
        if ki < len(kps) and kps[ki][2] > 0.2:   # confidence > 20%
            valid.append(kps[ki][:2])
    if not valid:
        return None
    return np.mean(valid, axis=0)


def analyze_pose_window(
    frames_kps: list[Optional[np.ndarray]],
    frame_h: int,
    frame_w: int,
    cfg: SensitivityConfig
) -> dict:
    """
    Analyse la trajectoire des poignets relativement au visage.
    Retourne un dict avec les scores géométriques détaillés.

    Inspiré de slap_vision.py Passe 3 (geo confirm).
    """
    n = len(frames_kps)
    if n < 3:
        return {"pose_score": 0.0}

    best_score = 0.0
    best_details = {}

    for wrist_kp in [KP_LEFT_WRIST, KP_RIGHT_WRIST]:
        positions = []
        face_centers = []

        for kps in frames_kps:
            if kps is None:
                positions.append(None)
                face_centers.append(None)
                continue
            face = get_face_center(kps)
            wrist_conf = kps[wrist_kp][2] if wrist_kp < len(kps) else 0
            if face is None or wrist_conf < cfg.pose_confidence_min:
                positions.append(None)
                face_centers.append(None)
                continue
            wrist_xy = kps[wrist_kp][:2]
            positions.append(wrist_xy)
            face_centers.append(face)

        valid = [(p, f) for p, f in zip(positions, face_centers) if p is not None and f is not None]
        if len(valid) < 3:
            continue

        wrists = np.array([v[0] for v in valid])
        faces  = np.array([v[1] for v in valid])

        # Distances main→visage normalisées par hauteur frame
        dists = np.linalg.norm(wrists - faces, axis=1) / frame_h

        # 1. Score proximité : distance minimale atteinte
        min_dist = np.min(dists)
        proximity_score = np.clip(1.0 - (min_dist / cfg.hand_face_proximity), 0, 1)

        # 2. Score approche : la main se rapproche avant le peak
        mid_idx = len(dists) // 2
        approach_score = 0.0
        if mid_idx > 0:
            approach = dists[0] - dists[mid_idx]   # réduction de distance
            approach_score = np.clip(approach / (dists[0] + 1e-6), 0, 1)

        # 3. Score retrait : la main s'éloigne après l'impact
        retreat_score = 0.0
        if mid_idx < len(dists) - 1:
            retreat = dists[-1] - dists[mid_idx]
            retreat_score = np.clip(retreat / (dists[mid_idx] + 1e-6), 0, 1)

        # 4. Vitesse : dérivée de la position du poignet
        velocities = np.linalg.norm(np.diff(wrists, axis=0), axis=1) / frame_h
        peak_velocity = float(np.max(velocities)) if len(velocities) else 0.0
        velocity_score = np.clip(peak_velocity / (cfg.wrist_velocity_threshold * 2), 0, 1)

        # 5. Décélération : la main ralentit brutalement à l'impact
        decel_score = 0.0
        if len(velocities) >= 3:
            peak_v_idx = int(np.argmax(velocities))
            if peak_v_idx > 0 and peak_v_idx < len(velocities) - 1:
                v_before = velocities[peak_v_idx]
                v_after  = velocities[peak_v_idx + 1] if peak_v_idx + 1 < len(velocities) else 0
                decel_score = np.clip((v_before - v_after) / (v_before + 1e-6), 0, 1)

        # 6. Direction latérale (gifle horizontale)
        if len(wrists) >= 2:
            deltas = np.diff(wrists, axis=0)
            lateral_ratios = np.abs(deltas[:, 0]) / (np.linalg.norm(deltas, axis=1) + 1e-6)
            lateral_score = float(np.mean(lateral_ratios))
        else:
            lateral_score = 0.0

        # Filtre minimum : vitesse requise + proximité souple (×1.5 pour plans en angle)
        if peak_velocity < cfg.wrist_velocity_threshold or min_dist > cfg.hand_face_proximity * 1.5:
            continue

        # Score composite pose
        pose_score = (
            proximity_score  * 0.30 +
            approach_score   * 0.20 +
            retreat_score    * 0.15 +
            velocity_score   * 0.20 +
            decel_score      * 0.10 +
            lateral_score    * 0.05
        )

        # Bonus vitesse élevée
        if peak_velocity > cfg.wrist_velocity_threshold * 2:
            pose_score = min(1.0, pose_score * 1.2)

        if pose_score > best_score:
            best_score = pose_score
            best_details = {
                "proximity": float(proximity_score),
                "approach":  float(approach_score),
                "retreat":   float(retreat_score),
                "velocity":  float(velocity_score),
                "decel":     float(decel_score),
                "lateral":   float(lateral_score),
                "min_dist":  float(min_dist),
                "peak_v":    float(peak_velocity),
                "wrist":     "left" if wrist_kp == KP_LEFT_WRIST else "right",
            }

    return {"pose_score": float(best_score), **best_details}


def pass3_pose(video_path: str, candidates: list[AudioCandidate],
               cfg: SensitivityConfig, fps: float,
               yolo_model, verbose: bool = False) -> list[SlapDetection]:
    """
    Passe 3 : confirmation géométrique par YOLOv8-pose.
    Pour chaque candidat, analyse ±0.6s de frames.
    """
    if not candidates or yolo_model is None:
        # Sans YOLO, on retourne les candidats audio comme détections directes
        detections = []
        for c in candidates:
            detections.append(SlapDetection(
                t_start=c.t - 0.5,
                t_end=c.t + 0.5,
                t_peak=c.t,
                final_score=c.score,
                audio_score=c.score,
                pose_score=0.0,
                motion_score=c.features.get("motion_norm", 0.0),
                details=c.features,
            ))
        return detections

    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    win_s = 1.0    # ±1.0 seconde (élargi pour capturer l'approche avant impact)
    target_fps = 12
    step = max(1, int(fps / target_fps))

    # Seuil de confiance poignet : en dessous → fallback audio-only
    WRIST_CONF_MIN = 0.15
    # Seuil audio-only fallback : score audio fort sans confirmation visuelle possible
    AUDIO_FALLBACK_THRESHOLD = 0.60

    detections = []

    for cand in candidates:
        f_start = max(0, int((cand.t - win_s) * fps))
        f_end   = min(total_frames - 1, int((cand.t + win_s) * fps))

        # Extraire les frames
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

        # Détecter les poses sur chaque frame
        frames_kps = []
        max_wrist_conf = 0.0
        for frame in frames_bgr:
            results = yolo_model(frame, verbose=False)
            best_kps = None
            best_conf = 0.0
            for r in results:
                if r.keypoints is None or len(r.keypoints.data) == 0:
                    continue
                for kps_tensor in r.keypoints.data:
                    kps = kps_tensor.cpu().numpy()   # (17, 3) : x, y, conf
                    # Choisir la personne la plus confiante
                    mean_conf = float(np.mean(kps[:, 2]))
                    if mean_conf > best_conf:
                        best_conf = mean_conf
                        best_kps = kps
                        wrist_conf = max(float(kps[KP_LEFT_WRIST][2]), float(kps[KP_RIGHT_WRIST][2]))
                        max_wrist_conf = max(max_wrist_conf, wrist_conf)
            frames_kps.append(best_kps)

        motion_score = cand.features.get("motion_norm", 0.5)

        # Fallback audio-only : poignets hors-champ mais audio fort
        if max_wrist_conf < WRIST_CONF_MIN and cand.score >= AUDIO_FALLBACK_THRESHOLD:
            final_score = cand.score * 0.85   # légère pénalité sans confirmation visuelle
            # Seuil réduit pour fallbacks : pas de pose disponible
            if final_score >= cfg.final_threshold * 0.80:
                detections.append(SlapDetection(
                    t_start=cand.t - 0.5,
                    t_end=cand.t + 0.5,
                    t_peak=cand.t,
                    final_score=float(final_score),
                    audio_score=float(cand.score),
                    pose_score=0.0,
                    motion_score=float(motion_score),
                    details={**cand.features, "pose_score": 0.0, "fallback": "audio_only",
                             "wrists_offscreen": True, "max_wrist_conf": float(max_wrist_conf)},
                ))
            continue

        # Analyse géométrique normale
        geo = analyze_pose_window(frames_kps, frame_h, frame_w, cfg)
        pose_score = geo.get("pose_score", 0.0)

        # Fallback audio-fort : pose partielle mais signal audio convaincant
        # (score < 0.30 = angle difficile, wrist peu mobile, gifle courte)
        if pose_score < 0.30 and cand.score >= AUDIO_FALLBACK_THRESHOLD:
            final_score = cand.score * 0.82
            # Seuil réduit pour fallbacks
            if final_score >= cfg.final_threshold * 0.80:
                detections.append(SlapDetection(
                    t_start=cand.t - 0.5,
                    t_end=cand.t + 0.5,
                    t_peak=cand.t,
                    final_score=float(final_score),
                    audio_score=float(cand.score),
                    pose_score=0.0,
                    motion_score=float(motion_score),
                    details={**cand.features, "pose_score": 0.0, "fallback": "audio_strong",
                             "max_wrist_conf": float(max_wrist_conf)},
                ))
            continue

        # Fusion audio + pose
        final_score = (
            cfg.audio_weight * cand.score +
            cfg.pose_weight  * pose_score
        )

        if final_score >= cfg.final_threshold:
            detections.append(SlapDetection(
                t_start=cand.t - 0.5,
                t_end=cand.t + 0.5,
                t_peak=cand.t,
                final_score=float(final_score),
                audio_score=float(cand.score),
                pose_score=float(pose_score),
                motion_score=float(motion_score),
                details={**cand.features, **geo},
            ))

    cap.release()

    if verbose:
        console.print(f"  Passe 3 : {len(detections)}/{len(candidates)} gifles confirmées par pose")

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(
    video_path: str,
    sensitivity: str = "medium",
    output_json: Optional[str] = None,
    export_clips_dir: Optional[str] = None,
    verbose: bool = True,
    yolo_model_path: str = YOLO_MODEL,
) -> list[SlapDetection]:

    cfg = PRESETS[sensitivity]
    video_path = str(Path(video_path).resolve())

    console.rule(f"[bold cyan]Best Slap Detector[/bold cyan] — sensibilité : [yellow]{sensitivity}[/yellow]" if RICH else
                 f"Best Slap Detector — sensibilité : {sensitivity}")
    console.print(f"Vidéo : {video_path}")

    # Charger YOLO
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO(yolo_model_path)
            console.print(f"  YOLO chargé : {yolo_model_path}")
        except Exception as e:
            console.print(f"  [yellow]YOLO non disponible ({e}) — Passe 3 audio-only[/yellow]" if RICH else
                          f"  WARN: YOLO non disponible ({e}) — Passe 3 audio-only")

    # ── Passe 1 : Audio ──────────────────────────────────────────────────────
    console.print("\n[bold]Passe 1[/bold] — Analyse audio..." if RICH else "\nPasse 1 — Analyse audio...")
    y, sr = extract_audio(video_path, AUDIO_SR)
    duration_s = len(y) / sr
    console.print(f"  Durée : {duration_s:.1f}s")

    audio_candidates = pass1_audio(y, sr, cfg, verbose=verbose)

    if not audio_candidates:
        console.print("  [dim]Aucun candidat audio — fin de l'analyse[/dim]" if RICH else
                      "  Aucun candidat audio — fin de l'analyse")
        return []

    console.print(f"  {len(audio_candidates)} candidats : " +
                  ", ".join(f"{c.t:.2f}s (score={c.score:.2f})" for c in audio_candidates))

    # ── Passe 2 : Motion Gate ─────────────────────────────────────────────────
    console.print("\n[bold]Passe 2[/bold] — Motion gate..." if RICH else "\nPasse 2 — Motion gate...")
    fps = get_video_fps(video_path)
    confirmed = pass2_motion(video_path, audio_candidates, cfg, fps, verbose=verbose)

    if not confirmed:
        console.print("  Aucun candidat après motion gate")
        return []

    # ── Passe 3 : Pose Confirm ────────────────────────────────────────────────
    console.print("\n[bold]Passe 3[/bold] — Confirmation par pose..." if RICH else "\nPasse 3 — Confirmation par pose...")
    detections = pass3_pose(video_path, confirmed, cfg, fps, yolo_model, verbose=verbose)

    # NMS final
    detections = nms_detections(detections, gap_s=CLUSTER_GAP_S)

    # ── Résultats ─────────────────────────────────────────────────────────────
    console.rule("Résultats" if not RICH else "[bold green]Résultats[/bold green]")

    if not detections:
        console.print("  Aucune gifle détectée.")
    else:
        if RICH:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", width=3)
            table.add_column("Timestamp", width=10)
            table.add_column("Score final", width=12)
            table.add_column("Audio", width=8)
            table.add_column("Pose", width=8)
            table.add_column("Main", width=8)
            for i, d in enumerate(detections, 1):
                wrist = d.details.get("wrist", "?")
                table.add_row(
                    str(i),
                    f"{d.t_peak:.2f}s",
                    f"{d.final_score:.3f}",
                    f"{d.audio_score:.3f}",
                    f"{d.pose_score:.3f}",
                    wrist,
                )
            console.print(table)
        else:
            for i, d in enumerate(detections, 1):
                print(f"  #{i} t={d.t_peak:.2f}s  score={d.final_score:.3f} "
                      f"(audio={d.audio_score:.3f}, pose={d.pose_score:.3f})")

    # ── Export JSON ───────────────────────────────────────────────────────────
    results = {
        "video": video_path,
        "sensitivity": sensitivity,
        "total_duration_s": round(duration_s, 2),
        "n_detections": len(detections),
        "detections": [
            {
                "t_start": round(d.t_start, 3),
                "t_end":   round(d.t_end, 3),
                "t_peak":  round(d.t_peak, 3),
                "score":   round(d.final_score, 4),
                "audio_score": round(d.audio_score, 4),
                "pose_score":  round(d.pose_score, 4),
                "details": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in d.details.items()},
            }
            for d in detections
        ],
    }

    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"\nRésultats écrits : {output_json}")

    # ── Export clips ──────────────────────────────────────────────────────────
    if export_clips_dir and detections:
        Path(export_clips_dir).mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem
        console.print(f"\nExport clips → {export_clips_dir}")
        for i, d in enumerate(detections, 1):
            clip_name = f"{video_stem}_gifle_{i:02d}_{d.t_peak:.1f}s.mp4"
            clip_path = str(Path(export_clips_dir) / clip_name)
            export_clip(video_path, d.t_start, d.t_end, clip_path)
            console.print(f"  [{i}] {clip_name}")

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Best Slap Detector — pipeline 3 passes audio + motion + pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python best_detector.py video.mp4
  python best_detector.py video.mp4 --sensitivity high
  python best_detector.py video.mp4 --output results.json --export-clips ./clips/
  python best_detector.py video.mp4 --yolo yolov8n-pose.pt   # version légère CPU
        """,
    )
    parser.add_argument("video", help="Chemin vers la vidéo à analyser")
    parser.add_argument(
        "--sensitivity", "-s",
        choices=["low", "medium", "high"],
        default="medium",
        help="Sensibilité de détection (défaut: medium)",
    )
    parser.add_argument("--output", "-o", default=None, help="Fichier JSON de sortie")
    parser.add_argument("--export-clips", "-c", default=None, help="Dossier pour exporter les clips")
    parser.add_argument("--yolo", default=YOLO_MODEL, help=f"Modèle YOLO-pose (défaut: {YOLO_MODEL})")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Erreur : fichier introuvable : {args.video}")
        sys.exit(1)

    detections = run_detection(
        video_path=args.video,
        sensitivity=args.sensitivity,
        output_json=args.output,
        export_clips_dir=args.export_clips,
        verbose=not args.quiet,
        yolo_model_path=args.yolo,
    )

    sys.exit(0 if detections is not None else 1)


if __name__ == "__main__":
    main()
