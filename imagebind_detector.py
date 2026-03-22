#!/usr/bin/env python3
"""
imagebind_detector.py — Détecteur de gifles unifié audio + visuel via ImageBind
Architecture :
  Passe 0 — Embeddings texte (prompts slap positifs / négatifs)
  Passe 1 — Fenêtres audio -> ImageBind audio embeddings -> score audio
  Passe 2 — Top K candidats audio -> frames vidéo -> ImageBind visual embeddings -> score visuel
  Passe 3 — Fusion, NMS, export JSON + clips

Usage :
  python imagebind_detector.py VIDEO.mkv --output gifles.json --export-clips ./clips/
  python imagebind_detector.py  (utilise Pieta par défaut)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import cv2
from PIL import Image
from torchvision import transforms as tv_transforms

# ── ImageBind ─────────────────────────────────────────────────────────────────
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as ib_data

# ── Chemin par défaut ─────────────────────────────────────────────────────────
DEFAULT_VIDEO = (
    r"F:\[kr]\copie_de_[kr]_D\kim_ki_duk"
    r"\Pieta.2012.BluRay.1080p.DTS.x264-CHD"
    r"\Pieta.2012.BluRay.1080p.DTS.x264-CHD.mkv"
)

# ── Hyperparamètres ────────────────────────────────────────────────────────────
TARGET_SR       = 16_000
MEL_BINS        = 128
MEL_FRAMES      = 204        # 2 s à 10 ms hop
AUDIO_WINDOW_S  = 2.0        # durée de chaque fenêtre audio
AUDIO_HOP_S     = 1.0        # pas entre fenêtres (overlap 50 %) — x2 plus rapide
AUDIO_BATCH     = 64         # fenêtres par batch GPU
VISUAL_BATCH    = 32         # frames par batch GPU
TOP_AUDIO_K     = 300        # candidats audio envoyés à la passe visuelle
NMS_GAP_S       = 5.0        # gap minimum entre deux détections (NMS)
FRAMES_PER_TS   = 3          # frames extraites autour de chaque candidat
FRAME_OFFSETS_S = [-0.4, 0.0, 0.4]   # décalages en secondes
CLIP_BEFORE_S   = 3.0
CLIP_AFTER_S    = 2.0

# ── Prompts texte ─────────────────────────────────────────────────────────────
SLAP_POS_PROMPTS = [
    "a person slapping another person in the face",
    "someone getting slapped hard across the face",
    "a violent face slap between two people",
    "hand hitting someone's face in anger",
    "a person being struck in the face by another person",
    "a sharp slap sound of skin hitting skin",
]

SLAP_NEG_PROMPTS = [
    "a door slamming shut loudly",
    "people clapping their hands together",
    "someone walking down a hallway",
    "a quiet conversation between two people",
    "a person sitting still doing nothing",
    "an object falling on the floor",
]

NEG_WEIGHT = 0.4   # pondération des prompts négatifs


# ── Utilitaires ───────────────────────────────────────────────────────────────
def fmt_tc(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


@dataclass
class Detection:
    time_s: float
    audio_score: float
    visual_score: float
    final_score: float

    @property
    def timecode(self) -> str:
        return fmt_tc(self.time_s)

    def to_dict(self) -> dict:
        return {
            "time_s": round(self.time_s, 2),
            "timecode": self.timecode,
            "audio_score": round(self.audio_score, 3),
            "visual_score": round(self.visual_score, 3),
            "final_score": round(self.final_score, 3),
        }


# ── Chargement modèle ─────────────────────────────────────────────────────────
def load_model(device: str) -> imagebind_model.ImageBindModel:
    print("  Chargement ImageBind huge...", end=" ", flush=True)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)
    print(f"OK ({device})")
    return model


# ── Embeddings texte ──────────────────────────────────────────────────────────
def get_text_embeddings(
    model, prompts: List[str], device: str
) -> torch.Tensor:
    """Retourne les embeddings normalisés pour une liste de prompts texte."""
    text_data = ib_data.load_and_transform_text(prompts, device)
    with torch.no_grad():
        emb = model({ModalityType.TEXT: text_data})[ModalityType.TEXT]
    return F.normalize(emb, dim=-1)


# ── Extraction audio ──────────────────────────────────────────────────────────
def extract_audio_mono(video_path: str) -> torch.Tensor:
    """Extrait l'audio mono 16 kHz du film via ffmpeg."""
    print("  Extraction audio (ffmpeg)...", end=" ", flush=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ac", "1", "-ar", str(TARGET_SR),
        "-vn", "-y", tmp.name, "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    waveform, sr = torchaudio.load(tmp.name)
    os.unlink(tmp.name)
    assert sr == TARGET_SR
    print(f"OK ({waveform.shape[1]/TARGET_SR/60:.1f} min)")
    return waveform[0]   # (T,)


# ── Mel spectrogram complet (1 appel pour tout le film) ──────────────────────
CHUNK_S = 600   # 10 minutes par chunk

def full_melspec(waveform: torch.Tensor) -> torch.Tensor:
    """
    Calcule le mel spectrogram Kaldi en chunks de CHUNK_S secondes.
    Entrée  : (T,) float mono 16kHz
    Sortie  : (total_frames, MEL_BINS)
    """
    chunk_samp = CHUNK_S * TARGET_SR
    overlap    = 1600   # 0.1s d'overlap pour ne pas perdre de frames aux bords
    chunks = []
    start = 0
    n = waveform.shape[0]
    while start < n:
        end = min(start + chunk_samp + overlap, n)
        seg = waveform[start:end]
        w = seg.unsqueeze(0)
        w = w - w.mean()
        fb = torchaudio.compliance.kaldi.fbank(
            w,
            htk_compat=True,
            sample_frequency=TARGET_SR,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=MEL_BINS,
            dither=0.0,
            frame_shift=10,
        )
        # On enlève les frames de l'overlap sauf pour le dernier chunk
        if start > 0:
            fb = fb[int(overlap / TARGET_SR * 100):]   # ~10 frames
        chunks.append(fb)
        start += chunk_samp
    fbank = torch.cat(chunks, dim=0)
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank   # (total_frames, MEL_BINS)


# ── Passe 1 : embeddings audio par fenêtres (slicing vectorisé) ───────────────
def compute_audio_embeddings(
    model,
    waveform: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Calcule le mel spectrogram complet en 1 appel, puis découpe en fenêtres
    et encode par batches avec ImageBind.
    Retourne (embeddings (N, D), timestamps (N,)).
    """
    HOP_FRAMES    = int(AUDIO_HOP_S * 100)   # 0.5s × 100 frames/s = 50 frames
    # 10ms frame shift -> 100 frames/s

    print("  Calcul mel spectrogram complet...", end=" ", flush=True)
    fbank = full_melspec(waveform)   # (total_frames, MEL_BINS)
    total_mel = fbank.shape[0]
    print(f"OK ({total_mel} frames)")

    # Découpe en fenêtres de MEL_FRAMES avec hop HOP_FRAMES
    starts = list(range(0, total_mel - MEL_FRAMES, HOP_FRAMES))
    timestamps = [
        (s / 100.0) + AUDIO_WINDOW_S / 2   # centre de la fenêtre en secondes
        for s in starts
    ]
    total = len(starts)
    print(f"  {total} fenetres audio a encoder...")

    all_embs: List[torch.Tensor] = []
    for i in range(0, total, AUDIO_BATCH):
        batch_starts = starts[i : i + AUDIO_BATCH]
        # Construire le batch : (B, 1, MEL_BINS, MEL_FRAMES)
        batch = torch.stack([
            fbank[s : s + MEL_FRAMES].T.unsqueeze(0)   # (1, MEL_BINS, MEL_FRAMES)
            for s in batch_starts
        ]).to(device)
        with torch.no_grad():
            emb = model({ModalityType.AUDIO: batch})[ModalityType.AUDIO]
        all_embs.append(F.normalize(emb, dim=-1).cpu())
        done = min(i + AUDIO_BATCH, total)
        if done % (AUDIO_BATCH * 10) == 0 or done == total:
            print(f"    {done}/{total}...", end="\r", flush=True)
    print()

    return torch.cat(all_embs, dim=0), timestamps   # (N, D), list[float]


# ── Passe 2 : embeddings visuels ──────────────────────────────────────────────
_VISUAL_PREPROCESS = tv_transforms.Compose([
    tv_transforms.Resize(224, interpolation=tv_transforms.InterpolationMode.BICUBIC),
    tv_transforms.CenterCrop(224),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])


def compute_visual_embeddings(
    model,
    video_path: str,
    timestamps: List[float],
    device: str,
) -> torch.Tensor:
    """
    Pour chaque timestamp, extrait FRAMES_PER_TS frames autour du timestamp,
    encode via ImageBind, et retourne la moyenne par timestamp.
    Retourne (N_timestamps, D).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_tensors: List[torch.Tensor] = []
    ts_indices:  List[int]          = []

    for ts_idx, t in enumerate(timestamps):
        for offset_s in FRAME_OFFSETS_S:
            t_frame = max(0.0, t + offset_s)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_frame * fps))
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                all_tensors.append(_VISUAL_PREPROCESS(img))
                ts_indices.append(ts_idx)

    cap.release()

    total_frames = len(all_tensors)
    print(f"  {total_frames} frames à encoder...")

    flat_embs: List[torch.Tensor] = []
    for i in range(0, total_frames, VISUAL_BATCH):
        batch = torch.stack(all_tensors[i : i + VISUAL_BATCH]).to(device)
        with torch.no_grad():
            emb = model({ModalityType.VISION: batch})[ModalityType.VISION]
        flat_embs.append(F.normalize(emb, dim=-1).cpu())
        done = min(i + VISUAL_BATCH, total_frames)
        print(f"    {done}/{total_frames}...", end="\r", flush=True)
    print()

    flat_embs_t = torch.cat(flat_embs, dim=0)   # (total_frames, D)

    # Moyenne par timestamp
    D = flat_embs_t.shape[-1]
    ts_embs  = torch.zeros(len(timestamps), D)
    ts_count = torch.zeros(len(timestamps))
    for i, ti in enumerate(ts_indices):
        ts_embs[ti]  += flat_embs_t[i]
        ts_count[ti] += 1
    ts_count = ts_count.clamp(min=1).unsqueeze(-1)
    return F.normalize(ts_embs / ts_count, dim=-1)   # (N, D)


# ── NMS ───────────────────────────────────────────────────────────────────────
def nms(detections: List[Detection], gap_s: float = NMS_GAP_S) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.final_score, reverse=True)
    kept: List[Detection] = []
    for d in detections:
        if all(abs(d.time_s - k.time_s) >= gap_s for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.time_s)


# ── Export clips ──────────────────────────────────────────────────────────────
def export_clip(video_path: str, det: Detection, out_dir: str, idx: int):
    os.makedirs(out_dir, exist_ok=True)
    t_start = max(0.0, det.time_s - CLIP_BEFORE_S)
    duration = CLIP_BEFORE_S + CLIP_AFTER_S
    tc = det.timecode.replace(":", "-")
    out_path = os.path.join(out_dir, f"slap_{idx:03d}_{tc}_f{det.final_score:.2f}.mp4")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ss", str(t_start), "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac", "-crf", "23",
        "-y", out_path, "-loglevel", "error",
    ])


# ── Pipeline principal ────────────────────────────────────────────────────────
def run_detection(
    video_path: str,
    output_json: str,
    export_clips_dir: Optional[str] = None,
    top_audio_k: int = TOP_AUDIO_K,
    audio_weight: float = 0.5,
    visual_weight: float = 0.5,
    min_final_score: float = 0.80,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sep = "-" * 70
    print(sep)
    print("  ImageBind Slap Detector")
    print(f"  Vidéo  : {video_path}")
    print(f"  Device : {device}  |  Top-K audio : {top_audio_k}")
    print(sep)

    # ── Chargement modèle ──────────────────────────────────────────────────────
    model = load_model(device)

    # ── Passe 0 : embeddings texte ─────────────────────────────────────────────
    print("\nPasse 0 — Embeddings texte...")
    pos_emb = get_text_embeddings(model, SLAP_POS_PROMPTS,  device)   # (P, D)
    neg_emb = get_text_embeddings(model, SLAP_NEG_PROMPTS,  device)   # (N, D)
    print(f"  {pos_emb.shape[0]} prompts positifs, {neg_emb.shape[0]} prompts négatifs")

    # ── Passe 1 : audio (avec cache .pt) ──────────────────────────────────────
    _cache = output_json.replace(".json", "_audio_cache.pt")
    print("\nPasse 1 — Analyse audio...")
    if os.path.exists(_cache):
        print(f"  Cache trouve: {_cache}")
        _c = torch.load(_cache, weights_only=True)
        audio_embs, timestamps, duration_s = _c["embs"], _c["timestamps"], _c["duration_s"]
        print(f"  {len(timestamps)} embeddings audio charges depuis le cache")
    else:
        waveform     = extract_audio_mono(video_path)
        duration_s   = waveform.shape[0] / TARGET_SR
        audio_embs, timestamps = compute_audio_embeddings(model, waveform, device)
        torch.save({"embs": audio_embs, "timestamps": timestamps, "duration_s": duration_s}, _cache)
        print(f"  Cache sauvegarde: {_cache}")

    # Score audio = sim_positive − NEG_WEIGHT × sim_negative
    audio_embs_gpu = audio_embs.to(device)
    pos_sim = (audio_embs_gpu @ pos_emb.T).mean(dim=-1)
    neg_sim = (audio_embs_gpu @ neg_emb.T).mean(dim=-1)
    raw_audio = (pos_sim - NEG_WEIGHT * neg_sim).cpu()

    # Normalisation min-max -> [0, 1]
    a_min, a_max = raw_audio.min(), raw_audio.max()
    audio_scores = (raw_audio - a_min) / (a_max - a_min + 1e-8)

    # Top K candidats
    k = min(top_audio_k, len(timestamps))
    top_idx   = torch.topk(audio_scores, k).indices.tolist()
    top_ts    = [timestamps[i] for i in top_idx]
    top_ascore = audio_scores[top_idx].tolist()
    print(f"  -> {k} candidats audio sélectionnés (score min : {min(top_ascore):.3f})")

    # ── Passe 2 : visuel ───────────────────────────────────────────────────────
    print("\nPasse 2 — Confirmation visuelle...")
    vis_embs     = compute_visual_embeddings(model, video_path, top_ts, device)
    vis_embs_gpu = vis_embs.to(device)

    pos_vis = (vis_embs_gpu @ pos_emb.T).mean(dim=-1)
    neg_vis = (vis_embs_gpu @ neg_emb.T).mean(dim=-1)
    raw_vis = (pos_vis - NEG_WEIGHT * neg_vis).cpu()
    v_min, v_max = raw_vis.min(), raw_vis.max()
    vis_scores = (raw_vis - v_min) / (v_max - v_min + 1e-8)

    # ── Fusion & filtrage ──────────────────────────────────────────────────────
    detections: List[Detection] = []
    for t, a_s, v_s in zip(top_ts, top_ascore, vis_scores.tolist()):
        final = audio_weight * a_s + visual_weight * v_s
        if final >= min_final_score:
            detections.append(Detection(
                time_s=t, audio_score=a_s, visual_score=v_s, final_score=final
            ))

    detections = nms(detections)

    # ── Affichage ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {len(detections)} gifle(s) détectée(s) :")
    for i, d in enumerate(detections):
        print(
            f"  #{i+1:02d}  {d.timecode}  "
            f"final={d.final_score:.3f}  "
            f"audio={d.audio_score:.3f}  "
            f"visuel={d.visual_score:.3f}"
        )

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    result = {
        "video": video_path,
        "duration_s": round(duration_s, 1),
        "model": "ImageBind-huge",
        "top_audio_k": top_audio_k,
        "audio_weight": audio_weight,
        "visual_weight": visual_weight,
        "min_final_score": min_final_score,
        "n_detections": len(detections),
        "detections": [d.to_dict() for d in detections],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON -> {output_json}")

    # ── Export clips ───────────────────────────────────────────────────────────
    if export_clips_dir:
        print(f"  Export clips -> {export_clips_dir}")
        for i, d in enumerate(detections):
            export_clip(video_path, d, export_clips_dir, i + 1)

    return detections


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Détecteur de gifles ImageBind")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("--output",        "-o", default="gifles_imagebind.json")
    parser.add_argument("--export-clips",        default=None)
    parser.add_argument("--top-k",         type=int,   default=TOP_AUDIO_K)
    parser.add_argument("--min-score",     type=float, default=0.20)
    parser.add_argument("--audio-weight",  type=float, default=0.5)
    parser.add_argument("--visual-weight", type=float, default=0.5)
    args = parser.parse_args()

    run_detection(
        video_path       = args.video,
        output_json      = args.output,
        export_clips_dir = args.export_clips,
        top_audio_k      = args.top_k,
        audio_weight     = args.audio_weight,
        visual_weight    = args.visual_weight,
        min_final_score  = args.min_score,
    )


if __name__ == "__main__":
    main()
