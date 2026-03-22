"""
gifle_detector_v3.py
====================
Detecteur de gifles - pipeline 4 passes optimal

Passe 1  Onset detection   librosa (CPU, <1 min)      -> ~800 candidats
Passe 2  Audio model       CLAP HTSAT-fused (zero-shot) OU
                           PANNs CNN14 (supervise AudioSet "Slap,smack" #467)
Passe 3  DFN5B ViT-H-378   visuel statique (GPU)      -> ~25 candidats
Passe 4  MediaPipe          wrist velocity + head snap -> ~12-15 resultats

Total : ~15-20 min pour un film entier
Modeles : LAION-CLAP HTSAT-fused / PANNs CNN14 + DFN5B-CLIP-ViT-H-14-378 + MediaPipe Pose
"""

import os, sys, json, argparse, subprocess, tempfile
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2

# ── Constantes ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CLAP_DURATION    = 1.0       # secondes fenetre CLAP
TOP_ONSET_K      = 1000      # max candidats onset
NMS_ONSET_GAP_S  = 0.3       # fusion onsets proches
NMS_FINAL_GAP_S  = 3.0       # NMS final
AUDIO_SURE       = 0.72      # score CLAP -> pas besoin visuel
AUDIO_MIN        = 0.30      # rejet direct
FINAL_MIN_SCORE  = 0.45      # seuil score fusionne
VISUAL_FRAMES    = 3
VISUAL_BATCH     = 8
MEDIAPIPE_FRAMES = 16        # +/-8 frames autour du pic
WRIST_VEL_THRESH = 0.03      # vitesse normalisee min poignet (0-1 espace image)
HEAD_SNAP_THRESH = 0.015     # deplacement lateral tete normalise
CLIP_BEFORE_S    = 2.0
CLIP_AFTER_S     = 3.0

CKPT_DIR   = os.environ.get("IMAGEBIND_CACHE", r"C:\.checkpoints")
CLAP_CKPT  = os.path.join(CKPT_DIR, "music_audioset_epoch_15_esc_90.14.pt")
PANNS_CKPT = os.path.join(CKPT_DIR, "Cnn14_16k_mAP=0.438.pth")
# PANNs CNN14 parametres (doit correspondre au checkpoint 16k)
PANNS_N_FFT    = 512
PANNS_HOP_LEN  = 160
PANNS_MEL_BINS = 64
PANNS_FMIN     = 50
PANNS_FMAX     = 8000
PANNS_SLAP_IDX = 467        # AudioSet "Slap, smack"
PANNS_WIN_S    = 1.0        # fenetre de scan PANNs (secondes)
PANNS_HOP_S    = 0.1        # pas de scan PANNs (secondes)
PANNS_SCALE    = 0.10       # P=PANNS_SCALE -> score normalise=1.0
# Seuils PANNs (en score normalise = raw_P / PANNS_SCALE)
PANNS_SURE     = 0.45       # confident -> pas besoin visuel (P~=0.045)
PANNS_MIN      = 0.08       # rejet direct (P~=0.008)

SLAP_POS_PROMPTS = [
    "a person slapping another person in the face",
    "the sound of a slap hitting skin",
    "a violent face slap",
    "a sharp crack of a palm hitting a cheek",
    "a hand striking a face forcefully",
    "a sudden sharp skin impact sound",
    "someone getting slapped hard",
]
SLAP_NEG_PROMPTS = [
    "a door slamming shut",
    "footsteps on a hard floor",
    "a glass placed on a table",
    "people clapping their hands",
    "a fist punch impact",
    "a book dropped on the floor",
    "a car door closing",
    "background music",
    "someone crying",
]

VISUAL_POS_PROMPTS = [
    "a person slapping another person in the face",
    "a hand striking someone across the face",
    "a violent slap scene in a film",
    "person recoiling from a face slap",
    "a face being slapped with an open hand",
]
VISUAL_NEG_PROMPTS = [
    "a door opening or closing",
    "people having a calm conversation",
    "a person walking",
    "an empty room",
    "two people talking calmly",
]


# ── Utilitaires ─────────────────────────────────────────────────────────────

def tc(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio_mono(video_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr),
           "-vn", "-f", "wav", tmp, "-loglevel", "error"]
    subprocess.run(cmd, check=True)
    wav, _ = librosa.load(tmp, sr=sr, mono=True)
    os.unlink(tmp)
    return wav


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24.0


def extract_frames_around(video_path: str, time_s: float,
                           n_frames: int, fps: float) -> List[np.ndarray]:
    """Extrait n_frames centrees sur time_s via ffmpeg, retourne liste BGR."""
    half = n_frames // 2
    start = max(0.0, time_s - half / fps)
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
            "-vframes", str(n_frames), "-q:v", "2",
            os.path.join(tmpdir, "f%04d.jpg"), "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        for i in range(1, n_frames + 1):
            p = os.path.join(tmpdir, f"f{i:04d}.jpg")
            if os.path.exists(p):
                frames.append(cv2.imread(p))
    return frames


# ── Passe 1 : Onset detection ────────────────────────────────────────────────

def detect_onsets(wav: np.ndarray, sr: int = SAMPLE_RATE,
                  top_k: int = TOP_ONSET_K,
                  nms_gap: float = NMS_ONSET_GAP_S) -> List[float]:
    print("  Calcul onset strength...", flush=True)
    hop = 512
    onset_env = librosa.onset.onset_strength(
        y=wav, sr=sr, hop_length=hop, aggregate=np.median, fmax=8000
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        backtrack=False, pre_max=3, post_max=3,
        pre_avg=5, post_avg=5, delta=0.05, wait=1
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    onset_strengths = onset_env[onset_frames]

    order = np.argsort(onset_strengths)[::-1]
    times_sorted = onset_times[order]

    kept = []
    for t in times_sorted:
        if all(abs(t - k) >= nms_gap for k in kept):
            kept.append(float(t))
        if len(kept) >= top_k:
            break

    print(f"  {len(onset_frames)} onsets -> {len(kept)} candidats (NMS={nms_gap}s)", flush=True)
    return kept


# ── Passe 2 : CLAP ───────────────────────────────────────────────────────────

def load_clap(device: str):
    import laion_clap
    if not os.path.exists(CLAP_CKPT):
        raise FileNotFoundError(f"Checkpoint CLAP manquant: {CLAP_CKPT}")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(CLAP_CKPT)
    model.eval()
    if device == "cuda":
        model = model.cuda()
    return model


def clap_score(clap_model, wav: np.ndarray, timestamps: List[float],
               device: str) -> np.ndarray:
    all_texts = SLAP_POS_PROMPTS + SLAP_NEG_PROMPTS
    with torch.no_grad():
        text_embs = clap_model.get_text_embedding(all_texts, use_tensor=True)
    text_embs = F.normalize(text_embs, dim=-1)
    pos_embs = text_embs[:len(SLAP_POS_PROMPTS)]
    neg_embs = text_embs[len(SLAP_POS_PROMPTS):]

    half = int(CLAP_DURATION * SAMPLE_RATE / 2)
    target_len = int(CLAP_DURATION * SAMPLE_RATE)
    raw_scores = []
    BATCH = 64

    print(f"  Scoring {len(timestamps)} candidats CLAP...", flush=True)
    for i in range(0, len(timestamps), BATCH):
        batch_ts = timestamps[i:i+BATCH]
        clips = []
        for t in batch_ts:
            c = int(t * SAMPLE_RATE)
            seg = wav[max(0, c-half):min(len(wav), c+half)]
            if len(seg) < target_len:
                seg = np.pad(seg, (0, target_len - len(seg)))
            clips.append(seg[:target_len].astype(np.float32))

        clips_t = torch.from_numpy(np.stack(clips))
        if device == "cuda":
            clips_t = clips_t.cuda()
        with torch.no_grad():
            audio_embs = clap_model.get_audio_embedding_from_data(
                x=clips_t, use_tensor=True
            )
        audio_embs = F.normalize(audio_embs, dim=-1)
        pos_s = (audio_embs @ pos_embs.T).mean(dim=-1)
        neg_s = (audio_embs @ neg_embs.T).mean(dim=-1)
        raw_scores.extend((pos_s - 0.4 * neg_s).cpu().tolist())

        if (i // BATCH + 1) % 4 == 0 or i + BATCH >= len(timestamps):
            print(f"    {min(i+BATCH, len(timestamps))}/{len(timestamps)}...", flush=True)

    raw = np.array(raw_scores)
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


# ── Passe 2b : PANNs CNN14 (alternative supervisee a CLAP) ──────────────────

class _ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x, pool=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':     x = F.avg_pool2d(x, kernel_size=pool)
        elif pool_type == 'max':   x = F.max_pool2d(x, kernel_size=pool)
        elif pool_type == 'avg+max': x = F.avg_pool2d(x, kernel_size=pool) + F.max_pool2d(x, kernel_size=pool)
        return x


class _STFT1d(nn.Module):
    """STFT via 1D conv, miroir du spectrogram_extractor PANNs."""
    def __init__(self, n_fft=512, hop=160):
        super().__init__()
        n_uniq = n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop = hop
        self.conv_real = nn.Conv1d(1, n_uniq, n_fft, stride=hop, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, n_uniq, n_fft, stride=hop, padding=0, bias=False)
    def forward(self, w):
        x = w.unsqueeze(1)
        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        r = self.conv_real(x); i = self.conv_imag(x)
        p = (r**2 + i**2).transpose(1, 2).unsqueeze(1)  # (B, 1, T, n_uniq)
        return p


class _LogMel(nn.Module):
    """Mel filterbank + log10, miroir du logmel_extractor PANNs."""
    def __init__(self, sr=16000, n_fft=512, n_mels=64, fmin=50, fmax=8000):
        super().__init__()
        melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.register_buffer('melW', torch.FloatTensor(melW))
    def forward(self, p):
        mel = torch.clamp(p @ self.melW, min=1e-10)
        return 10.0 * torch.log10(mel)


class _Cnn14(nn.Module):
    """PANNs CNN14, 527 classes AudioSet, 16kHz."""
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = _STFT1d(PANNS_N_FFT, PANNS_HOP_LEN)
        self.logmel_extractor = _LogMel(SAMPLE_RATE, PANNS_N_FFT, PANNS_MEL_BINS,
                                         PANNS_FMIN, PANNS_FMAX)
        self.bn0         = nn.BatchNorm2d(PANNS_MEL_BINS)
        self.conv_block1 = _ConvBlock2d(1, 64)
        self.conv_block2 = _ConvBlock2d(64, 128)
        self.conv_block3 = _ConvBlock2d(128, 256)
        self.conv_block4 = _ConvBlock2d(256, 512)
        self.conv_block5 = _ConvBlock2d(512, 1024)
        self.conv_block6 = _ConvBlock2d(1024, 2048)
        self.fc1         = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, w):
        x = self.spectrogram_extractor.stft(w)     # (B,1,T,257)
        x = self.logmel_extractor(x)               # (B,1,T,64)
        x = x.transpose(1, 3); x = self.bn0(x); x = x.transpose(1, 3)
        x = self.conv_block1(x, pool=(2,2))
        x = F.dropout(x, 0.2, training=False)
        x = self.conv_block2(x, pool=(2,2))
        x = F.dropout(x, 0.2, training=False)
        x = self.conv_block3(x, pool=(2,2))
        x = F.dropout(x, 0.2, training=False)
        x = self.conv_block4(x, pool=(2,2))
        x = F.dropout(x, 0.2, training=False)
        x = self.conv_block5(x, pool=(2,2))
        x = F.dropout(x, 0.2, training=False)
        x = self.conv_block6(x, pool=(1,1))
        x = F.dropout(x, 0.2, training=False)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2); x2 = torch.mean(x, dim=2); x = x1 + x2
        x = F.dropout(x, 0.5, training=False)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, 0.5, training=False)
        return torch.sigmoid(self.fc_audioset(x))  # (B, 527)


def load_panns(device: str):
    """Charge PANNs CNN14 (checkpoint 16kHz) depuis PANNS_CKPT."""
    if not os.path.exists(PANNS_CKPT):
        raise FileNotFoundError(f"Checkpoint PANNs manquant: {PANNS_CKPT}")
    model = _Cnn14()
    ckpt = torch.load(PANNS_CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f"  [PANNs WARN] Missing: {len(missing)} keys", flush=True)
    model.eval().to(device)
    print("  PANNs CNN14 charge (16kHz, 527 classes AudioSet)", flush=True)
    return model


def panns_score(panns_model, wav: np.ndarray, timestamps: List[float],
                device: str) -> np.ndarray:
    """
    Pour chaque timestamp, scanne une fenetre PANNs_WIN_S autour avec hop PANNs_HOP_S.
    Retourne le max P(Slap,smack) normalise par PANNS_SCALE, capee a 1.0.
    Score: 1.0 = P>=0.10, 0 = silence/bruit.
    """
    win_n = int(PANNS_WIN_S * SAMPLE_RATE)
    ctx_n = int(1.5 * SAMPLE_RATE)  # +/-1.5s de contexte autour du timestamp
    hop_n = int(PANNS_HOP_S * SAMPLE_RATE)
    n_wav = len(wav)

    raw_scores = []
    print(f"  Scoring {len(timestamps)} candidats PANNs CNN14...", flush=True)
    for i, t in enumerate(timestamps):
        c = int(t * SAMPLE_RATE)
        seg_start = max(0, c - ctx_n)
        seg_end   = min(n_wav, c + ctx_n + win_n)
        seg = wav[seg_start:seg_end]

        max_p = 0.0
        for start in range(0, max(1, len(seg) - win_n + 1), hop_n):
            chunk = seg[start:start + win_n]
            if len(chunk) < win_n:
                chunk = np.pad(chunk, (0, win_n - len(chunk)))
            wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                p = panns_model(wav_t)[0, PANNS_SLAP_IDX].item()
            max_p = max(max_p, p)

        raw_scores.append(max_p)
        if (i + 1) % 50 == 0 or i + 1 == len(timestamps):
            print(f"    {i+1}/{len(timestamps)}...", flush=True)

    raw = np.array(raw_scores)
    # Normalisation calibree : P=PANNS_SCALE -> score=1.0
    return np.clip(raw / PANNS_SCALE, 0.0, 1.0)


# ── Passe 3 : DFN5B visuel statique ─────────────────────────────────────────

def load_dfn5b(device: str):
    import open_clip
    print("  Chargement DFN5B-CLIP-ViT-H-14-378...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14-378-quickgelu', pretrained='dfn5b'
    )
    tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
    model.eval().to(device)
    return model, preprocess, tokenizer


def dfn5b_score(dfn_model, preprocess, tokenizer,
                video_path: str, timestamps: List[float], device: str) -> np.ndarray:
    import open_clip
    from PIL import Image

    all_texts = VISUAL_POS_PROMPTS + VISUAL_NEG_PROMPTS
    tokens = tokenizer(all_texts).to(device)
    with torch.no_grad():
        text_embs = dfn_model.encode_text(tokens)
    text_embs = F.normalize(text_embs, dim=-1)
    pos_embs = text_embs[:len(VISUAL_POS_PROMPTS)]
    neg_embs = text_embs[len(VISUAL_POS_PROMPTS):]

    n = len(timestamps)
    n_frames_total = n * VISUAL_FRAMES
    print(f"  Extraction+encoding {n_frames_total} frames DFN5B (streaming)...", flush=True)

    # Streaming : on extrait et encode par timestamp (pas d'accumulation RAM)
    ts_scores = []
    buf_imgs = []
    buf_ts_idx = []   # (ts_index, frame_index)

    def flush_buf():
        if not buf_imgs:
            return
        batch = torch.stack(buf_imgs).to(device)
        with torch.no_grad():
            img_embs = F.normalize(dfn_model.encode_image(batch), dim=-1)
        ps = (img_embs @ pos_embs.T).mean(dim=-1)
        ns = (img_embs @ neg_embs.T).mean(dim=-1)
        scores = (ps - 0.35 * ns).cpu().tolist()
        del batch, img_embs
        torch.cuda.empty_cache()
        buf_imgs.clear()
        return scores

    per_ts = [[] for _ in range(n)]
    done = 0
    for ti, t in enumerate(timestamps):
        frame_scores = []
        buf = []
        for off in np.linspace(-0.5, 0.5, VISUAL_FRAMES):
            t_seek = max(0, t + off)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp = f.name
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t_seek), "-i", video_path,
                 "-vframes", "1", "-q:v", "3", tmp, "-loglevel", "error"],
                check=True
            )
            try:
                buf.append(preprocess(Image.open(tmp).convert("RGB")))
            except Exception:
                buf.append(torch.zeros(3, 378, 378))
            os.unlink(tmp)

        # Encoder les VISUAL_FRAMES frames de ce timestamp
        batch = torch.stack(buf).to(device)
        with torch.no_grad():
            img_embs = F.normalize(dfn_model.encode_image(batch), dim=-1)
        ps = (img_embs @ pos_embs.T).mean(dim=-1)
        ns = (img_embs @ neg_embs.T).mean(dim=-1)
        score = (ps - 0.35 * ns).mean().item()
        del batch, img_embs, buf
        torch.cuda.empty_cache()
        per_ts.append(score)
        done += VISUAL_FRAMES
        if (ti + 1) % 20 == 0 or ti == n - 1:
            print(f"    {done}/{n_frames_total}...", flush=True)

    raw = np.array(per_ts[n:])  # skip empty init
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros(n)


# ── Visual sweep 1fps ────────────────────────────────────────────────────────

def visual_sweep(video_path: str, output_json: str,
                 dfn_model, preprocess, tokenizer,
                 device: str, fps_sweep: float = 1.0,
                 min_score: float = 0.50,
                 nms_gap_s: float = NMS_FINAL_GAP_S,
                 sweep_cache: Optional[str] = None) -> List[float]:
    """
    Scan visuel complet du film a fps_sweep images/seconde via DFN5B.
    Retourne liste de timestamps candidats (peaks visuels).
    Cache les embeddings pour reutilisation (prompt-independent).
    """
    import open_clip

    cache_path = sweep_cache or (output_json.replace(".json", SWEEP_CACHE_SUFFIX))

    # ── Embeddings texte ──────────────────────────────────────────────────
    all_texts = VISUAL_POS_PROMPTS + VISUAL_NEG_PROMPTS
    tokens = tokenizer(all_texts).to(device)
    with torch.no_grad():
        text_embs = dfn_model.encode_text(tokens)
    text_embs = F.normalize(text_embs, dim=-1)
    pos_embs = text_embs[:len(VISUAL_POS_PROMPTS)]
    neg_embs = text_embs[len(VISUAL_POS_PROMPTS):]

    # ── Embeddings visuels (cache ou calcul) ──────────────────────────────
    if os.path.exists(cache_path):
        print(f"  Cache sweep trouve: {cache_path}", flush=True)
        cache = torch.load(cache_path, weights_only=True)
        img_embs_all = cache["embs"]        # (N, 512)
        frame_times  = cache["frame_times"] # list[float]
        print(f"  {len(frame_times)} frames chargees depuis cache", flush=True)
    else:
        from PIL import Image
        # Dossier persistant a cote du cache (pas de TemporaryDirectory = pas d'OOM)
        tmpdir = cache_path.replace(".pt", "_frames")
        os.makedirs(tmpdir, exist_ok=True)

        print(f"  Extraction frames {fps_sweep}fps (ffmpeg) -> {tmpdir}...", flush=True)
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps_sweep}",
            "-q:v", "4",
            os.path.join(tmpdir, "f%06d.jpg"),
            "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)

        frame_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jpg"))
        n_frames = len(frame_files)
        frame_times = [i / fps_sweep for i in range(n_frames)]
        print(f"  {n_frames} frames extraites", flush=True)

        # Encoder par batch — imgs_buf vide apres chaque batch (pas d'accumulation RAM)
        print(f"  Encoding {n_frames} frames DFN5B (batch={VISUAL_BATCH})...", flush=True)
        all_embs = []
        imgs_buf = []
        fnames_buf = []
        for fi, fname in enumerate(frame_files):
            fnames_buf.append(os.path.join(tmpdir, fname))
            if len(fnames_buf) == VISUAL_BATCH or fi == n_frames - 1:
                imgs = []
                for fp in fnames_buf:
                    try:
                        imgs.append(preprocess(Image.open(fp).convert("RGB")))
                    except Exception:
                        imgs.append(torch.zeros(3, 378, 378))
                batch = torch.stack(imgs).to(device)
                with torch.no_grad():
                    embs = F.normalize(dfn_model.encode_image(batch), dim=-1)
                all_embs.append(embs.cpu())
                del batch, imgs
                fnames_buf = []
                if (fi // VISUAL_BATCH + 1) % 20 == 0 or fi == n_frames - 1:
                    print(f"    {fi+1}/{n_frames}...", flush=True)

        img_embs_all = torch.cat(all_embs, dim=0)  # (N, 512)
        del all_embs

        # Sauvegarder cache
        torch.save({"embs": img_embs_all, "frame_times": frame_times}, cache_path)
        print(f"  Cache sweep sauvegarde: {cache_path}", flush=True)

    # ── Scores visuels sur toute la timeline ─────────────────────────────
    img_embs_all = img_embs_all.to(device)
    pos_s = (img_embs_all @ pos_embs.T).mean(dim=-1)
    neg_s = (img_embs_all @ neg_embs.T).mean(dim=-1)
    raw_scores = (pos_s - 0.35 * neg_s).cpu().numpy()

    # Normaliser
    lo, hi = raw_scores.min(), raw_scores.max()
    scores_norm = (raw_scores - lo) / (hi - lo) if hi > lo else raw_scores

    # Trouver peaks (candidats visuels)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(scores_norm, height=min_score, distance=int(nms_gap_s * fps_sweep))
    peak_times = [frame_times[p] for p in peaks]
    peak_scores = scores_norm[peaks]

    print(f"  {len(peaks)} candidats visuels (seuil={min_score}, NMS={nms_gap_s}s)", flush=True)

    # Afficher les top candidats
    order = np.argsort(peak_scores)[::-1]
    for i in order[:20]:
        print(f"    {tc(peak_times[i])}  vis={peak_scores[i]:.3f}", flush=True)

    return list(peak_times), list(peak_scores.tolist())


# ── Passe 4 : MediaPipe wrist velocity + head snap ───────────────────────────

POSE_MODEL_PATH   = os.path.join(CKPT_DIR, "pose_landmarker_heavy.task")
SWEEP_CACHE_SUFFIX = "_visual_sweep_cache.pt"   # cache embeddings DFN5B 1fps

def mediapipe_motion_score(video_path: str, timestamps: List[float],
                            fps: float) -> Tuple[np.ndarray, List[dict]]:
    """
    Pour chaque timestamp :
    - Extrait MEDIAPIPE_FRAMES frames
    - Detecte pose via MediaPipe Tasks API (pose_landmarker_heavy)
    - Calcule vitesse max poignet (wrist velocity)
    - Calcule deplacement lateral tete (head snap)
    Retourne scores [0,1] et details par timestamp.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"Modele MediaPipe manquant: {POSE_MODEL_PATH}\n"
            "Telechargez pose_landmarker_heavy.task depuis storage.googleapis.com"
        )

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=4,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    scores = []
    details = []

    print(f"  Analyse MediaPipe sur {len(timestamps)} candidats...", flush=True)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for idx, t in enumerate(timestamps):
            frames = extract_frames_around(video_path, t, MEDIAPIPE_FRAMES, fps)
            if len(frames) < 4:
                scores.append(0.0)
                details.append({"wrist_vel": 0.0, "head_snap": 0.0, "n_frames": len(frames)})
                continue

            # Landmarks par frame via Tasks API
            landmarks_per_frame = []
            for frame in frames:
                if frame is None:
                    landmarks_per_frame.append(None)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    # Prendre la premiere pose detectee
                    lm = result.pose_landmarks[0]
                    landmarks_per_frame.append({
                        "left_wrist":  (lm[15].x, lm[15].y, lm[15].visibility),
                        "right_wrist": (lm[16].x, lm[16].y, lm[16].visibility),
                        "nose":        (lm[0].x,  lm[0].y,  lm[0].visibility),
                        "left_ear":    (lm[7].x,  lm[7].y,  lm[7].visibility),
                        "right_ear":   (lm[8].x,  lm[8].y,  lm[8].visibility),
                    })
                else:
                    landmarks_per_frame.append(None)

            valid = [(i, lm) for i, lm in enumerate(landmarks_per_frame) if lm is not None]
            if len(valid) < 3:
                scores.append(0.0)
                details.append({"wrist_vel": 0.0, "head_snap": 0.0, "n_frames": len(valid)})
                continue

            # Wrist velocity max (poignet le plus rapide entre frames consecutives)
            max_wrist_vel = 0.0
            for i in range(len(valid) - 1):
                fi, lm_a = valid[i]
                fj, lm_b = valid[i+1]
                if fj - fi > 3:
                    continue
                for side in ["left_wrist", "right_wrist"]:
                    if lm_a[side][2] > 0.3 and lm_b[side][2] > 0.3:
                        dx = lm_b[side][0] - lm_a[side][0]
                        dy = lm_b[side][1] - lm_a[side][1]
                        vel = (dx**2 + dy**2) ** 0.5
                        max_wrist_vel = max(max_wrist_vel, vel)

            # Head snap : deplacement lateral du nez dans la 2eme moitie des frames
            # (la tete se deplace APRES l'impact)
            mid = len(valid) // 2
            head_snap = 0.0
            nose_xs_before = [lm["nose"][0] for _, lm in valid[:mid] if lm["nose"][2] > 0.3]
            nose_xs_after  = [lm["nose"][0] for _, lm in valid[mid:] if lm["nose"][2] > 0.3]
            if nose_xs_before and nose_xs_after:
                x_before = np.mean(nose_xs_before)
                x_after  = np.mean(nose_xs_after)
                head_snap = abs(x_after - x_before)

            # Score combine : wrist velocity + head snap
            wrist_norm = min(max_wrist_vel / 0.15, 1.0)   # 0.15 = vitesse "rapide" normalisee
            head_norm  = min(head_snap / 0.05, 1.0)        # 0.05 = snap significatif

            # Double signal = tres confiant ; signal unique = partiel
            if wrist_norm > 0.3 and head_norm > 0.3:
                score = 0.5 * wrist_norm + 0.5 * head_norm
            elif wrist_norm > 0.5:
                score = 0.6 * wrist_norm   # bras rapide suffit si tres prononce
            elif head_norm > 0.5:
                score = 0.5 * head_norm    # head snap seul possible (gifle hors champ)
            else:
                score = 0.3 * wrist_norm + 0.3 * head_norm

            scores.append(float(score))
            details.append({
                "wrist_vel": round(max_wrist_vel, 4),
                "head_snap": round(head_snap, 4),
                "wrist_norm": round(wrist_norm, 3),
                "head_norm": round(head_norm, 3),
                "n_valid_frames": len(valid),
            })

            if (idx + 1) % 5 == 0 or idx + 1 == len(timestamps):
                print(f"    {idx+1}/{len(timestamps)}...", flush=True)

    arr = np.array(scores)
    lo, hi = arr.min(), arr.max()
    normalized = (arr - lo) / (hi - lo) if hi > lo else arr
    return normalized, details


# ── NMS & export ─────────────────────────────────────────────────────────────

def apply_nms(detections: list, gap_s: float) -> list:
    candidates = sorted(detections, key=lambda x: x["final_score"], reverse=True)
    kept = []
    for d in candidates:
        if all(abs(d["time_s"] - k["time_s"]) >= gap_s for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda x: x["time_s"])


def export_clip(video_path, time_s, out_path,
                before=CLIP_BEFORE_S, after=CLIP_AFTER_S):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    start = max(0, time_s - before)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", video_path,
         "-t", str(before + after), "-c", "copy", out_path, "-loglevel", "error"],
        check=True
    )


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_detection(
    video_path: str,
    output_json: str,
    export_clips_dir: Optional[str] = None,
    top_onset_k: int = TOP_ONSET_K,
    audio_weight: float = 0.35,
    visual_weight: float = 0.30,
    motion_weight: float = 0.35,
    min_final_score: float = FINAL_MIN_SCORE,
    audio_sure: float = AUDIO_SURE,
    audio_min: float = AUDIO_MIN,
    nms_gap_s: float = NMS_FINAL_GAP_S,
    skip_visual: bool = False,
    skip_motion: bool = False,
    visual_sweep_mode: bool = False,
    sweep_cache: Optional[str] = None,
    audio_model: str = "panns",   # "clap" | "panns" | "both"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_label = {"clap": "CLAP HTSAT-fused", "panns": "PANNs CNN14",
                   "both": "CLAP+PANNs"}.get(audio_model, audio_model)
    print("=" * 70)
    print(f"  Gifle Detector v3  |  {audio_label} + DFN5B + MediaPipe")
    print(f"  Video  : {video_path}")
    print(f"  Device : {device}")
    print("=" * 70, flush=True)

    fps = get_video_fps(video_path)
    print(f"  FPS video : {fps:.2f}", flush=True)

    # ── Passe 1 : Onset ────────────────────────────────────────────────────
    print("\n[Passe 1] Onset detection (CPU)...", flush=True)
    wav = extract_audio_mono(video_path)
    duration_s = len(wav) / SAMPLE_RATE
    print(f"  Duree : {duration_s/60:.1f} min", flush=True)
    onset_times = detect_onsets(wav, top_k=top_onset_k)

    # ── Passe 2 : Audio classification (CLAP | PANNs | les deux) ──────────
    use_clap  = audio_model in ("clap", "both")
    use_panns = audio_model in ("panns", "both")

    # Seuils adaptes selon modele
    _audio_sure = PANNS_SURE if use_panns and not use_clap else audio_sure
    _audio_min  = PANNS_MIN  if use_panns and not use_clap else audio_min

    audio_scores = np.zeros(len(onset_times))

    if use_clap:
        print("\n[Passe 2] Classification audio LAION-CLAP HTSAT-fused...", flush=True)
        clap_model = load_clap(device)
        clap_s = clap_score(clap_model, wav, onset_times, device)
        del clap_model
        torch.cuda.empty_cache()
        audio_scores = np.maximum(audio_scores, clap_s)

    if use_panns:
        print("\n[Passe 2] Classification audio PANNs CNN14 (AudioSet 'Slap,smack')...", flush=True)
        panns_model = load_panns(device)
        panns_s = panns_score(panns_model, wav, onset_times, device)
        del panns_model
        torch.cuda.empty_cache()
        if use_clap:
            # Ensemble : moyenne des deux scores normalises
            audio_scores = 0.5 * audio_scores + 0.5 * panns_s
        else:
            audio_scores = panns_s

    del wav

    keep_mask = audio_scores >= _audio_min
    filt_ts = [t for t, k in zip(onset_times, keep_mask) if k]
    filt_sc = audio_scores[keep_mask]
    print(f"  {keep_mask.sum()}/{len(onset_times)} passes seuil audio ({_audio_min:.3f})", flush=True)

    sure_mask  = filt_sc >= _audio_sure
    ambigu_mask = ~sure_mask
    sure_ts   = [t for t, k in zip(filt_ts, sure_mask) if k]
    sure_sc   = filt_sc[sure_mask]
    ambigu_ts = [t for t, k in zip(filt_ts, ambigu_mask) if k]
    ambigu_sc = filt_sc[ambigu_mask]
    print(f"  -> {len(sure_ts)} surs (>={_audio_sure:.3f}), {len(ambigu_ts)} ambigus", flush=True)

    # ── Visual sweep (optionnel) : DFN5B sur tout le film 1fps ────────────
    sweep_extra_ts = []
    sweep_extra_vs = []
    dfn_model_ref  = [None]   # pour reutiliser si deja charge

    if visual_sweep_mode:
        print("\n[Sweep] Visual sweep 1fps DFN5B...", flush=True)
        dfn_model, preprocess, tokenizer = load_dfn5b(device)
        dfn_model_ref[0] = (dfn_model, preprocess, tokenizer)
        sweep_ts, sweep_vs = visual_sweep(
            video_path, output_json, dfn_model, preprocess, tokenizer,
            device, fps_sweep=1.0, min_score=0.45,
            nms_gap_s=nms_gap_s, sweep_cache=sweep_cache
        )
        sweep_extra_ts = sweep_ts
        sweep_extra_vs = sweep_vs
        print(f"  {len(sweep_ts)} candidats visuels supplementaires", flush=True)

    # ── Passe 3 : DFN5B ────────────────────────────────────────────────────
    vis_sure   = np.ones(len(sure_ts))
    vis_ambigu = np.zeros(len(ambigu_ts))

    if not skip_visual and len(ambigu_ts) > 0:
        print(f"\n[Passe 3] DFN5B-CLIP visuel ({len(ambigu_ts)} ambigus)...", flush=True)
        if dfn_model_ref[0] is not None:
            dfn_model, preprocess, tokenizer = dfn_model_ref[0]
        else:
            dfn_model, preprocess, tokenizer = load_dfn5b(device)
        vis_ambigu = dfn5b_score(dfn_model, preprocess, tokenizer,
                                  video_path, ambigu_ts, device)
        del dfn_model
        dfn_model_ref[0] = None
        torch.cuda.empty_cache()

    # Fusion provisoire pour selectionner candidats pour MediaPipe
    # Inclure sweep candidats (visuels seuls, audio=0)
    sweep_audio = np.zeros(len(sweep_extra_ts))
    sweep_visual = np.array(sweep_extra_vs) if sweep_extra_vs else np.zeros(0)

    all_ts     = sure_ts + ambigu_ts + sweep_extra_ts
    all_audio  = np.concatenate([sure_sc, ambigu_sc, sweep_audio])
    all_visual = np.concatenate([vis_sure, vis_ambigu, sweep_visual])

    # Score pre-MediaPipe
    pre_scores = audio_weight * all_audio + visual_weight * all_visual
    # Garder candidats au-dessus du seuil pour MediaPipe
    mp_mask = pre_scores >= (FINAL_MIN_SCORE * 0.7)
    mp_ts   = [t for t, k in zip(all_ts, mp_mask) if k]
    mp_idx  = [i for i, k in enumerate(mp_mask) if k]
    print(f"\n[Passe 4] MediaPipe sur {len(mp_ts)} candidats (pre-filtre)...", flush=True)

    motion_scores_full = np.zeros(len(all_ts))
    motion_details_full = [{}] * len(all_ts)

    if not skip_motion and len(mp_ts) > 0:
        mp_norm, mp_details = mediapipe_motion_score(video_path, mp_ts, fps)
        for i, orig_i in enumerate(mp_idx):
            motion_scores_full[orig_i] = mp_norm[i]
            motion_details_full[orig_i] = mp_details[i]

    # ── Fusion finale ──────────────────────────────────────────────────────
    detections = []
    for i, t in enumerate(all_ts):
        a = float(all_audio[i])
        v = float(all_visual[i])
        m = float(motion_scores_full[i])
        is_sure = i < len(sure_ts)

        if skip_motion:
            final = (audio_weight + motion_weight/2) * a + (visual_weight + motion_weight/2) * v
        elif skip_visual:
            final = (audio_weight + visual_weight/2) * a + (motion_weight + visual_weight/2) * m
        else:
            final = audio_weight * a + visual_weight * v + motion_weight * m

        detections.append({
            "time_s": float(t),
            "timecode": tc(t),
            "audio_score": round(a, 3),
            "visual_score": round(v, 3),
            "motion_score": round(m, 3),
            "final_score": round(final, 3),
            "audio_certain": bool(is_sure),
            "motion_detail": motion_details_full[i],
        })

    detections = [d for d in detections if d["final_score"] >= min_final_score]
    detections = apply_nms(detections, nms_gap_s)

    # ── Affichage ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {len(detections)} gifles detectees  (seuil={min_final_score}, NMS={nms_gap_s}s)")
    print(f"{'='*70}")
    for i, d in enumerate(detections):
        tag = "[A]" if d["audio_certain"] else "[AVM]"
        print(f"  #{i+1:03d} {d['timecode']}  final={d['final_score']:.3f}"
              f"  audio={d['audio_score']:.3f}  vis={d['visual_score']:.3f}"
              f"  motion={d['motion_score']:.3f}  {tag}", flush=True)

    # ── JSON ───────────────────────────────────────────────────────────────
    _model_audio_name = {
        "clap":  "LAION-CLAP-HTSAT-fused",
        "panns": "PANNs-CNN14-16k-AudioSet467",
        "both":  "LAION-CLAP-HTSAT-fused + PANNs-CNN14",
    }.get(audio_model, audio_model)
    result = {
        "video": video_path,
        "duration_s": round(duration_s, 1),
        "model_audio": _model_audio_name,
        "model_visual": "DFN5B-CLIP-ViT-H-14-378",
        "model_motion": "MediaPipe Pose (complexity=2)",
        "weights": {"audio": audio_weight, "visual": visual_weight, "motion": motion_weight},
        "thresholds": {"audio_sure": _audio_sure, "audio_min": _audio_min,
                       "final_min": min_final_score, "nms_gap_s": nms_gap_s},
        "stats": {
            "onset_candidates": len(onset_times),
            "after_audio": int(keep_mask.sum()),
            "after_visual_filter": len(mp_ts),
            "final_detections": len(detections),
        },
        "detections": detections,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON -> {output_json}", flush=True)

    if export_clips_dir:
        print(f"  Export clips -> {export_clips_dir}/", flush=True)
        for i, d in enumerate(detections):
            name = f"gifle_{i+1:03d}_{d['timecode'].replace(':','-')}_s{d['final_score']:.2f}.mp4"
            export_clip(video_path, d["time_s"], os.path.join(export_clips_dir, name))
        print(f"  {len(detections)} clips exportes.", flush=True)

    return detections


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gifle Detector v3 - CLAP + DFN5B + MediaPipe")
    parser.add_argument("video")
    parser.add_argument("--output", default="gifles_v3.json")
    parser.add_argument("--export-clips", default=None, dest="export_clips")
    parser.add_argument("--top-k", type=int, default=TOP_ONSET_K)
    parser.add_argument("--min-score", type=float, default=FINAL_MIN_SCORE)
    parser.add_argument("--audio-sure", type=float, default=AUDIO_SURE)
    parser.add_argument("--audio-min", type=float, default=AUDIO_MIN)
    parser.add_argument("--nms-gap", type=float, default=NMS_FINAL_GAP_S)
    parser.add_argument("--audio-weight", type=float, default=0.35)
    parser.add_argument("--visual-weight", type=float, default=0.30)
    parser.add_argument("--motion-weight", type=float, default=0.35)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--visual-sweep", action="store_true", dest="visual_sweep",
                        help="Scan visuel complet 1fps DFN5B (garantit detection gifles visibles)")
    parser.add_argument("--sweep-cache", default=None, dest="sweep_cache",
                        help="Cache embeddings visuels sweep (.pt) pour reutilisation")
    parser.add_argument("--audio-model", default="panns", dest="audio_model",
                        choices=["clap", "panns", "both"],
                        help="Modele audio: clap (zero-shot), panns (supervise AudioSet), both (ensemble)")
    args = parser.parse_args()

    run_detection(
        video_path=args.video,
        output_json=args.output,
        export_clips_dir=args.export_clips,
        top_onset_k=args.top_k,
        audio_weight=args.audio_weight,
        visual_weight=args.visual_weight,
        motion_weight=args.motion_weight,
        min_final_score=args.min_score,
        audio_sure=args.audio_sure,
        audio_min=args.audio_min,
        nms_gap_s=args.nms_gap,
        skip_visual=args.skip_visual,
        skip_motion=args.skip_motion,
        visual_sweep_mode=args.visual_sweep,
        sweep_cache=args.sweep_cache,
        audio_model=args.audio_model,
    )


if __name__ == "__main__":
    main()
