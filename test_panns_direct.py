"""
Test PANNs CNN14 directly (full model including internal spectrogram extractor).
Scans AudioSet class #467 "Slap, smack" with sliding windows on Pieta 15:00-16:00.
"""

import os, sys, subprocess, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# --- CONFIG ---
CKPT     = r"C:\.checkpoints\Cnn14_16k_mAP=0.438.pth"
FILM     = r"F:\[kr]\copie_de_[kr]_D\kim_ki_duk\Pieta.2012.1080p.BluRay.x264.AAC5.1-[YTS.MX].mp4"
T_START  = 900        # 15:00
T_END    = 960        # 16:00
OUTPUT_FILE = r"G:\projets_ia\slaps\panns_test_result.txt"
WIN_S    = 1.0        # window 1.0s (safe for CNN14 with 5x pool(2,2))
HOP_S    = 0.05       # hop 50ms -> fine temporal resolution
SR       = 16000
# PANNs 16k model parameters (must match checkpoint)
N_FFT    = 512        # window_size=512 -> conv_real.weight [257,1,512]
HOP_LEN  = 160        # 10ms @ 16kHz
MEL_BINS = 64
FMIN     = 50
FMAX     = 8000
SLAP_IDX = 467        # AudioSet "Slap, smack"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# --- SPECTROGRAM EXTRACTOR (mirrors PANNs internal STFT conv) ---

class Spectrogram(nn.Module):
    """STFT via learnable conv layers, matching PANNs checkpoint."""
    def __init__(self, n_fft=512, hop_length=160, freeze_parameters=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freeze_parameters = freeze_parameters
        # Fourier basis stored as conv weights
        self.stft = _STFT(n_fft=n_fft, hop_length=hop_length)
        if freeze_parameters:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        """waveform: (B, samples) -> spectrogram: (B, 1, T, n_fft//2+1)"""
        return self.stft(waveform)


class _STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Build DFT basis
        n_uniq = n_fft // 2 + 1  # 257
        # conv_real: [n_uniq, 1, n_fft], conv_imag: [n_uniq, 1, n_fft]
        self.conv_real = nn.Conv1d(1, n_uniq, n_fft, stride=hop_length, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, n_uniq, n_fft, stride=hop_length, padding=0, bias=False)

    def forward(self, waveform):
        """waveform: (B, samples) -> power_spec: (B, 1, T, n_fft//2+1)"""
        x = waveform.unsqueeze(1)  # (B, 1, samples)
        # Pad with reflect
        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        real = self.conv_real(x)   # (B, n_uniq, T)
        imag = self.conv_imag(x)   # (B, n_uniq, T)
        power = real ** 2 + imag ** 2  # (B, n_uniq, T)
        power = power.transpose(1, 2)  # (B, T, n_uniq)
        power = power.unsqueeze(1)     # (B, 1, T, n_uniq)
        return power


class LogmelFilterBank(nn.Module):
    """Mel filterbank + log10, mirrors PANNs checkpoint."""
    def __init__(self, sr=16000, n_fft=512, n_mels=64, fmin=50, fmax=8000,
                 freeze_parameters=True):
        super().__init__()
        self.freeze_parameters = freeze_parameters
        # melW: [n_fft//2+1, n_mels]
        melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                   fmin=fmin, fmax=fmax).T  # (n_fft//2+1, n_mels)
        self.register_buffer('melW', torch.FloatTensor(melW))
        if freeze_parameters:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, power_spec):
        """power_spec: (B, 1, T, n_fft//2+1) -> logmel: (B, 1, T, n_mels)"""
        # matmul: (..., n_fft//2+1) x (n_fft//2+1, n_mels) -> (..., n_mels)
        mel = torch.clamp(power_spec @ self.melW, min=1e-10)
        logmel = 10.0 * torch.log10(mel)  # (B, 1, T, n_mels)
        return logmel


# --- CNN14 ARCHITECTURE ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14(nn.Module):
    """Full CNN14 with internal spectrogram+mel extractor, matching PANNs checkpoint."""
    def __init__(self, sr=16000, n_fft=512, hop_length=160, mel_bins=64,
                 fmin=50, fmax=8000, num_classes=527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=mel_bins,
                                                 fmin=fmin, fmax=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

    def forward(self, waveform):
        """
        waveform: (batch, samples)
        returns: (batch, num_classes) clipwise sigmoid probabilities
        """
        x = self.spectrogram_extractor(waveform)   # (B, 1, T, n_fft//2+1)
        x = self.logmel_extractor(x)               # (B, 1, T, mel_bins)

        # BN0 normalizes over mel dimension -- transpose so mel is at dim 1
        x = x.transpose(1, 3)   # (B, mel_bins, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (B, 1, T, mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)

        # Global avg+max pool over time and freq dims
        x = torch.mean(x, dim=3)      # (B, 2048, T')
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2                   # (B, 2048)

        x = F.dropout(x, p=0.5, training=False)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=False)
        clipwise = torch.sigmoid(self.fc_audioset(x))
        return clipwise


# --- LOAD MODEL ---

def load_cnn14(ckpt_path, device):
    model = Cnn14(sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, mel_bins=MEL_BINS,
                  fmin=FMIN, fmax=FMAX, num_classes=527)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model']
    # Load ALL keys including spectrogram_extractor and logmel_extractor
    # But our LogmelFilterBank uses 'melW' buffer while checkpoint has 'logmel_extractor.melW'
    # -> strict=False handles mapping automatically
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [INFO] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [INFO] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval().to(device)
    return model


# --- EXTRACT AUDIO ---

def extract_audio(film_path, t_start, duration, sr=16000):
    """Extracts mono audio from film via ffmpeg, returns numpy array."""
    fd, tmp = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    cmd = [
        'ffmpeg', '-y', '-ss', str(t_start), '-t', str(duration),
        '-i', film_path,
        '-ac', '1', '-ar', str(sr), '-f', 'wav', tmp
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:300]}")
    wav, _ = librosa.load(tmp, sr=sr, mono=True)
    os.unlink(tmp)
    return wav


# --- SLIDING WINDOW SCAN ---

def scan_slap(model, waveform, sr, win_s, hop_s, device):
    """
    Slides a window over waveform, returns (timestamps, slap_probs).
    timestamps: seconds relative to start of waveform.
    """
    win_samples = int(win_s * sr)
    hop_samples = int(hop_s * sr)
    n = len(waveform)
    n_windows = max(1, (n - win_samples) // hop_samples + 1)

    timestamps = []
    slap_probs = []

    for i, start in enumerate(range(0, n - win_samples + 1, hop_samples)):
        if i % 100 == 0:
            t_abs = T_START + (start + win_samples / 2) / sr
            print(f"  [{i}/{n_windows}] t={t_abs:.1f}s", flush=True)
        chunk = waveform[start:start + win_samples]
        wav_t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)  # (1, samples)
        with torch.no_grad():
            probs = model(wav_t)   # (1, 527)
        p_slap = probs[0, SLAP_IDX].item()
        t_center = (start + win_samples / 2) / sr
        timestamps.append(t_center)
        slap_probs.append(p_slap)

    return np.array(timestamps), np.array(slap_probs)


# --- SANITY CHECK ---

def verify_model(model, device):
    """Quick check: feed silence and speech-like noise."""
    print("\n[Sanity check]")
    # Silence: expect very low slap probability
    silence = torch.zeros(1, SR).to(device)
    with torch.no_grad():
        p_silence = model(silence)[0, SLAP_IDX].item()
    print(f"  P(slap|silence) = {p_silence:.4f}")

    # White noise: should also be low
    noise = torch.randn(1, SR).to(device) * 0.01
    with torch.no_grad():
        p_noise = model(noise)[0, SLAP_IDX].item()
    print(f"  P(slap|noise)   = {p_noise:.4f}")

    # Check top-3 classes for noise
    noise2 = torch.randn(1, SR).to(device) * 0.1
    with torch.no_grad():
        all_p = model(noise2)[0]
    top3 = torch.topk(all_p, 3)
    print(f"  Top-3 classes for loud noise: {[(i.item(), f'{v.item():.4f}') for i,v in zip(top3.indices, top3.values)]}")


# --- MAIN ---

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading CNN14 from {CKPT}...")
    model = load_cnn14(CKPT, DEVICE)
    print("Model loaded OK.")

    verify_model(model, DEVICE)

    duration = T_END - T_START
    print(f"\nExtracting {duration}s audio from film ({T_START}s-{T_END}s)...")
    wav = extract_audio(FILM, T_START, duration, sr=SR)
    print(f"Audio shape: {wav.shape}, sr={SR}")

    print(f"\nScanning with window={WIN_S}s, hop={HOP_S}s ({int((len(wav)-int(WIN_S*SR))//int(HOP_S*SR)+1)} windows)...")
    ts, probs = scan_slap(model, wav, SR, WIN_S, HOP_S, DEVICE)

    lines = []
    lines.append("=" * 60)
    lines.append(f"Top 30 P(Slap,smack) in [{T_START}s-{T_END}s]  win={WIN_S}s hop={HOP_S}s")
    lines.append("=" * 60)
    idx_sorted = np.argsort(probs)[::-1]
    for rank, i in enumerate(idx_sorted[:30]):
        abs_t = T_START + ts[i]
        mm = int(abs_t // 60)
        ss = abs_t % 60
        bar = '#' * int(probs[i] * 40)
        lines.append(f"  #{rank+1:02d}  {mm:02d}:{ss:05.2f}  P={probs[i]:.4f}  {bar}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Zone critique 15:15-15:35 (915-935s):")
    lines.append("=" * 60)
    for t_rel, p in zip(ts, probs):
        abs_t = T_START + t_rel
        if 915 <= abs_t <= 935:
            mm = int(abs_t // 60)
            ss = abs_t % 60
            bar = '#' * int(p * 40)
            lines.append(f"  {mm:02d}:{ss:05.2f}  P={p:.4f}  {bar}")

    lines.append("")
    lines.append("Stats:")
    lines.append(f"  Max P(slap): {probs.max():.4f} @ abs={T_START + ts[probs.argmax()]:.1f}s")
    lines.append(f"  Mean: {probs.mean():.4f}  Std: {probs.std():.4f}")
    lines.append(f"  P>0.10: {(probs>0.10).sum()}  P>0.05: {(probs>0.05).sum()}  P>0.02: {(probs>0.02).sum()}")

    output = "\n".join(lines)
    print(output)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output + "\n")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
