"""
Encode les frames 1fps deja extraites vers un cache .pt
Batch=8, empty_cache() entre chaque batch = stable sur 6GB
"""
import os, sys, torch, torch.nn.functional as F
import open_clip
from PIL import Image

FRAMES_DIR = r"G:\projets_ia\slaps\pieta_v3_sweep_visual_sweep_cache_frames"
CACHE_OUT  = r"G:\projets_ia\slaps\pieta_v3_sweep_visual_sweep_cache.pt"
BATCH      = 8
DEVICE     = "cuda"

print(f"Chargement DFN5B...", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14-378-quickgelu', pretrained='dfn5b', device=DEVICE)
model.eval()
print("OK", flush=True)

frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg'))
n = len(frame_files)
fps_sweep = 1.0
frame_times = [i / fps_sweep for i in range(n)]
print(f"{n} frames a encoder (batch={BATCH})...", flush=True)

all_embs = []
buf_fnames = []

for fi, fname in enumerate(frame_files):
    buf_fnames.append(os.path.join(FRAMES_DIR, fname))

    if len(buf_fnames) == BATCH or fi == n - 1:
        imgs = []
        for fp in buf_fnames:
            try:
                imgs.append(preprocess(Image.open(fp).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 378, 378))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            embs = F.normalize(model.encode_image(batch), dim=-1)
        all_embs.append(embs.cpu())
        del batch, imgs, embs
        torch.cuda.empty_cache()
        buf_fnames = []

        if (fi + 1) % 320 == 0 or fi == n - 1:
            print(f"  {fi+1}/{n}...", flush=True)

embs_all = torch.cat(all_embs, dim=0)
print(f"Shape finale: {embs_all.shape}", flush=True)
torch.save({"embs": embs_all, "frame_times": frame_times}, CACHE_OUT)
print(f"Cache sauvegarde: {CACHE_OUT}", flush=True)
