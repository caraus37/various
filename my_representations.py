"""
My Vowel Representations
========================
Computational Linguistics - Week 5

Loads your week3homework.wav, extracts the first 30 seconds,
computes MFCC features and Whisper encoder embeddings,
and produces a side-by-side heatmap comparison.

Place week3homework.wav in the same folder as this script and run:
  python3 my_representations.py

Output: my_representations.png

Dependencies:
  pip3 install openai-whisper librosa numpy matplotlib
  (The first run downloads Whisper tiny ~150MB automatically)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WAV_PATH    = os.path.join(SCRIPT_DIR, 'week3homework.wav')
OUT_PATH    = os.path.join(SCRIPT_DIR, 'my_representations.png')
MAX_SECONDS = 30

try:
    import librosa
    import whisper
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip3 install openai-whisper librosa numpy matplotlib")
    sys.exit(1)

if not os.path.isfile(WAV_PATH):
    print(f"Missing: {WAV_PATH}")
    print("Place week3homework.wav in the same folder as this script.")
    sys.exit(1)

# ── Load audio — first 30 seconds only ───────────────────────────
print("Loading week3homework.wav (first 30 seconds)...")
y, sr = librosa.load(WAV_PATH, sr=16000, mono=True,
                     duration=MAX_SECONDS)
actual_duration = len(y) / sr
print(f"  Duration used: {actual_duration:.1f}s")

# Pad to exactly 30 seconds if shorter
if len(y) < MAX_SECONDS * sr:
    y = np.pad(y, (0, MAX_SECONDS * sr - len(y)))
    print(f"  Padded to 30s")

# ── MFCC features ─────────────────────────────────────────────────
print("Computing MFCC features...")
mfcc = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=12,
    n_fft=512, hop_length=160, win_length=400,
)
print(f"  MFCC shape: {mfcc.shape}  "
      f"(12 coefficients × {mfcc.shape[1]} frames)")

# ── Whisper encoder embeddings ────────────────────────────────────
print("Loading Whisper tiny model...")
model = whisper.load_model("tiny")
model.eval()

print("Computing encoder embeddings...")
# Convert numpy array to Whisper's expected format
audio_tensor = torch.from_numpy(y).float()
mel = whisper.log_mel_spectrogram(audio_tensor)   # (80, 3000)

with torch.no_grad():
    embeddings = model.encoder(mel.unsqueeze(0))

embeddings = embeddings.squeeze(0).cpu().numpy()  # (1500, 384)
print(f"  Embedding shape: {embeddings.shape}  "
      f"({embeddings.shape[0]} frames × {embeddings.shape[1]} dimensions)")

# ── Plot ──────────────────────────────────────────────────────────
def normalize(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)

mfcc_norm = normalize(mfcc)
emb_norm  = normalize(embeddings.T)   # (384, 1500)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                constrained_layout=True)

# ── Top: MFCC ─────────────────────────────────────────────────────
im1 = ax1.imshow(mfcc_norm, aspect='auto', origin='lower',
                  cmap='Blues', interpolation='nearest',
                  extent=[0, MAX_SECONDS, 0, 12])
ax1.set_title('MFCC Features  —  12 human-designed coefficients × 30s',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('MFCC coefficient', fontsize=10)
ax1.set_xlabel('Time (seconds)', fontsize=10)
ax1.set_yticks([0, 3, 6, 9, 12])
ax1.set_yticklabels(['C1', 'C3', 'C6', 'C9', 'C12'], fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Normalised amplitude', fontsize=9)
ax1.text(0.99, 0.95, '12 dimensions', transform=ax1.transAxes,
         ha='right', va='top', fontsize=9, fontweight='bold',
         color='#1E3A8A',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#DBEAFE',
                   edgecolor='#1E3A8A', alpha=0.9))

# ── Bottom: Whisper embeddings ────────────────────────────────────
im2 = ax2.imshow(emb_norm, aspect='auto', origin='lower',
                  cmap='Oranges', interpolation='nearest',
                  extent=[0, MAX_SECONDS, 0, 384])
ax2.set_title('Whisper Encoder Embeddings  —  384 learned dimensions × 30s',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Embedding dimension', fontsize=10)
ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_yticks([0, 96, 192, 288, 384])
ax2.set_yticklabels(['0', '96', '192', '288', '384'], fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Normalised activation', fontsize=9)
ax2.text(0.99, 0.95, '384 dimensions', transform=ax2.transAxes,
         ha='right', va='top', fontsize=9, fontweight='bold',
         color='#78350F',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7',
                   edgecolor='#78350F', alpha=0.9))

fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {OUT_PATH}")
plt.close('all')
print("Done. Open my_representations.png.")
