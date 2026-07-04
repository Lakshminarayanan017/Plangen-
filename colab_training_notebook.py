"""
================================================================================
PlanGen — AR Transformer Training Notebook (Google Colab)
================================================================================

Copy each cell below into separate Colab cells and run in order.

Expected Google Drive layout (new account setup):
  My Drive/
    plangen.zip                 ← project source code
    PlanGen_weights/
      ar_checkpoint_latest.pt  ← latest trained checkpoint (epoch 42)
      ar_training_history.json ← full loss history from all accounts

Training ALWAYS resumes from the highest-epoch checkpoint found on Drive.
Checkpoint + history saved PERMANENTLY to Drive/PlanGen_weights/ after every epoch.

Author: PlanGen Team
================================================================================"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Mount Google Drive & Install Dependencies                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 1: Mount Drive & Install Dependencies ---
# Run this cell first. It will prompt for Google Drive authorization.

from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys

# Install PyTorch (Colab usually has it, but ensure correct version for CUDA)
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'torchvision', 'torchaudio',
    'pydantic>=2.0',
    'numpy>=1.24',
])

# Verify GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected — training will be very slow!")
    print("   Go to Runtime → Change runtime type → GPU (T4 or better)")

print(f"   PyTorch: {torch.__version__}")
print(f"   Python:  {sys.version.split()[0]}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Unzip PlanGen Project                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, shutil, zipfile

ZIP_PATH    = '/content/drive/MyDrive/plangen.zip'
PROJECT_DIR = '/content/PlanGen'

# Clean any previous extraction
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
    print(f"🧹 Cleaned previous extraction at {PROJECT_DIR}")

# Verify zip exists
if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"❌ plangen.zip not found at {ZIP_PATH}\n"
        f"   Upload plangen.zip to your Google Drive root folder first."
    )

print(f"📦 Extracting {ZIP_PATH} ...")
zip_size = os.path.getsize(ZIP_PATH) / (1024**2)
print(f"   Zip size: {zip_size:.0f} MB")

# ── Smart extraction: handle Windows backslash paths ─────────────────────
# PowerShell's Compress-Archive embeds backslashes (e.g. PlanGen\modules\...).
# Python's zipfile on Linux treats backslashes as literal filename characters,
# NOT directory separators — so extractall() dumps everything flat.
# We fix this by manually extracting each entry with corrected paths.
os.makedirs(PROJECT_DIR, exist_ok=True)
extracted_count = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for entry in zf.infolist():
        # Normalise: convert ALL backslashes to forward slashes
        norm_name = entry.filename.replace('\\', '/')

        # Strip the leading "PlanGen/" prefix so files land in PROJECT_DIR
        for prefix in ('PlanGen/', 'plangen/'):
            if norm_name.lower().startswith(prefix):
                norm_name = norm_name[len(prefix):]
                break

        # Skip directory-only entries and empty names
        if not norm_name or norm_name.endswith('/'):
            continue

        dest_path = os.path.join(PROJECT_DIR, norm_name)
        dest_dir  = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        with zf.open(entry) as src, open(dest_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

        extracted_count += 1

print(f"   Extracted {extracted_count} files")
print(f"✅ Project extracted to {PROJECT_DIR}")

# ── Patch: copy any files uploaded separately to Drive ───────────────────
# Upload missing files directly to My Drive root and they'll be copied here.
DRIVE_ROOT = '/content/drive/MyDrive'
DRIVE_PATCHES = {
    'model_trainer.py':              f'{PROJECT_DIR}/modules/step4_generate/training/model_trainer.py',
    'autoregressive_transformer.py': f'{PROJECT_DIR}/modules/step4_generate/autoregressive_transformer.py',
    'gnn_encoder.py':                f'{PROJECT_DIR}/modules/step4_generate/gnn_encoder.py',
    'data_prep.py':                  f'{PROJECT_DIR}/modules/step4_generate/training/data_prep.py',
    'models.py':                     f'{PROJECT_DIR}/models.py',
}
for drive_filename, dest_path in DRIVE_PATCHES.items():
    drive_src = os.path.join(DRIVE_ROOT, drive_filename)
    if not os.path.exists(dest_path) and os.path.exists(drive_src):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(drive_src, dest_path)
        print(f"📋 Patched from Drive: {drive_filename}")

# ── Verify critical files exist ───────────────────────────────────────────
all_ok = True
for check_path in [
    'modules/step4_generate/training/model_trainer.py',
    'modules/step4_generate/autoregressive_transformer.py',
    'modules/step4_generate/gnn_encoder.py',
    'modules/step4_generate/weights/cache',
]:
    full   = os.path.join(PROJECT_DIR, check_path)
    exists = os.path.exists(full)
    print(f"   {'✅' if exists else '❌'} {check_path}")
    if not exists:
        all_ok = False

if not all_ok:
    # Show what actually landed in PROJECT_DIR to help debug
    print("\n📂 Files found in /content/PlanGen (top-level):")
    for item in sorted(os.listdir(PROJECT_DIR))[:20]:
        print(f"   {item}")
    raise FileNotFoundError(
        "Some critical files are still missing.\n"
        "Upload any missing .py files individually to My Drive root and re-run this cell."
    )

print("\n✅ All critical files present — ready to proceed!")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Verify Checkpoint & Data Integrity                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 3: Verify Checkpoint & Training Data ---
# Validates the checkpoint loads correctly and training data is present.
#
# Checkpoint search order (picks the HIGHEST epoch number found):
#   1. My Drive/PlanGen_weights/ar_checkpoint_latest.pt  ← your uploaded folder
#   2. Local zip-extracted file  (usually an old epoch from a previous run)
#   3. My Drive/ root  (fallback for loose files dropped without a folder)

import os, sys, json, shutil
import torch

PROJECT_DIR  = '/content/PlanGen'
WEIGHTS_DIR  = os.path.join(PROJECT_DIR, 'modules', 'step4_generate', 'weights')
CACHE_DIR    = os.path.join(WEIGHTS_DIR, 'cache')
CKPT_PATH    = os.path.join(WEIGHTS_DIR, 'ar_checkpoint_latest.pt')
HISTORY_PATH = os.path.join(WEIGHTS_DIR, 'ar_training_history.json')
DRIVE_ROOT   = '/content/drive/MyDrive'
DRIVE_BACKUP = '/content/drive/MyDrive/PlanGen_weights'

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Helper: read the epoch stored inside a checkpoint ────────────────────
def _peek_epoch(path):
    """Returns epoch int from checkpoint, or -1 on any failure."""
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
        return int(c.get('epoch', -1))
    except Exception:
        return -1

# ── Collect ALL candidate checkpoint files from every location ────────────
# KEY FIX: we do NOT guard with 'if not exists' — we ALWAYS check Drive
# so that a freshly-uploaded Drive file beats the old zip-extracted one.
print('\n🔎 Scanning all checkpoint sources (comparing epoch numbers)...')

candidates = []  # list of (label, ckpt_path, hist_path_or_None)

# 1. Zip-extracted local file (may be old)
if os.path.exists(CKPT_PATH):
    candidates.append(('zip-local', CKPT_PATH, HISTORY_PATH if os.path.exists(HISTORY_PATH) else None))

# 2. Drive PlanGen_weights/ folder  ← PRIMARY source for new-account setup
_db_ckpt = os.path.join(DRIVE_BACKUP, 'ar_checkpoint_latest.pt')
_db_hist = os.path.join(DRIVE_BACKUP, 'ar_training_history.json')
if os.path.exists(_db_ckpt):
    candidates.append(('Drive/PlanGen_weights/', _db_ckpt, _db_hist if os.path.exists(_db_hist) else None))
else:
    print(f'⚠️  PlanGen_weights/ folder not found at {DRIVE_BACKUP}')
    print('   Make sure the folder was uploaded to My Drive before running this cell.')

# 3. Drive ROOT fallback — loose files without a subfolder
_ROOT_CKPT_NAMES = ['ar_checkpoint_latest.pt', 'ar_checkpoint_latest (1).pt']
_ROOT_HIST_NAMES = ['ar_training_history.json', 'ar_training_history (1).json']
for _cn in _ROOT_CKPT_NAMES:
    _src = os.path.join(DRIVE_ROOT, _cn)
    if os.path.exists(_src):
        _hpath = next((os.path.join(DRIVE_ROOT, h) for h in _ROOT_HIST_NAMES
                       if os.path.exists(os.path.join(DRIVE_ROOT, h))), None)
        candidates.append((f'Drive root (loose file: {_cn})', _src, _hpath))
        break

# ── Compare epochs and pick the NEWEST one ───────────────────────────────
best_epoch = -1
best_label = None
best_ckpt  = None
best_hist  = None

for label, cpath, hpath in candidates:
    ep   = _peek_epoch(cpath)
    size = os.path.getsize(cpath) / (1024**2)
    marker = ''
    if ep > best_epoch:
        best_epoch = ep
        best_label = label
        best_ckpt  = cpath
        best_hist  = hpath
        marker = '  <-- NEWEST'
    print(f'   [{label}]  completed epoch={ep+1 if ep>=0 else "?"}  ({size:.0f} MB){marker}')

if best_ckpt is None:
    raise FileNotFoundError(
        '\n❌ No checkpoint found anywhere!\n'
        f'   Expected location : {_db_ckpt}\n'
        f'   Drive root checked: {DRIVE_ROOT}/ar_checkpoint_latest.pt\n'
        '   Make sure PlanGen_weights/ folder is in My Drive and contains\n'
        '   ar_checkpoint_latest.pt, then re-run this cell.'
    )

# ── Install the best checkpoint as the active working copy ───────────────
if best_ckpt != CKPT_PATH:
    shutil.copy2(best_ckpt, CKPT_PATH)
    print(f'\n✅ Installed newest checkpoint from [{best_label}] (epoch {best_epoch+1})')
    print(f'   (Overwrote older zip-extracted version)')
else:
    print(f'\n✅ Local zip checkpoint is already the newest (epoch {best_epoch+1})')

if best_hist and best_hist != HISTORY_PATH:
    shutil.copy2(best_hist, HISTORY_PATH)
    print(f'   History JSON also updated from [{best_label}]')
elif not os.path.exists(HISTORY_PATH):
    print('   No history JSON found — will start fresh history tracking')

# ── Validate checkpoint ──────────────────────────────────────────────────
print("\n🔍 Validating checkpoint...")
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        f"❌ Checkpoint not found at {CKPT_PATH}\n"
        f"   Ensure ar_checkpoint_latest.pt is in the zip file."
    )

ckpt_size = os.path.getsize(CKPT_PATH) / (1024**2)
print(f"   Checkpoint size: {ckpt_size:.0f} MB")

# Load and validate structure
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
required_keys = {'epoch', 'gnn', 'ar_model', 'optimiser', 'best_val_loss'}
present_keys  = set(ckpt.keys())
missing_keys  = required_keys - present_keys

if missing_keys:
    raise ValueError(f"❌ Checkpoint missing keys: {missing_keys}")

epoch_completed = ckpt['epoch']
best_val        = ckpt['best_val_loss']
n_gnn_params    = sum(v.numel() for v in ckpt['gnn'].values())
n_ar_params     = sum(v.numel() for v in ckpt['ar_model'].values())

print(f"   ✅ Checkpoint valid")
print(f"   Last completed epoch : {epoch_completed + 1}")
print(f"   Will resume at epoch : {epoch_completed + 2}")
print(f"   Best val loss so far : {best_val:.4f}")
print(f"   GNN parameters       : {n_gnn_params:,}")
print(f"   AR Transformer params: {n_ar_params:,}")
print(f"   Total parameters     : {(n_gnn_params + n_ar_params)/1e6:.1f}M")

del ckpt  # free memory

# ── Validate training data ───────────────────────────────────────────────
print("\n🔍 Validating training data cache...")
cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
for cf in sorted(cache_files):
    size = os.path.getsize(os.path.join(CACHE_DIR, cf)) / (1024**2)
    print(f"   ✅ {cf} ({size:.0f} MB)")

if not cache_files:
    raise FileNotFoundError(
        f"❌ No .pkl cache files in {CACHE_DIR}\n"
        f"   Training data is missing from the zip."
    )

# ── Validate training history ────────────────────────────────────────────
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    n_epochs = len(history.get('train_loss', []))
    print(f"\n📊 Training history: {n_epochs} epochs completed")
    if history.get('train_loss'):
        print(f"   First train loss : {history['train_loss'][0]:.4f}")
        print(f"   Last train loss  : {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        print(f"   First val loss   : {history['val_loss'][0]:.4f}")
        print(f"   Last val loss    : {history['val_loss'][-1]:.4f}")
    print(f"   Best val loss    : {history.get('best_val_loss', 'N/A')}")
else:
    print("\n⚠️  No training history found — will start fresh history tracking")

print("\n✅ All validations passed — ready to train!")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Configure Training Hyperparameters                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 4: Configure Hyperparameters ---
# Adjust these values based on your GPU and training needs.

# ─── Training Target ─────────────────────────────────────────────────────
# Currently at epoch 42 — set target above that to continue training.
# The trainer reads the checkpoint epoch and only trains the remaining epochs.
TARGET_EPOCHS   = 100      # Train until epoch 100 total (42 done → 58 more)
BATCH_SIZE      = 32       # 32 for T4 (16GB), 16 if OOM, 64 for A100

# ─── Optimiser ───────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-4     # AdamW learning rate
WEIGHT_DECAY    = 1e-5     # L2 regularisation
GRAD_CLIP       = 1.0      # Gradient clipping max norm
WARMUP_STEPS    = 500      # Linear warmup steps

# ─── Architecture (MUST match checkpoint — do NOT change) ────────────────
D_MODEL         = 512      # Transformer hidden dimension
N_LAYERS        = 12       # Transformer layers
N_HEADS         = 8        # Attention heads
D_FF            = 2048     # Feed-forward hidden dimension
GNN_DIM         = 256      # GNN output dimension

# ─── Paths ───────────────────────────────────────────────────────────────
PROJECT_DIR     = '/content/PlanGen'
WEIGHTS_DIR     = f'{PROJECT_DIR}/modules/step4_generate/weights'
CACHE_DIR       = f'{WEIGHTS_DIR}/cache'

# ─── Drive backup directory (created automatically) ──────────────────────
DRIVE_BACKUP_DIR = '/content/drive/MyDrive/PlanGen_weights'

print("=" * 60)
print("  Training Configuration")
print("=" * 60)
print(f"  Target epochs  : {TARGET_EPOCHS}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Learning rate  : {LEARNING_RATE}")
print(f"  Model          : d={D_MODEL}, L={N_LAYERS}, h={N_HEADS}")
print(f"  Drive backup   : {DRIVE_BACKUP_DIR}")
print("=" * 60)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Run Training (Resumes from Checkpoint)                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 5: Run Training ---
# This is the main training cell. It will:
#   1. Load data from cache
#   2. Build models + load checkpoint
#   3. Train from the last saved epoch to TARGET_EPOCHS
#   4. Save checkpoint to local disk + Google Drive after EVERY epoch
#   5. Print loss metrics per epoch
#
# If Colab disconnects, just re-run Cells 1-4 then this cell.
# Training will resume from the last Drive-backed checkpoint.

import os, sys, time, json, shutil, logging

# ── Setup logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

# ── Add project to Python path ──────────────────────────────────────────
PROJECT_DIR = '/content/PlanGen'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Paths ────────────────────────────────────────────────────────────────
WEIGHTS_DIR      = f'{PROJECT_DIR}/modules/step4_generate/weights'
CKPT_PATH        = f'{WEIGHTS_DIR}/ar_checkpoint_latest.pt'
HISTORY_PATH     = f'{WEIGHTS_DIR}/ar_training_history.json'
DRIVE_BACKUP_DIR = '/content/drive/MyDrive/PlanGen_weights'
DRIVE_CKPT       = f'{DRIVE_BACKUP_DIR}/ar_checkpoint_latest.pt'
DRIVE_HISTORY    = f'{DRIVE_BACKUP_DIR}/ar_training_history.json'
DRIVE_ROOT       = '/content/drive/MyDrive'

os.makedirs(WEIGHTS_DIR,      exist_ok=True)
os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)

# ── Restore the HIGHEST-EPOCH checkpoint (same logic as Cell 3) ───────────
# We ALWAYS compare epochs from all sources — never skip Drive just because
# a local (zip-extracted, stale) file already exists.
def _peek_epoch_c5(path):
    """Returns epoch int from checkpoint, or -1 on any failure."""
    try:
        import torch as _t
        c = _t.load(path, map_location='cpu', weights_only=False)
        return int(c.get('epoch', -1))
    except Exception:
        return -1

print('\n🔎 Scanning checkpoint sources...')
_c5_candidates = []

# Source A: zip-extracted local (often stale)
if os.path.exists(CKPT_PATH):
    _c5_candidates.append(('zip-local', CKPT_PATH,
                           HISTORY_PATH if os.path.exists(HISTORY_PATH) else None))

# Source B: Drive/PlanGen_weights/  ← your uploaded folder (epoch 42)
if os.path.exists(DRIVE_CKPT):
    _c5_candidates.append(('Drive/PlanGen_weights/', DRIVE_CKPT,
                           DRIVE_HISTORY if os.path.exists(DRIVE_HISTORY) else None))

# Source C: Drive root loose files (fallback)
for _cn in ['ar_checkpoint_latest.pt', 'ar_checkpoint_latest (1).pt']:
    _rsrc = os.path.join(DRIVE_ROOT, _cn)
    if os.path.exists(_rsrc):
        _rh = next((os.path.join(DRIVE_ROOT, h)
                    for h in ['ar_training_history.json', 'ar_training_history (1).json']
                    if os.path.exists(os.path.join(DRIVE_ROOT, h))), None)
        _c5_candidates.append((f'Drive root ({_cn})', _rsrc, _rh))
        break

_c5_best_ep, _c5_best_src, _c5_best_ckpt, _c5_best_hist = -1, None, None, None
for _lbl, _cp, _hp in _c5_candidates:
    _ep   = _peek_epoch_c5(_cp)
    _size = os.path.getsize(_cp) / (1024**2)
    _mark = '  <-- WILL USE' if _ep > _c5_best_ep else ''
    if _ep > _c5_best_ep:
        _c5_best_ep, _c5_best_src  = _ep, _lbl
        _c5_best_ckpt, _c5_best_hist = _cp, _hp
    print(f'   [{_lbl}]  epoch={_ep+1 if _ep>=0 else "?"}  ({_size:.0f} MB){_mark}')

if _c5_best_ckpt is None:
    raise FileNotFoundError(
        '\n❌ No checkpoint found!\n'
        f'   Upload PlanGen_weights/ folder to My Drive and re-run.'
    )

if _c5_best_ckpt != CKPT_PATH:
    print(f'\n♻️  Installing checkpoint from [{_c5_best_src}] (epoch {_c5_best_ep+1})')
    shutil.copy2(_c5_best_ckpt, CKPT_PATH)
    if _c5_best_hist:
        shutil.copy2(_c5_best_hist, HISTORY_PATH)
    print('✅ Checkpoint ready')
else:
    print(f'\n✅ Local checkpoint is already the newest (epoch {_c5_best_ep+1})')

# ── Run training ─────────────────────────────────────────────────────────
from modules.step4_generate.training.model_trainer import ARTrainer

print("\n" + "=" * 60)
print("  🚀 Starting Training")
print("=" * 60)

trainer = ARTrainer(
    cache_dir    = f'{WEIGHTS_DIR}/cache',
    out_dir      = WEIGHTS_DIR,
    d_model      = 512,     # Must match checkpoint
    n_heads      = 8,       # Must match checkpoint
    n_layers     = 12,      # Must match checkpoint
    d_ff         = 2048,    # Must match checkpoint
    lr           = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
    batch_size   = BATCH_SIZE,
    grad_clip    = GRAD_CLIP,
    warmup_steps = WARMUP_STEPS,
    seed         = 42,
)

t0 = time.time()

history = trainer.train(
    epochs       = TARGET_EPOCHS,
    force_rebuild = False,    # Use existing cache (MUCH faster)
    export_numpy = False,     # Export separately in Cell 6
)

elapsed = time.time() - t0
hours   = int(elapsed // 3600)
mins    = int((elapsed % 3600) // 60)

print("\n" + "=" * 60)
print("  ✅ Training Complete!")
print("=" * 60)
print(f"  Total time     : {hours}h {mins}m")
print(f"  Epochs trained : {len(history['train_loss'])}")
print(f"  Best val loss  : {history['best_val_loss']:.4f}")
if history['train_loss']:
    print(f"  Final train    : {history['train_loss'][-1]:.4f}")
if history['val_loss']:
    print(f"  Final val      : {history['val_loss'][-1]:.4f}")

# ── Final Drive backup (redundant but safe) ──────────────────────────────
try:
    shutil.copy2(CKPT_PATH, DRIVE_CKPT)
    shutil.copy2(HISTORY_PATH, DRIVE_HISTORY)
    print(f"\n💾 Final backup saved to: {DRIVE_BACKUP_DIR}")
except Exception as e:
    print(f"\n⚠️  Final Drive backup failed: {e}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Export NumPy Weights for Deployment                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 6: Export Deployment Weights ---
# Converts the PyTorch checkpoint to NumPy .npz files for production
# inference (no PyTorch required at runtime).

import os, sys, shutil
import torch

PROJECT_DIR = '/content/PlanGen'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

WEIGHTS_DIR      = f'{PROJECT_DIR}/modules/step4_generate/weights'
CKPT_PATH        = f'{WEIGHTS_DIR}/ar_checkpoint_latest.pt'
DRIVE_BACKUP_DIR = '/content/drive/MyDrive/PlanGen_weights'

print("🔄 Exporting NumPy weights for deployment...\n")

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
print(f"   Loaded checkpoint (epoch {ckpt['epoch'] + 1})")

# Build models and load state dicts
from modules.step4_generate.gnn_encoder import GNNEncoder
from modules.step4_generate.autoregressive_transformer import LayoutTransformer

gnn      = GNNEncoder()
ar_model = LayoutTransformer()

gnn.load_state_dict(ckpt['gnn'])
ar_model.load_state_dict(ckpt['ar_model'])
print("   ✅ Models loaded")

# Export
gnn_npz_path = os.path.join(WEIGHTS_DIR, 'gnn_encoder.npz')
ar_npz_path  = os.path.join(WEIGHTS_DIR, 'ar_transformer.npz')

gnn.save_numpy_weights(gnn_npz_path)
ar_model.save_numpy_weights(ar_npz_path)

# Copy to Drive
for f in [gnn_npz_path, ar_npz_path]:
    fname = os.path.basename(f)
    dest  = os.path.join(DRIVE_BACKUP_DIR, fname)
    shutil.copy2(f, dest)
    size  = os.path.getsize(f) / (1024**2)
    print(f"   💾 {fname} -> Drive ({size:.1f} MB)")

del ckpt, gnn, ar_model
torch.cuda.empty_cache()

print("\n✅ Deployment weights exported and saved to Drive!")
print(f"   {DRIVE_BACKUP_DIR}/gnn_encoder.npz")
print(f"   {DRIVE_BACKUP_DIR}/ar_transformer.npz")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Verify All Outputs on Drive                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- CELL 7: Final Verification ---
# Lists all saved files on Drive and validates integrity.

import os, json

DRIVE_DIR = '/content/drive/MyDrive/PlanGen_weights'

print("=" * 60)
print("  📁 PlanGen Weights on Google Drive")
print("=" * 60)

if not os.path.exists(DRIVE_DIR):
    print(f"  ❌ Directory not found: {DRIVE_DIR}")
else:
    total_size = 0
    for fname in sorted(os.listdir(DRIVE_DIR)):
        fpath = os.path.join(DRIVE_DIR, fname)
        size  = os.path.getsize(fpath)
        total_size += size
        icon = "✅"
        print(f"  {icon}  {fname:<40}  {size/(1024**2):>8.1f} MB")

    print(f"\n  {'Total':<46}  {total_size/(1024**2):>8.1f} MB")

    # Validate history
    hist_path = os.path.join(DRIVE_DIR, 'ar_training_history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        print(f"\n  📊 Training Summary:")
        print(f"     Epochs completed : {len(h.get('train_loss', []))}")
        print(f"     Best val loss    : {h.get('best_val_loss', 'N/A')}")
        if h.get('train_loss'):
            print(f"     Loss progression : {h['train_loss'][0]:.4f} -> {h['train_loss'][-1]:.4f}")

print("\n" + "=" * 60)
print("  🎉 All done! Your trained model is safely on Google Drive.")
print("  Download these files to your project's weights/ directory:")
print("    - ar_checkpoint_latest.pt   (for resuming training)")
print("    - ar_transformer.npz        (for deployment inference)")
print("    - gnn_encoder.npz           (for deployment inference)")
print("=" * 60)
