$ErrorActionPreference = "Stop"

$src = "c:\Users\Welcome\Desktop\PlanGen"
$zip = "c:\Users\Welcome\Desktop\plangen.zip"

# Remove old zip if exists
if (Test-Path $zip) { Remove-Item $zip -Force }

# Files to include (relative to project root, using FORWARD SLASHES for cross-platform compatibility)
$files = @(
    "models.py",
    "colab_training_notebook.py",
    "modules/__init__.py",
    "modules/step4_generate/__init__.py",
    "modules/step4_generate/autoregressive_transformer.py",
    "modules/step4_generate/gnn_encoder.py",
    "modules/step4_generate/training/__init__.py",
    "modules/step4_generate/training/model_trainer.py",
    "modules/step4_generate/training/data_prep.py",
    "sources/__init__.py",
    "sources/rule_loader.py",
    "extracted data/enricher_rules.json",
    "modules/step4_generate/weights/ar_checkpoint_latest.pt",
    "modules/step4_generate/weights/ar_training_history.json",
    "modules/step4_generate/weights/cache/ar_train_b040c629.pkl",
    "modules/step4_generate/weights/cache/ar_val_b040c629.pkl"
)

# Use Python to create the zip with forward-slash paths (Linux-compatible)
# PowerShell's Compress-Archive embeds Windows backslashes which break extraction on Linux/Colab
$pythonScript = @"
import zipfile, os, sys

src   = r'$src'
dst   = r'$zip'
files = $($files | ConvertTo-Json)

print(f'Creating zip: {dst}')
total = 0
with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rel in files:
        abs_path = os.path.join(src, rel.replace('/', os.sep))
        if not os.path.exists(abs_path):
            print(f'  ! MISSING: {rel}')
            continue
        # Archive name MUST use forward slashes so Linux/Colab extracts correctly
        arc_name = 'PlanGen/' + rel
        size_mb  = os.path.getsize(abs_path) / (1024*1024)
        total   += os.path.getsize(abs_path)
        print(f'  + {rel:<70}  {size_mb:>7.1f} MB')
        zf.write(abs_path, arc_name)

zip_mb = os.path.getsize(dst) / (1024*1024)
print(f'')
print(f'Total uncompressed : {total/(1024*1024):.0f} MB')
print(f'plangen.zip size   : {zip_mb:.0f} MB')
print(f'Location           : {dst}')
"@

Write-Host "Building plangen.zip using Python (cross-platform paths)..."
Write-Host ""
python -c $pythonScript

