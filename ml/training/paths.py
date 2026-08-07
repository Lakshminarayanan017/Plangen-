"""
paths.py — canonical filesystem locations for the data pipeline.

All bulk corpora live under <project_root>/ml/data/ (gitignored). Prepared
outputs live under ml/training/prepared/ (also gitignored; only the small
manifest + committed eyeball artifact are tracked).

Small distilled knowledge files that the RUNTIME reads (zone_patterns.json
and friends) are not here — they live in <project_root>/data/.
"""

from __future__ import annotations

import os

# ml/training/paths.py -> ml/ is one level up, project root is two
_HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(ML_ROOT)

DATA_DIR = os.path.join(ML_ROOT, "data")

# Source corpora
CUBICASA_NORMALIZED = os.path.join(DATA_DIR, "normalized_extraction.json")
CUBICASA_RAW_DIR = os.path.join(DATA_DIR, "cubicasa")
RPLAN_DIR = os.path.join(DATA_DIR, "rplan")

# Prepared outputs
PREPARED_DIR = os.path.join(_HERE, "prepared")
AUDIT_DIR = os.path.join(_HERE, "audit")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
