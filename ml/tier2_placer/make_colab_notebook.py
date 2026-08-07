"""
make_colab_notebook.py — emit colab_train.ipynb, the resumable training
notebook the user runs on a free Colab GPU.

The notebook: installs deps, mounts Drive (for resumable checkpoints),
expects the repo + prepared data uploaded (or pulled from Drive), runs
tier2_placer.train with engine-metric checkpoint selection, and reports the
merge-gate A/B (trained placer vs PriorProposer on the golden harness).

    python -m tier2_placer.make_colab_notebook   # writes colab_train.ipynb
"""

from __future__ import annotations

import json
import os

_CELLS = [
    ("md", [
        "# PlanGen — Tier-2 Placer Training (M5)\n",
        "\n",
        "Trains the placement transformer on the prepared CubiCasa data and\n",
        "selects the checkpoint by **engine metrics** (not loss). Runtime →\n",
        "GPU. Expects the PlanGen repo and `ml/training/prepared/` present\n",
        "(upload a zip, or keep them on Drive for resumable runs).",
    ]),
    ("code", [
        "# 1. GPU check + deps\n",
        "import torch; print('CUDA:', torch.cuda.is_available(),\n",
        "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n",
        "!pip -q install ijson scipy\n",
    ]),
    ("code", [
        "# 2. Mount Drive (resumable checkpoints survive disconnects)\n",
        "from google.colab import drive; drive.mount('/content/drive')\n",
        "import os\n",
        "CKPT_DIR = '/content/drive/MyDrive/plangen_placer'\n",
        "os.makedirs(CKPT_DIR, exist_ok=True)\n",
    ]),
    ("code", [
        "# 3. Get the code + prepared data.\n",
        "#    Option A: upload plangen.zip (the repo root, with ml/training/prepared/)\n",
        "#    Option B: keep them on Drive and symlink.\n",
        "# from google.colab import files; files.upload()   # plangen.zip\n",
        "# !unzip -q plangen.zip -d /content/\n",
        "import sys; sys.path.insert(0, '/content/PlanGen')   # the repo ROOT\n",
    ]),
    ("code", [
        "# 4. Train (full run). Checkpoints/npz mirrored to Drive.\n",
        "import sys\n",
        "sys.argv = ['train',\n",
        "    '--epochs', '60', '--batch', '16', '--lr', '3e-4',\n",
        "    '--eval-every', '2', '--eval-n', '50',\n",
        "    '--out-dir', CKPT_DIR, '--device', 'cuda']\n",
        "from ml.tier2_placer.train import main as train_main\n",
        "train_main()\n",
    ]),
    ("code", [
        "# 5. Merge gate: trained placer vs PriorProposer on the golden harness.\n",
        "import numpy as np, os\n",
        "from ml.tier2_placer.numpy_infer import NumpyPlacer\n",
        "from ml.tier2_placer.tier2_placer import Tier2Placer\n",
        "from modules.step4_generate.engine.orchestrator import Orchestrator\n",
        "from ml.harness.briefs import golden_briefs\n",
        "placer = Tier2Placer(NumpyPlacer.from_npz(os.path.join(CKPT_DIR,'placer.npz')))\n",
        "def mean_score(orch):\n",
        "    xs=[]\n",
        "    for b in golden_briefs():\n",
        "        r=orch.generate(b); xs.append(r.best.verdict.soft_score if r.best else 0)\n",
        "    return float(np.mean(xs))\n",
        "base = mean_score(Orchestrator())\n",
        "tuned = mean_score(Orchestrator(proposer=placer))\n",
        "print(f'PriorProposer {base:.2f}  vs  Tier2Placer {tuned:.2f}  '\n",
        "      f'({\"MERGE\" if tuned>base else \"stays on branch\"})')\n",
    ]),
    ("code", [
        "# 6. Download the production weights (torch-free NumPy path)\n",
        "from google.colab import files\n",
        "files.download(os.path.join(CKPT_DIR, 'placer.npz'))\n",
    ]),
]


def build_notebook() -> dict:
    cells = []
    for kind, src in _CELLS:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": src})
        else:
            cells.append({"cell_type": "code", "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": src})
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


def main() -> int:
    path = os.path.join(os.path.dirname(__file__), "colab_train.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_notebook(), f, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
