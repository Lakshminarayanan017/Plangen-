"""
render_samples.py — the mandatory eyeball gate (gate d).

"The v1 bbox disaster was visible to the naked eye and nobody looked"
(engine_architecture.md §8.4). So we look: this renders N random prepared
samples as self-contained SVG — the 64×64 boundary mask as the footprint,
each room's 32×32 seed cell as a zone-colored dot labelled with its type and
size class, in canonical generation order. Reviewing this sheet is how we
confirm the prepared TARGETS are sane before a single training step. The
output HTML is committed as the artifact proving the gate ran.

Run:  python -m training.render_samples --n 100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID
from ml.training.paths import PREPARED_DIR
from ml.training.prep_cubicasa import BOUNDARY_GRID

EYEBALL_HTML = os.path.join(PREPARED_DIR, "eyeball_samples.html")

_ZONE_FILL = {"public": "#3b6fd4", "service": "#e08a3b", "private": "#2e9e5b"}
_ABBR = {
    "living_room": "LIV", "drawing_room": "DRW", "dining_room": "DIN",
    "kitchen": "KIT", "master_bedroom": "MBR", "bedroom": "BED",
    "bathroom": "BA", "toilet": "WC", "foyer": "FOY", "hallway": "HAL",
    "store": "STO", "utility": "UTL", "garage": "GAR", "office": "OFF",
    "staircase": "STR",
}


def _abbr(rtype: str) -> str:
    return _ABBR.get(rtype, rtype[:3].upper())


def render_sample_svg(sample: Dict, mask: np.ndarray, px: int = 5) -> str:
    """One prepared sample → SVG (mask footprint + labelled seed dots)."""
    W = H = BOUNDARY_GRID * px
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
          f'height="{H + 34}" viewBox="0 0 {W} {H + 34}">',
          f'<rect width="100%" height="100%" fill="#ffffff"/>']
    # boundary mask footprint — run-length encode each row into one rect per
    # horizontal run (a 64x64 grid becomes ~100 rects, not ~3000)
    for y in range(mask.shape[0]):
        row = mask[y]
        x = 0
        while x < len(row):
            if row[x]:
                x1 = x
                while x1 < len(row) and row[x1]:
                    x1 += 1
                el.append(f'<rect x="{x*px}" y="{y*px}" '
                          f'width="{(x1-x)*px}" height="{px}" fill="#eceff4"/>')
                x = x1
            else:
                x += 1
    # seed dots on the 32-grid, mapped to the 64-grid pixel space
    scale = BOUNDARY_GRID / SEED_GRID          # 2
    for r in sample["rooms"]:
        cx = (r["col"] + 0.5) * scale * px
        cy = (r["row"] + 0.5) * scale * px
        fill = _ZONE_FILL.get(r["zone"], "#888")
        el.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="9" fill="{fill}" '
                  f'fill-opacity="0.85"/>')
        el.append(
            f'<text x="{cx:.0f}" y="{cy+3:.0f}" text-anchor="middle" '
            f'font-family="Segoe UI, sans-serif" font-size="8" '
            f'font-weight="700" fill="#fff">{_abbr(r["rtype"])}'
            f'<tspan font-size="6" font-weight="400">·{r["size_class"]}'
            f'</tspan></text>')
    cap = (f'{sample["plan_id"].split("/")[-1]} · '
           f'{sample["plot_w_ft"]:.0f}×{sample["plot_h_ft"]:.0f}ft · '
           f'ent {sample["entrance_side"]} · {len(sample["rooms"])} rooms')
    el.append(f'<text x="4" y="{H+22}" font-family="Segoe UI, sans-serif" '
              f'font-size="11" fill="#333">{cap}</text>')
    el.append("</svg>")
    return "".join(el)


def render_html(out_dir: str, n: int, seed: int, html_path: str) -> str:
    with open(os.path.join(out_dir, "samples.jsonl"), encoding="utf-8") as f:
        samples: List[Dict] = [json.loads(ln) for ln in f if ln.strip()]
    masks = np.load(os.path.join(out_dir, "masks.npy"))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(samples))[:min(n, len(samples))]
    cards = [f'<div class="c">{render_sample_svg(samples[int(i)], masks[samples[int(i)]["mask_index"]])}</div>'
             for i in pick]
    html = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>PlanGen — prepared data eyeball</title><style>"
        "body{font-family:Segoe UI,sans-serif;background:#fafafa;margin:16px}"
        "h1{font-size:17px}.legend{font-size:13px;margin:6px 0 14px}"
        ".dot{display:inline-block;width:10px;height:10px;border-radius:50%;"
        "vertical-align:middle;margin:0 4px 0 12px}"
        ".grid{display:flex;flex-wrap:wrap;gap:12px}"
        ".c{background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px}"
        "</style>"
        f"<h1>Prepared data eyeball — {len(pick)} random samples</h1>"
        "<div class='legend'>seed cell = room centroid on the 32×32 grid; "
        "label = type·size-class; footprint = 64×64 boundary mask."
        "<span class='dot' style='background:#3b6fd4'></span>public"
        "<span class='dot' style='background:#e08a3b'></span>service"
        "<span class='dot' style='background:#2e9e5b'></span>private</div>"
        f"<div class='grid'>{''.join(cards)}</div>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="training.render_samples")
    p.add_argument("--out-dir", default=PREPARED_DIR)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--html", default=EYEBALL_HTML)
    args = p.parse_args(argv)

    if not os.path.exists(os.path.join(args.out_dir, "samples.jsonl")):
        print(f"no prepared data in {args.out_dir} — run prep_cubicasa first.")
        return 2
    path = render_html(args.out_dir, args.n, args.seed, args.html)
    print(f"eyeball artifact ({args.n} samples) -> {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
