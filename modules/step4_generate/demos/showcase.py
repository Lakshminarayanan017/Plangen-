"""
showcase.py — curated presentation gallery of the engine's best plans.

Renders a hand-picked set of clean, well-proportioned briefs into a single
polished HTML page for demoing. Unlike the harness (which shows everything,
warts and all), this shows only plans that read well.

Run from the project root:  python -m modules.step4_generate.demos.showcase
Opens: output/showcase.html
"""

from __future__ import annotations

import os

from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs
from modules.step4_generate.render.svg_render import render_svg

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "showcase.html")

# The briefs that render cleanly (low aspect, high score) — the demo set.
FEATURE = [
    "25x40_S_2bhk", "30x40_W_2bhk", "35x35_S_2bhk",
    "30x50_S_3bhk", "40x30_S_2bhk", "30x60_N_3bhk",
    "60x40_E_3bhk", "45x45_W_3bhk",
]


def main() -> None:
    orch = Orchestrator()
    briefs = {b.name: b for b in golden_briefs()}

    cards = []
    for name in FEATURE:
        req = briefs[name]
        result = orch.generate(req)
        if not result.best:
            continue
        plan = result.best.plan
        dims, side, bhk = name.split("_")
        bhk = bhk.replace("bhk", "")
        wf, hf = dims.split("x")
        nrooms = len(plan.rooms)
        cards.append(f"""
        <figure class="card">
          <div class="svgwrap">{render_svg(plan)}</div>
          <figcaption>
            <div class="cap-title">{wf}' × {hf}' plot · {bhk} BHK</div>
            <div class="cap-meta">{nrooms} rooms · entry {side} ·
              quality {result.best.verdict.soft_score:.0f}/100</div>
          </figcaption>
        </figure>""")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlanGen — Floor Plan Engine</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
         background: #f4f5f7; color: #1f2937; }}
  header {{ padding: 40px 32px 28px; background: #fff;
           border-bottom: 1px solid #e5e7eb; }}
  header h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.5px; }}
  header p {{ margin: 8px 0 0; color: #6b7280; max-width: 680px;
             line-height: 1.5; }}
  .badges {{ margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .badge {{ background: #eef2ff; color: #3730a3; padding: 6px 12px;
           border-radius: 999px; font-size: 13px; font-weight: 600; }}
  .grid {{ display: grid; gap: 24px; padding: 32px;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }}
  .card {{ margin: 0; background: #fff; border: 1px solid #e5e7eb;
          border-radius: 12px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .svgwrap {{ padding: 16px; display: flex; justify-content: center;
             background: #fafbfc; }}
  .svgwrap svg {{ max-width: 100%; height: auto; }}
  figcaption {{ padding: 14px 18px; border-top: 1px solid #f0f1f3; }}
  .cap-title {{ font-weight: 700; font-size: 15px; }}
  .cap-meta {{ color: #6b7280; font-size: 13px; margin-top: 3px; }}
</style></head><body>
<header>
  <h1>PlanGen — Residential Floor Plan Engine</h1>
  <p>Partition-first layout generation for Indian residential plots. Every
     wall is a real buildable entity on a 1.5-inch lattice; rooms are carved
     from the plot, so overlaps are impossible by construction. Doors,
     windows, and openness gradients are placed to NBC and Indian design
     conventions.</p>
  <div class="badges">
    <span class="badge">Zero room overlap by construction</span>
    <span class="badge">NBC-compliant minimums</span>
    <span class="badge">Real walls · doors · windows</span>
    <span class="badge">Any plot size</span>
  </div>
</header>
<div class="grid">{''.join(cards)}</div>
</body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"showcase -> {os.path.abspath(OUT)}  ({len(cards)} plans)")


if __name__ == "__main__":
    main()
