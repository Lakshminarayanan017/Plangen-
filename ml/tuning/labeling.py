"""
labeling.py — the human-preference layer.

Two responsibilities:
  1. Preferences  — the typed label set the tuner consumes, with JSON I/O.
  2. render_labeling_html — a self-contained page the human opens locally to
     record, per brief, which generated plan is best (and, optionally, which
     are unacceptable). It reads an embedded copy of the corpus and exports
     labels.json entirely client-side (no server), so labeling is one file,
     one sitting, no dependencies.

Label schema (labels.json):
    {
      "corpus_seed": 20260716,
      "labels": {
        "30x40_S_2bhk": {"best": "30x40_S_2bhk#2", "bad": ["...#5"]},
        ...
      }
    }
Only `best` is required for a brief to count; `bad` and `ranking` are
optional richer signal. A single `best` yields (best ≻ other) pairs; `bad`
adds (acceptable ≻ bad) pairs; `ranking` (if present) supersedes `best`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ml.tuning.corpus import Corpus

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LABELS_PATH = os.path.join(RESULTS_DIR, "labels.json")
LABELING_HTML_PATH = os.path.join(RESULTS_DIR, "labeling.html")


@dataclass
class BriefPref:
    """One brief's human preference."""
    best: Optional[str] = None
    bad: List[str] = field(default_factory=list)
    ranking: Optional[List[str]] = None

    def is_usable(self) -> bool:
        return bool(self.best or self.ranking or self.bad)


@dataclass
class Preferences:
    by_brief: Dict[str, BriefPref] = field(default_factory=dict)
    corpus_seed: Optional[int] = None

    def for_brief(self, name: str) -> Optional[BriefPref]:
        pref = self.by_brief.get(name)
        return pref if pref and pref.is_usable() else None

    @property
    def n_labeled(self) -> int:
        return sum(1 for p in self.by_brief.values() if p.is_usable())

    # ── JSON I/O ─────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, data: Dict) -> "Preferences":
        by_brief: Dict[str, BriefPref] = {}
        for name, raw in data.get("labels", {}).items():
            by_brief[name] = BriefPref(
                best=raw.get("best"),
                bad=list(raw.get("bad", [])),
                ranking=raw.get("ranking"),
            )
        return cls(by_brief=by_brief, corpus_seed=data.get("corpus_seed"))

    @classmethod
    def load(cls, path: str = LABELS_PATH) -> "Preferences":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


# ── the labeling page ────────────────────────────────────────────────────────

def render_labeling_html(corpus: Corpus, path: str = LABELING_HTML_PATH
                         ) -> str:
    """Write a self-contained labeling page for a corpus built with SVGs.
    Raises if candidates have no rendered SVG (build_corpus(render=True))."""
    briefs_payload = []
    for brief in corpus.labelable():
        cands = []
        for c in brief.candidates:
            if c.svg is None:
                raise ValueError(
                    f"candidate {c.cand_id} has no SVG — rebuild the corpus "
                    "with render=True before generating the labeling UI")
            cands.append({
                "id": c.cand_id,
                "score": c.base_score,
                "rank": c.rank,
                "evidence": _evidence_line(c.breakdown),
                "svg": c.svg,
            })
        briefs_payload.append({"name": brief.name, "candidates": cands})

    data_json = json.dumps({"corpus_seed": corpus.seed,
                            "briefs": briefs_payload})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_PAGE.replace("/*__DATA__*/", data_json))
    return path


# Breakdown keys worth surfacing in the UI, with short labels. Lower is
# better for all of these except wide_openings (more openness = better).
_EVIDENCE = [
    ("area_drift", "drift"), ("worst_aspect", "aspect"),
    ("freespace_violations", "free-viol"), ("hub_mean_dist", "hub-dist"),
    ("wall_efficiency", "wall-eff"), ("privacy_doors", "priv-doors"),
    ("wide_openings", "wides"), ("daylightless", "dark"),
]


def _evidence_line(breakdown: Dict[str, float]) -> str:
    parts = []
    for key, label in _EVIDENCE:
        if key in breakdown:
            val = breakdown[key]
            parts.append(f"{label} {val:g}")
    return " · ".join(parts)


# The page: embedded corpus, click-to-pick-best, optional mark-bad, and a
# client-side JSON export. No external assets, no network.
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlanGen — preference labeling</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: "Segoe UI", system-ui, sans-serif; margin: 0;
         background: #f4f4f6; color: #16181d; }
  header { position: sticky; top: 0; z-index: 10; background: #1f2330;
           color: #fff; padding: 14px 22px; display: flex; gap: 20px;
           align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,.2); }
  header h1 { font-size: 17px; margin: 0; font-weight: 600; }
  header .prog { font-size: 14px; opacity: .85; }
  header .spacer { flex: 1; }
  button { font: inherit; border: 0; border-radius: 7px; padding: 9px 16px;
           cursor: pointer; background: #3b6fd4; color: #fff; font-weight: 600; }
  button.ghost { background: #3a3f52; }
  button:disabled { opacity: .45; cursor: default; }
  .wrap { max-width: 1500px; margin: 0 auto; padding: 22px; }
  .brief { background: #fff; border: 1px solid #e0e0e6; border-radius: 12px;
           padding: 16px 18px; margin-bottom: 22px; }
  .brief > h2 { margin: 0 0 4px; font-size: 16px; }
  .brief > .hint { margin: 0 0 12px; font-size: 13px; color: #6b7280; }
  .brief.done { border-color: #2e9e5b; }
  .grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .cand { border: 2px solid #e2e2e8; border-radius: 10px; padding: 8px;
          background: #fafafb; width: 250px; cursor: pointer;
          transition: border-color .1s, box-shadow .1s; position: relative; }
  .cand:hover { border-color: #9db4e6; }
  .cand.best { border-color: #2e9e5b; box-shadow: 0 0 0 2px #2e9e5b33; }
  .cand.bad { border-color: #d1493f; opacity: .58; }
  .cand svg { width: 100%; height: auto; display: block; }
  .cand .meta { font-size: 12px; color: #444; margin-top: 6px;
                line-height: 1.4; }
  .cand .evi { font-size: 11px; color: #6b7280; margin-top: 3px; }
  .cand .tag { position: absolute; top: 8px; right: 8px; font-size: 11px;
               font-weight: 700; padding: 2px 7px; border-radius: 20px;
               color: #fff; display: none; }
  .cand.best .tag.best { display: block; background: #2e9e5b; }
  .cand.bad  .tag.bad  { display: block; background: #d1493f; }
  .cand .badbtn { margin-top: 6px; font-size: 11px; background: #ececf0;
                  color: #444; padding: 3px 8px; border-radius: 6px; }
  @media (prefers-color-scheme: dark) {
    body { background: #14161c; color: #e6e7ea; }
    .brief { background: #1c1f27; border-color: #2b2f3a; }
    .cand { background: #232733; border-color: #333846; }
    .cand .meta { color: #c4c7cf; } .brief > .hint { color: #9aa0ac; }
    .cand .badbtn { background: #333846; color: #cbd0da; }
  }
</style></head>
<body>
<header>
  <h1>PlanGen · preference labeling</h1>
  <span class="prog" id="prog">0 / 0 briefs labeled</span>
  <span class="spacer"></span>
  <button class="ghost" id="import">Import labels…</button>
  <button id="export" disabled>Export labels.json</button>
  <input type="file" id="file" accept="application/json" hidden>
</header>
<div class="wrap" id="wrap"></div>
<script>
const DATA = /*__DATA__*/;
const labels = {};   // brief -> {best, bad:[]}

function ev(c){ return c.evidence ? c.evidence : ''; }

function render(){
  const wrap = document.getElementById('wrap');
  wrap.innerHTML = '';
  DATA.briefs.forEach(b => {
    const L = labels[b.name] || (labels[b.name] = {best:null, bad:[]});
    const sec = document.createElement('section');
    sec.className = 'brief' + (L.best ? ' done' : '');
    sec.innerHTML = `<h2>${b.name}</h2>
      <p class="hint">Click the plan you'd ship as the best. Optionally mark
      any unacceptable ones "bad" for a stronger signal.</p>`;
    const grid = document.createElement('div');
    grid.className = 'grid';
    b.candidates.forEach(c => {
      const isBest = L.best === c.id, isBad = L.bad.includes(c.id);
      const div = document.createElement('div');
      div.className = 'cand' + (isBest ? ' best' : '') + (isBad ? ' bad' : '');
      div.innerHTML = `<span class="tag best">BEST</span>
        <span class="tag bad">bad</span>
        ${c.svg}
        <div class="meta">engine rank #${c.rank} · score ${c.score}</div>
        <div class="evi">${ev(c)}</div>
        <button class="badbtn">${isBad ? 'un-mark bad' : 'mark bad'}</button>`;
      div.addEventListener('click', e => {
        if (e.target.classList.contains('badbtn')) return;
        L.best = (L.best === c.id) ? null : c.id;
        L.bad = L.bad.filter(x => x !== c.id);
        render();
      });
      div.querySelector('.badbtn').addEventListener('click', e => {
        e.stopPropagation();
        if (L.bad.includes(c.id)) L.bad = L.bad.filter(x => x !== c.id);
        else { L.bad.push(c.id); if (L.best === c.id) L.best = null; }
        render();
      });
      grid.appendChild(div);
    });
    sec.appendChild(grid);
    wrap.appendChild(sec);
  });
  updateProgress();
}

function updateProgress(){
  const done = Object.values(labels).filter(l => l.best).length;
  document.getElementById('prog').textContent =
    `${done} / ${DATA.briefs.length} briefs labeled`;
  document.getElementById('export').disabled = done === 0;
}

document.getElementById('export').addEventListener('click', () => {
  const out = {corpus_seed: DATA.corpus_seed, labels: {}};
  for (const [name, l] of Object.entries(labels)) {
    if (!l.best && (!l.bad || !l.bad.length)) continue;
    out.labels[name] = {};
    if (l.best) out.labels[name].best = l.best;
    if (l.bad && l.bad.length) out.labels[name].bad = l.bad;
  }
  const blob = new Blob([JSON.stringify(out, null, 1)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'labels.json'; a.click();
});

document.getElementById('import').addEventListener('click',
  () => document.getElementById('file').click());
document.getElementById('file').addEventListener('change', ev => {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    try {
      const parsed = JSON.parse(r.result);
      for (const [name, l] of Object.entries(parsed.labels || {}))
        labels[name] = {best: l.best || null, bad: l.bad || []};
      render();
    } catch (e) { alert('Could not parse labels file: ' + e); }
  };
  r.readAsText(f);
});

render();
</script>
</body></html>
"""
