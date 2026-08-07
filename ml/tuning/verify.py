"""
verify.py — did tuning actually change, and improve, the selection?

Because the 20 weights only re-rank a FIXED candidate set, verification is
a re-ranking study on the existing corpus — no regeneration needed:

  • per brief, report the base-config best vs the tuned-config best, and
    whether each matches the human's labeled pick;
  • a side-by-side HTML (base-best | tuned-best) for every brief where the
    pick changed, so the human eyeballs the actual plans that swapped;
  • the honest aggregate: selection agreement with human labels, before vs
    after. This is in-sample (the tuner saw these labels); the trustworthy
    generalization number is TuneResult.tuned_top1_cv from cross-validation.

Nothing here maximizes raw score — raw score is not the objective.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.engine.contracts import EngineConfig
from ml.tuning.corpus import BriefCorpus, Corpus
from ml.tuning.features import WEIGHT_KEYS
from ml.tuning.labeling import Preferences

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
VERIFY_HTML_PATH = os.path.join(RESULTS_DIR, "verify.html")


@dataclass
class BriefVerdict:
    name: str
    base_best_id: str
    tuned_best_id: str
    human_best_id: Optional[str]
    changed: bool
    base_matches_human: Optional[bool]
    tuned_matches_human: Optional[bool]


@dataclass
class VerifyReport:
    per_brief: List[BriefVerdict]
    n_changed: int
    n_labeled: int
    base_agreement: Optional[float]      # in-sample, over labeled briefs
    tuned_agreement: Optional[float]

    def summary_line(self) -> str:
        if self.n_labeled == 0:
            return (f"{self.n_changed}/{len(self.per_brief)} briefs changed "
                    f"their best pick; no human labels to score agreement.")
        return (f"{self.n_changed}/{len(self.per_brief)} briefs changed best; "
                f"selection agreement with human labels "
                f"{self.base_agreement:.0%} → {self.tuned_agreement:.0%} "
                f"(in-sample over {self.n_labeled} labeled briefs).")


def _best_row(brief: BriefCorpus, w: np.ndarray) -> int:
    G = np.array([[c.features.get(k, 0.0) for k in WEIGHT_KEYS]
                  for c in brief.candidates], dtype=float)
    return int(np.argmax(G @ w))


def compare(corpus: Corpus, base: EngineConfig, tuned: EngineConfig,
            prefs: Optional[Preferences] = None) -> VerifyReport:
    """Re-rank every brief under base vs tuned weights and score both against
    the human labels (if provided)."""
    w_base = np.array([getattr(base, k) for k in WEIGHT_KEYS], dtype=float)
    w_tuned = np.array([getattr(tuned, k) for k in WEIGHT_KEYS], dtype=float)

    verdicts: List[BriefVerdict] = []
    base_hits = tuned_hits = labeled = 0
    for brief in corpus.briefs:
        if len(brief.candidates) < 2:
            continue
        b_row = _best_row(brief, w_base)
        t_row = _best_row(brief, w_tuned)
        b_id = brief.candidates[b_row].cand_id
        t_id = brief.candidates[t_row].cand_id

        human = prefs.for_brief(brief.name) if prefs else None
        human_best = human.best if human else None
        b_match = t_match = None
        if human_best:
            labeled += 1
            b_match = (b_id == human_best)
            t_match = (t_id == human_best)
            base_hits += int(b_match)
            tuned_hits += int(t_match)

        verdicts.append(BriefVerdict(
            name=brief.name, base_best_id=b_id, tuned_best_id=t_id,
            human_best_id=human_best, changed=(b_id != t_id),
            base_matches_human=b_match, tuned_matches_human=t_match))

    n_changed = sum(1 for v in verdicts if v.changed)
    return VerifyReport(
        per_brief=verdicts, n_changed=n_changed, n_labeled=labeled,
        base_agreement=(base_hits / labeled) if labeled else None,
        tuned_agreement=(tuned_hits / labeled) if labeled else None)


def weight_diff_table(base: EngineConfig, tuned: EngineConfig) -> List[Dict]:
    """Per-weight before/after with the multiplicative change, biggest
    relative move first."""
    rows = []
    for k in WEIGHT_KEYS:
        b, t = getattr(base, k), getattr(tuned, k)
        rows.append({"weight": k, "base": round(b, 3), "tuned": round(t, 3),
                     "x": round(t / b, 2) if b else float("inf")})
    rows.sort(key=lambda r: abs(r["x"] - 1.0), reverse=True)
    return rows


def render_verify_html(corpus: Corpus, report: VerifyReport,
                       diff_rows: List[Dict],
                       path: str = VERIFY_HTML_PATH) -> str:
    """Side-by-side base-best vs tuned-best for briefs whose pick changed.
    Requires an SVG-bearing corpus (build with render=True)."""
    svg_by_id: Dict[str, str] = {
        c.cand_id: (c.svg or "")
        for b in corpus.briefs for c in b.candidates}

    cards = []
    for v in report.per_brief:
        if not v.changed:
            continue
        tag = ""
        if v.human_best_id:
            if v.tuned_matches_human and not v.base_matches_human:
                tag = "<span class='win'>tuned now matches your pick</span>"
            elif v.base_matches_human and not v.tuned_matches_human:
                tag = "<span class='loss'>tuned moved away from your pick</span>"
            elif v.tuned_matches_human and v.base_matches_human:
                tag = "<span class='win'>both match your pick</span>"
        cards.append(
            f"<section class='brief'><h2>{html.escape(v.name)} {tag}</h2>"
            f"<div class='pair'>"
            f"<figure><figcaption>base best · {html.escape(v.base_best_id)}"
            f"</figcaption>{svg_by_id.get(v.base_best_id,'')}</figure>"
            f"<figure><figcaption>tuned best · {html.escape(v.tuned_best_id)}"
            f"</figcaption>{svg_by_id.get(v.tuned_best_id,'')}</figure>"
            f"</div></section>")

    diff_html = "".join(
        f"<tr><td>{r['weight']}</td><td>{r['base']}</td>"
        f"<td>{r['tuned']}</td><td>{r['x']}×</td></tr>"
        for r in diff_rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = (
        f"<h1>Tuning verification</h1><p class='sum'>"
        f"{html.escape(report.summary_line())}</p>"
        f"<h3>Weight changes (largest relative move first)</h3>"
        f"<table><tr><th>weight</th><th>base</th><th>tuned</th>"
        f"<th>×</th></tr>{diff_html}</table>"
        f"<h3>Briefs whose best pick changed ({report.n_changed})</h3>"
        f"{''.join(cards) or '<p>No pick changed.</p>'}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_VERIFY_PAGE.replace("/*__BODY__*/", body))
    return path


_VERIFY_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlanGen — tuning verification</title><style>
:root{color-scheme:light dark}
body{font-family:"Segoe UI",system-ui,sans-serif;margin:0 auto;max-width:1400px;
     padding:24px;background:#f4f4f6;color:#16181d}
h1{font-size:22px}.sum{font-size:15px;background:#eaf1ff;border:1px solid #cdddff;
     padding:10px 14px;border-radius:8px}
table{border-collapse:collapse;margin:8px 0 24px;font-size:13px}
th,td{border:1px solid #ddd;padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
.brief{background:#fff;border:1px solid #e0e0e6;border-radius:12px;
     padding:14px 18px;margin-bottom:20px}
.brief h2{font-size:16px;margin:0 0 10px}
.pair{display:flex;gap:20px;flex-wrap:wrap}
figure{margin:0}figcaption{font-size:12px;color:#666;margin-bottom:6px}
figure svg{width:340px;height:auto;border:1px solid #eee}
.win{font-size:12px;color:#2e9e5b;font-weight:700;margin-left:8px}
.loss{font-size:12px;color:#d1493f;font-weight:700;margin-left:8px}
@media(prefers-color-scheme:dark){body{background:#14161c;color:#e6e7ea}
.sum{background:#1a2740;border-color:#2a3f66}.brief{background:#1c1f27;
border-color:#2b2f3a}th,td{border-color:#333846}figure svg{border-color:#2b2f3a}}
</style></head><body>/*__BODY__*/</body></html>
"""
