"""
PlanGen API Server
==================
FastAPI REST wrapper around the PlanGen ML pipeline.
Serves the frontend static files and exposes all pipeline endpoints.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Pipeline imports ─────────────────────────────────────────────
from models import BuildingRequirements
from modules.step1_parse.parser import Module1Pipeline
from modules.step2_match.matcher import PatternMatcher
from modules.step3_enrich.enricher import Enricher
from api.engine_bridge import generate_layout

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PlanGen_API")

# ── Output directories ───────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Session store (in-memory for now) ────────────────────────────
sessions: Dict[str, dict] = {}
pipeline_status: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app):
    """Print the subsystem self-check at boot.

    Every knowledge source in this system fails soft, which means a silent
    startup is indistinguishable from a healthy one. It should never again be
    possible to run this server for months without noticing that step 2 is
    retrieving nothing.
    """
    from modules.diagnostics import format_report, self_check
    report = self_check()
    for line in format_report(report).splitlines():
        logger.info(line)
    if not report["healthy"]:
        logger.warning(
            "Running DEGRADED — %d subsystem(s) missing, %d on fallbacks. "
            "GET /api/v1/diagnostics for detail.",
            report["counts"]["missing"], report["counts"]["degraded"],
        )
    yield


# ── Request/Response models ──────────────────────────────────────
class ParseTextRequest(BaseModel):
    session_id: str
    text: str

class ParseAnswerRequest(BaseModel):
    session_id: str
    answer: str

class PipelineRunRequest(BaseModel):
    session_id: str
    options: Optional[dict] = None

class RegenerateRequest(BaseModel):
    session_id: str
    options: Optional[dict] = None

class ChooseRequest(BaseModel):
    session_id: str
    rank: int
    note: str = ""

class EditRequest(BaseModel):
    session_id: str
    text: str
    preview: bool = False        # parse and explain without generating

class SessionResponse(BaseModel):
    session_id: str
    created_at: str


# ── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="PlanGen API",
    version="2.1.0",
    description="AI-Powered Floor Plan Generator",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: get or create session ────────────────────────────────
def _signature_of(layout):
    """The ground floor's room positions, kept so a later edit can preserve
    them. Ground floor only: on a multi-floor building each floor is chosen
    against the one below (VRT-001 pins the stair), so the floors are not
    independently swappable and a per-floor signature would imply they are.
    """
    try:
        from modules.step4_generate.engine import continuity as cont
        if not layout.floors:
            return None
        return cont.signature_of_floor(layout.floors[0])
    except Exception:                      # never cost a run its result
        return None


def _signature_from_notes(engine_notes):
    """The signature the engine ranked with, if it reported one."""
    try:
        from modules.step4_generate.engine import continuity as cont
        blob = (engine_notes or {}).get("signature")
        return cont.signature_from_dict(json.loads(blob)) if blob else None
    except Exception:                      # never cost a run its result
        return None


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    return sessions[session_id]


# ══════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/v1/sessions")
def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "id": sid,
        "created_at": datetime.now().isoformat(),
        "parser": Module1Pipeline(),
        "step1_result": None,
        "requirements": None,
        "runs": {},
    }
    logger.info("Session created: %s", sid)
    return {"session_id": sid, "created_at": sessions[sid]["created_at"]}


@app.delete("/api/v1/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}


# ══════════════════════════════════════════════════════════════════
# STEP 1 — PARSE ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/v1/parse/text")
def parse_text(req: ParseTextRequest):
    session = _get_session(req.session_id)
    parser: Module1Pipeline = session["parser"]

    # handle_followup MERGES into the running conversation (and internally
    # falls back to a fresh execute() on the first message). Calling execute()
    # here unconditionally would wipe all previously gathered requirements
    # every time the user types mid-conversation.
    result = parser.handle_followup(req.text)
    session["step1_result"] = result

    if result.get("status") == "success" and result.get("data"):
        session["requirements"] = result["data"]

    return result


@app.get("/api/v1/parse/next-question")
def get_next_question(session_id: str):
    session = _get_session(session_id)
    parser: Module1Pipeline = session["parser"]

    try:
        action = parser.get_next_interactive_action()
        return action
    except Exception as e:
        return {"action": "complete", "message": str(e)}


@app.post("/api/v1/parse/answer")
def parse_answer(req: ParseAnswerRequest):
    session = _get_session(req.session_id)
    parser: Module1Pipeline = session["parser"]

    result = parser.process_interactive_answer(req.answer)
    session["step1_result"] = result

    if result.get("status") == "success" and result.get("data"):
        session["requirements"] = result["data"]

    return result


# ══════════════════════════════════════════════════════════════════
# STEP 2-5 — PIPELINE RUN
# ══════════════════════════════════════════════════════════════════

def _run_pipeline_task(run_id: str, session: dict, opts: dict, run_dir: Path):
    """Steps 2-5. `opts` may carry:
        requirements  a brief EDITED by plan_edit, used instead of the
                      session's own — so an edit never mutates the original
        overrides     resize / re-orient instructions applied AFTER step 3,
                      because the enricher would otherwise re-derive them
                      from its own rules and discard what the user asked
        seed_from     hold the engine seed of a previous run, so an edited
                      plan is the SAME house with the change, not a new one
    """
    steps_log = []
    
    def update_status(step, label, status="running", msg="", p_log=None):
        if p_log:
            steps_log.append(p_log)
        ts = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{ts}] {label}: {msg}"
        pipeline_status[run_id]["step"] = step
        pipeline_status[run_id]["logs"].append(log_line)

    try:
        # ── Build requirements model ─────────────────────────────
        update_status(1, "SYS", msg="Starting PlanGen pipeline")
        reqs = BuildingRequirements.model_validate(
            opts.get("requirements") or session["requirements"])
        with open(run_dir / "step1_final.json", "w") as f:
            json.dump(session["requirements"], f, indent=2)
        update_status(1, "PARSE", msg=f"Blueprint boundaries detected. Processing requirements...", p_log={"step": 1, "status": "complete", "label": "PARSE"})

        # ── STEP 2: MATCH ────────────────────────────────────────
        matcher = PatternMatcher()
        bundle = matcher.fetch_patterns(reqs)
        with open(run_dir / "step2_knowledge_bundle.json", "w") as f:
            json.dump(bundle.model_dump(), f, indent=2, default=str)

        step2_summary = {
            "matched_plans": len(bundle.matched_plans) if bundle.matched_plans else 0,
            "match_quality_score": round(bundle.match_quality_score, 3),
        }
        update_status(2, "MATCH", msg=f"Correlated structural footprint against {step2_summary['matched_plans']} reference models.", p_log={"step": 2, "status": "complete", "label": "MATCH", "summary": step2_summary})

        # ── STEP 3: ENRICH ───────────────────────────────────────
        update_status(3, "ENRICH", msg="Injecting NBC compliance constraints. Calculating load-bearing distribution...")
        use_gemini = opts.get("use_gemini_enricher", True)
        enricher = Enricher(use_gemini=use_gemini)
        enriched = enricher.enrich(reqs, bundle)

        with open(run_dir / "step3_enriched_plan.json", "w") as f:
            json.dump(enriched.model_dump(), f, indent=2, default=str)

        # post-enrichment overrides: the half of an edit step 3 would undo
        override_notes: list = []
        if opts.get("overrides"):
            from modules.step3_enrich.plan_edit import apply_overrides
            override_notes = apply_overrides(enriched, opts["overrides"])
            for note in override_notes:
                enriched.enrichment_warnings.append(f"Edit: {note}")
            if override_notes:
                update_status(3, "EDIT", msg="; ".join(override_notes[:3]))

        enrich_summary = enriched.summary()
        if enriched.program_plan and enriched.program_plan.get("headline"):
            update_status(3, "PROGRAM",
                          msg=enriched.program_plan["headline"])
        update_status(3, "ENRICH", msg="Gap-filling completed.", p_log={"step": 3, "status": "complete", "label": "ENRICH", "summary": enrich_summary})

        # ── STEP 4+5: GENERATE (wall-graph engine) + RENDER ──────
        update_status(4, "GENERATE", msg="Carving layout with the wall-graph partition engine (best of 6 candidates)...")
        # An edit holds the previous run's seed so the layout stays
        # recognisable — same house, changed room.
        # An edit holds the previous run's seed AND its layout: the seed
        # makes the carve start from the same place, and the signature makes
        # candidate selection prefer the one that keeps the plan
        # recognisable. Seed alone only made resemblance likely.
        layout, svg_filenames, engine_notes = generate_layout(
            enriched, opts.get("seed_from") or run_id, run_dir,
            previous=opts.get("previous_signature"))

        with open(run_dir / "step4_layout_plan.json", "w") as f:
            json.dump(layout.model_dump(), f, indent=2, default=str)

        layout_summary = layout.summary()
        if engine_notes.get("vastu"):
            update_status(4, "VASTU",
                          msg=f"Vastu compliance — {engine_notes['vastu']}")
        update_status(4, "GENERATE",
                      msg=f"Kept {engine_notes.get('kept_candidates', '?')} candidates; "
                          f"best score {engine_notes.get('best_score', '?')}. "
                          f"Time: {layout_summary.get('solve_time_ms', 0)}ms",
                      p_log={"step": 4, "status": "complete", "label": "GENERATE", "summary": layout_summary})

        update_status(5, "RENDER", msg=f"Rendered {len(svg_filenames)} floor plan sheet(s).",
                      p_log={"step": 5, "status": "complete", "label": "RENDER", "files": svg_filenames})

        # ── Store run data ───────────────────────────────────────
        run_data = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "steps": steps_log,
            "step2_summary": step2_summary,
            "step3_summary": enrich_summary,
            "step4_summary": layout_summary,
            "svg_files": svg_filenames,
            "layout_plan": layout.model_dump(),
            "vastu": json.loads(engine_notes["vastu_report"])
            if engine_notes.get("vastu_report") else None,
            "program": enriched.program_plan,
            "alternatives": json.loads(engine_notes["alternatives"])
            if engine_notes.get("alternatives") else [],
            "edit": opts.get("edit_summary"),
            "continuity": json.loads(engine_notes["continuity"])
            if engine_notes.get("continuity") else None,
            # kept so a FURTHER edit of this plan can preserve it too.
            # The engine's own answer wins: it is the signature the candidate
            # ranking used, so an edit compares like with like without
            # depending on two builders sharing a coordinate frame.
            "_signature": _signature_from_notes(engine_notes)
            or _signature_of(layout),
            "requirements": opts.get("requirements")
            or session["requirements"],
            # feature vectors for the preference log, captured at generation
            # so a later pick costs no engine work and cannot disagree with
            # the plans that were actually on screen
            "_preference_vectors": json.loads(
                engine_notes["preference_vectors"])
            if engine_notes.get("preference_vectors") else None,
            # one per alternative, so choosing one makes it the plan a later
            # edit preserves
            "_alternative_signatures": json.loads(
                engine_notes["alternative_signatures"])
            if engine_notes.get("alternative_signatures") else None,
        }
        session["runs"][run_id] = run_data
        
        pipeline_status[run_id]["status"] = "complete"
        pipeline_status[run_id]["result"] = {
            "run_id": run_id,
            "status": "complete",
            "steps": steps_log,
            "step2": {"summary": step2_summary},
            "step3": {"summary": enrich_summary},
            "step4": {"summary": layout_summary},
            "step5": {"svg_files": svg_filenames},
            "svg_files": svg_filenames,
            "layout_plan": layout.model_dump(),
            "vastu": run_data["vastu"],
            "program": run_data["program"],
            "alternatives": run_data["alternatives"],
            "continuity": run_data["continuity"],
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        pipeline_status[run_id]["status"] = "error"
        pipeline_status[run_id]["error"] = str(e)


@app.post("/api/v1/pipeline/run")
def pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    session = _get_session(req.session_id)

    if not session.get("requirements"):
        raise HTTPException(400, "Step 1 not complete — no requirements data")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    opts = req.options or {}
    
    pipeline_status[run_id] = {
        "status": "running",
        "step": 1,
        "logs": [],
        "run_id": run_id
    }
    
    background_tasks.add_task(_run_pipeline_task, run_id, session, opts, run_dir)

    return {"run_id": run_id, "status": "started"}

@app.post("/api/v1/pipeline/regenerate")
def pipeline_regenerate(req: RegenerateRequest,
                        background_tasks: BackgroundTasks):
    """Another go at the same brief.

    The engine seeds itself from the run_id (`engine_bridge._seed_from`), so a
    new run explores a genuinely different candidate set while staying
    reproducible — the same id always gives the same plans back.
    """
    session = _get_session(req.session_id)
    if not session.get("requirements"):
        raise HTTPException(400, "Step 1 not complete — no requirements data")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_r"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_status[run_id] = {"status": "running", "step": 1, "logs": [],
                               "run_id": run_id}
    background_tasks.add_task(_run_pipeline_task, run_id, session,
                              req.options or {}, run_dir)
    return {"run_id": run_id, "status": "started", "regenerated": True}


@app.post("/api/v1/runs/{run_id}/edit")
def edit_plan(run_id: str, req: EditRequest,
              background_tasks: BackgroundTasks):
    """Change a plan in words: "make the kitchen bigger, move the pooja
    room to the north east".

    The edit is applied to the BRIEF and the pipeline runs again, rather
    than the finished plan being mutated. Room sizes, floor assignment,
    bathroom attachment and the adjacency graph are all derived from each
    other in step 3; editing the output directly would leave every one of
    them describing a house that no longer exists.

    `preview: true` parses and explains without generating, so a user can
    see what was understood before spending a run on it.
    """
    from modules.step3_enrich.plan_edit import (
        apply_to_requirements, overrides_from, parse,
    )

    session = _get_session(req.session_id)
    previous = session.get("runs", {}).get(run_id)
    base_requirements = (previous or {}).get("requirements")         or session.get("requirements")
    if not base_requirements:
        raise HTTPException(400, "No brief to edit — run the pipeline first")

    edit = parse(req.text)
    if not edit.ok:
        return {
            "understood": False,
            "summary": edit.summary(),
            "unparsed": edit.unparsed,
            "message": ("I could not turn that into a change. Try naming a "
                        "room and what to do with it — for example \"make "
                        "the kitchen bigger\" or \"move the pooja room to "
                        "the north east\"."),
        }

    new_requirements, req_notes = apply_to_requirements(
        base_requirements, [i for i in edit.intents if i.is_requirement])
    overrides = overrides_from(edit.intents)

    if req.preview:
        return {"understood": True, "preview": True,
                "summary": edit.summary(), "changes": req_notes,
                "unparsed": edit.unparsed,
                "intents": [i.to_dict() for i in edit.intents]}

    new_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_e"
    run_dir = OUTPUT_DIR / new_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_status[new_run_id] = {"status": "running", "step": 1,
                                   "logs": [], "run_id": new_run_id}
    background_tasks.add_task(
        _run_pipeline_task, new_run_id, session,
        {"requirements": new_requirements,
         "overrides": overrides,
         "previous_signature": (previous or {}).get("_signature"),
         # hold the ORIGINAL seed: same house, changed room
         "seed_from": (previous or {}).get("seed_from") or run_id,
         "edit_summary": {"text": req.text, "summary": edit.summary(),
                          "changes": req_notes,
                          "unparsed": edit.unparsed,
                          "from_run": run_id}},
        run_dir)
    return {"understood": True, "run_id": new_run_id, "status": "started",
            "edited_from": run_id, "summary": edit.summary(),
            "changes": req_notes, "unparsed": edit.unparsed}


@app.get("/api/v1/runs/{run_id}/alternatives")
def list_alternatives(session_id: str, run_id: str):
    """The other plans the engine made for this brief.

    They were always generated — `k` candidates survive the reviewer every
    run — and until now only the top one was ever rendered.
    """
    session = _get_session(session_id)
    run = session.get("runs", {}).get(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found in this session")
    return {
        "run_id": run_id,
        "chosen_rank": run.get("chosen_rank"),
        "alternatives": run.get("alternatives", []),
    }


@app.post("/api/v1/runs/{run_id}/choose")
def choose_alternative(run_id: str, req: ChooseRequest):
    """Record which plan the user preferred.

    This is the only signal that teaches the critic TASTE. Perturbation
    labels (critic/perturb.py) teach it to recognise damage; a real pick
    among real options is the thing damage-detection cannot substitute for,
    and it cannot be collected retroactively — hence logging from the first
    day the UI can offer a choice.
    """
    session = _get_session(req.session_id)
    run = session.get("runs", {}).get(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found in this session")
    alts = run.get("alternatives", [])
    if not 0 <= req.rank < len(alts):
        raise HTTPException(400,
                            f"rank {req.rank} outside 0..{len(alts) - 1}")

    run["chosen_rank"] = req.rank
    # The chosen plan becomes the one a later edit holds on to. Recording the
    # pick and then editing a different layout would make the choice cosmetic.
    sigs = run.get("_alternative_signatures")
    if sigs and 0 <= req.rank < len(sigs):
        try:
            from modules.step4_generate.engine import continuity as cont
            run["_signature"] = cont.signature_from_dict(sigs[req.rank])
        except Exception as exc:
            logger.warning("could not adopt option %s of %s: %s",
                           req.rank, run_id, exc)
    logged = False
    payload = run.get("_preference_vectors")
    if payload and len(payload.get("vectors", [])) == len(alts):
        try:
            from modules.step4_generate.critic.preferences import (
                PreferenceRecord, DEFAULT_LOG,
            )
            import os as _os
            record = PreferenceRecord(
                brief=run_id, chosen=req.rank,
                vectors=payload["vectors"],
                soft_scores=payload.get("soft_scores", []),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                note=req.note)
            _os.makedirs(_os.path.dirname(_os.path.abspath(DEFAULT_LOG)),
                         exist_ok=True)
            with open(DEFAULT_LOG, "a", encoding="utf-8") as fh:
                fh.write(record.to_json())
                fh.write(chr(10))
            logged = True
        except Exception as exc:
            # a failed log must never cost the user their choice
            logger.warning("preference log failed for %s: %s", run_id, exc)

    return {"run_id": run_id, "chosen_rank": req.rank,
            "svg": alts[req.rank].get("svg"), "logged": logged}


@app.get("/api/v1/pipeline/status/{run_id}")
def pipeline_status_endpoint(run_id: str):
    if run_id not in pipeline_status:
        raise HTTPException(404, "Run ID not found")
    return pipeline_status[run_id]


# ══════════════════════════════════════════════════════════════════
# ARTIFACT / SVG ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/v1/runs/{run_id}/svg/{filename}")
def get_svg(run_id: str, filename: str):
    # Ensure .svg extension
    if not filename.endswith(".svg"):
        filename += ".svg"
    path = OUTPUT_DIR / run_id / filename
    if not path.exists():
        # Try common naming patterns
        run_dir = OUTPUT_DIR / run_id
        if run_dir.exists():
            svgs = list(run_dir.glob("*.svg"))
            # Try matching by floor name
            for svg in svgs:
                if filename.replace(".svg", "").lower() in svg.name.lower():
                    return FileResponse(svg, media_type="image/svg+xml")
        raise HTTPException(404, f"SVG not found: {filename}")
    return FileResponse(path, media_type="image/svg+xml")


@app.get("/api/v1/runs/{run_id}/files")
def list_run_files(run_id: str):
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"Run {run_id} not found")
    files = []
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "type": f.suffix,
            })
    return {"run_id": run_id, "files": files}


@app.get("/api/v1/runs/{run_id}/json/{artifact}")
def get_json_artifact(run_id: str, artifact: str):
    if not artifact.endswith(".json"):
        artifact += ".json"
    path = OUTPUT_DIR / run_id / artifact
    if not path.exists():
        raise HTTPException(404, f"Artifact not found: {artifact}")
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
# HEALTH / CONFIG
# ══════════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
def health():
    engine_ready = False
    try:
        from api.engine_bridge import Orchestrator  # noqa: F401
        engine_ready = True
    except Exception:
        pass

    from modules.diagnostics import self_check
    diag = self_check()

    return {
        # "ok" means the service is up. It does NOT mean every subsystem is
        # healthy — read `diagnostics.overall` for that. Conflating the two is
        # how this project ran for months on silent fallbacks.
        "status": "ok",
        "engine": "wall_graph_carver (modules/step4_generate)",
        "engine_ready": engine_ready,
        "version": "3.0.0",
        "diagnostics": diag,
    }


@app.get("/api/v1/diagnostics")
def diagnostics():
    """Full subsystem report — what is real and what is running on fallbacks."""
    from modules.diagnostics import self_check
    return self_check()


@app.get("/api/v1/config/options")
def config_options():
    return {
        "plot_shapes": ["rectangular", "L-shaped", "irregular", "square"],
        "directions": ["north", "south", "east", "west", "north_east", "north_west", "south_east", "south_west"],
        "room_types": ["Bedroom", "Master Bedroom", "Kitchen", "Living Room", "Dining Room",
                       "Pooja Room", "Bathroom", "Balcony", "Study Room", "Store Room",
                       "Staircase", "Car Parking", "Utility", "Passage"],
        "floors_max": 3,
        "solvers": ["wall_graph_carver"],
    }


# ══════════════════════════════════════════════════════════════════
# SERVE FRONTEND STATIC FILES
# ══════════════════════════════════════════════════════════════════

# The UI is plain HTML/CSS/JS — no build step. It is served from this origin so
# the page can call /api/v1/* directly, with no CORS hop and no second server.
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>PlanGen API is running.</h1>"
        f"<p>No UI found at <code>{FRONTEND_DIR}</code>.</p>",
        status_code=200,
    )


# Mounted last so every /api/v1/* route above still wins the match.
# html=True lets /chatPage.html and /assets/* resolve directly.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
else:
    logger.warning("frontend/ not found — serving the API only.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
