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
from modules.step4_generate.generator import LayoutGenerator

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

class SessionResponse(BaseModel):
    session_id: str
    created_at: str


# ── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="PlanGen API",
    version="2.1.0",
    description="AI-Powered Floor Plan Generator",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: get or create session ────────────────────────────────
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

    result = parser.execute(req.text)
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
        update_status(1, "SYS", msg="Initiating PlanGen Engine v3.4.2")
        reqs = BuildingRequirements.model_validate(session["requirements"])
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

        enrich_summary = enriched.summary()
        update_status(3, "ENRICH", msg="Gap-filling completed.", p_log={"step": 3, "status": "complete", "label": "ENRICH", "summary": enrich_summary})

        # ── STEP 4: GENERATE ─────────────────────────────────────
        update_status(4, "GENERATE", msg="Generating layout plan with Constraint Solver (AR Engine)...")
        prefer_cpsat = opts.get("prefer_cpsat", True)
        cpsat_timeout = opts.get("cpsat_timeout_s", 20.0)
        generator = LayoutGenerator(prefer_cpsat=prefer_cpsat, cpsat_timeout_s=cpsat_timeout)
        layout = generator.generate(enriched, run_id=run_id)

        with open(run_dir / "step4_layout_plan.json", "w") as f:
            json.dump(layout.model_dump(), f, indent=2, default=str)

        layout_summary = layout.summary()
        update_status(4, "GENERATE", msg=f"Layout generation complete. Time: {layout_summary.get('solve_time_ms', 0)}ms", p_log={"step": 4, "status": "complete", "label": "GENERATE", "summary": layout_summary})

        # ── STEP 5: RENDER SVG ───────────────────────────────────
        update_status(5, "RENDER", msg="Rendering final SVG blueprints...")
        svg_filenames = []
        try:
            from modules.step4_generate.renderer import FloorPlanRenderer
            renderer = FloorPlanRenderer(
                plan=layout,
                output_dir=str(run_dir),
                project_name=f"PlanGen — {run_id}",
            )
            svg_paths = renderer.render_all()
            svg_filenames = [Path(p).name for p in svg_paths]
            update_status(5, "RENDER", msg=f"Rendered {len(svg_filenames)} view(s).", p_log={"step": 5, "status": "complete", "label": "RENDER", "files": svg_filenames})
        except Exception as e:
            logger.warning("Render failed: %s", e)
            update_status(5, "ERROR", msg=f"Render failed: {str(e)}", p_log={"step": 5, "status": "error", "label": "RENDER", "error": str(e)})

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
    ar_ready = False
    try:
        from modules.step4_generate.autoregressive_engine import AutoregressiveLayoutEngine
        engine = AutoregressiveLayoutEngine()
        ar_ready = engine._gnn is not None and engine._ar is not None
    except Exception:
        pass

    return {
        "status": "ok",
        "ar_engine_ready": ar_ready,
        "model_epoch": 50,
        "version": "2.1.0",
    }


@app.get("/api/v1/config/options")
def config_options():
    return {
        "plot_shapes": ["rectangular", "L-shaped", "irregular", "square"],
        "directions": ["north", "south", "east", "west", "north_east", "north_west", "south_east", "south_west"],
        "room_types": ["Bedroom", "Master Bedroom", "Kitchen", "Living Room", "Dining Room",
                       "Pooja Room", "Bathroom", "Balcony", "Study Room", "Store Room",
                       "Staircase", "Car Parking", "Utility", "Passage"],
        "floors_max": 3,
        "solvers": ["autoregressive", "cp_sat", "greedy"],
    }


# ══════════════════════════════════════════════════════════════════
# SERVE FRONTEND STATIC FILES
# ══════════════════════════════════════════════════════════════════

FRONTEND_DIR = PROJECT_ROOT / "frontend"

@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PlanGen API is running. Frontend not found.</h1>")


# Mount static files (CSS, JS, assets)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
