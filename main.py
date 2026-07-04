import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from modules.step1_parse.parser import Module1Pipeline
from modules.step2_match.matcher import PatternMatcher
from modules.step3_enrich.enricher import Enricher
from modules.step4_generate.generator import LayoutGenerator

# ── JSON Output Directory ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _PROJECT_ROOT / "extracted data" / "prompt_extraction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure global logger for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PlanGen_Main")


def _save_json(data: dict, label: str, run_id: str) -> Path:
    """
    Persist a JSON dict to the prompt_extraction folder.

    Files are named:  <run_id>_<label>.json
    Example:          20260612_210605_step1_parsed.json

    Returns the path to the saved file.
    """
    filename = f"{run_id}_{label}.json"
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved → %s", filepath)
    return filepath


def run_pipeline(user_prompt: str, image_path: str = None):
    """
    Executes the PlanGen pipeline end-to-end.

    Supports three modes:
      1. Text-only:      run_pipeline("I want a 3BHK...")
      2. Image-only:     run_pipeline(None, image_path="sketch.jpg")
      3. Text + Image:   run_pipeline("3BHK villa", image_path="plot.jpg")

    After initial parsing, enters an interactive loop to gather any
    missing details from the user via console input.

    All intermediate and final JSON outputs are saved to:
      extracted data/prompt_extraction/<timestamp>_<label>.json
    """
    # Unique run ID based on timestamp (used for all file names in this run)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("STARTING PIPELINE EXECUTION  [run_id=%s]", run_id)
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: PARSE (with optional image analysis)
    # ---------------------------------------------------------
    parser = Module1Pipeline()

    # Choose execution mode based on inputs
    if image_path:
        logger.info("Image input detected: %s", image_path)
        step1_result = parser.execute_with_image(
            image_input=image_path,
            user_text=user_prompt,
        )
    elif user_prompt:
        step1_result = parser.execute(user_prompt)
    else:
        logger.error("No input provided (neither text nor image).")
        print("\nError: Please provide a text prompt, an image, or both.")
        return None

    # ── Save Step 1 parsed data ──
    if step1_result.get("data"):
        _save_json(step1_result["data"], "step1_parsed", run_id)

    # ---------------------------------------------------------
    # INTERACTIVE LOOP: Fill in missing details
    # ---------------------------------------------------------
    if step1_result["status"] in ("incomplete", "error") and step1_result.get("data"):
        print("\n" + "=" * 60)
        print("INTERACTIVE DETAIL GATHERING")
        print("=" * 60)

        # Show initial architect response
        if step1_result.get("clarification_prompt"):
            print(f"\nArchitect: {step1_result['clarification_prompt']}")

        # Enter the interactive loop
        action = parser.get_next_interactive_action()

        while action["action"] == "ask":
            # Display the question
            print(f"\nArchitect: {action['question']}")

            # Show what's still missing (debug info)
            missing = action.get("missing_summary", {})
            tier1_missing = missing.get("tier1", [])
            if tier1_missing:
                logger.info("Still missing (Tier 1): %s", ", ".join(tier1_missing))

            # Get user's answer
            try:
                answer = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSession ended by user.")
                return None

            if not answer:
                print("(Please provide an answer to continue)")
                continue

            # Process the answer
            result = parser.process_interactive_answer(answer)

            if result["status"] == "interactive":
                # More questions to ask
                action = {
                    "action": result["action"],
                    "question": result.get("question"),
                    "missing_summary": result.get("missing_summary"),
                }
            elif result["status"] == "success":
                # All data collected!
                step1_result = result
                break
            elif result["status"] == "incomplete":
                # Check if there are more interactive questions
                action = parser.get_next_interactive_action()
                if action["action"] == "complete":
                    # Interactive loop is done — use whatever we have
                    step1_result = result
                    break
            else:
                # Error or unexpected status
                print(f"\nArchitect: {result.get('message', 'Something went wrong.')}")
                action = parser.get_next_interactive_action()

    # ---------------------------------------------------------
    # CHECK FINAL STATUS
    # ---------------------------------------------------------
    if step1_result["status"] != "success":
        logger.warning("Pipeline paused: Step 1 requires more information.")
        print("\nArchitect:")
        print(step1_result.get("clarification_prompt", "I need a few more details to proceed."))
        return None

    # ── Save final Step 1 result (after interactive gathering) ──
    if step1_result.get("data"):
        _save_json(step1_result["data"], "step1_final", run_id)

    # The output is a dict, we need to convert it back to the Pydantic object
    # for strong typing in Step 2, or just pass the model object.
    from models import BuildingRequirements
    reqs_model = BuildingRequirements.model_validate(step1_result["data"])

    # ---------------------------------------------------------
    # STEP 2: FETCH MATCHING PATTERNS
    # ---------------------------------------------------------
    matcher = PatternMatcher()
    knowledge_bundle = matcher.fetch_patterns(reqs_model)

    # ── Save Step 2 knowledge bundle ──
    bundle_path = _save_json(
        knowledge_bundle.model_dump(), "step2_knowledge_bundle", run_id
    )

    logger.info("Step 2 complete — bundle quality=%.3f", knowledge_bundle.match_quality_score)

    # ---------------------------------------------------------
    # STEP 3: ENRICH (gap-fill → EnrichedPlan)
    # ---------------------------------------------------------
    enricher = Enricher(use_gemini=True)
    enriched_plan = enricher.enrich(reqs_model, knowledge_bundle)

    # ── Save Step 3 enriched plan ──
    enriched_path = _save_json(
        enriched_plan.model_dump(), "step3_enriched_plan", run_id
    )

    # ── Log enrichment summary ──
    summary = enriched_plan.summary()
    logger.info(
        "Step 3 complete — %d rooms (%d implicit), %d floors, "
        "area=%s sqft, budget_ok=%s, vastu=%s",
        summary["total_rooms"],
        summary["implicit_rooms_added"],
        summary["total_floors"],
        summary["total_target_area_sqft"],
        summary["area_budget_ok"],
        summary["vastu_enabled"],
    )
    if enriched_plan.enrichment_warnings:
        for w in enriched_plan.enrichment_warnings:
            logger.warning("  ⚠  %s", w)

    # ---------------------------------------------------------
    # STEP 4: GENERATE LAYOUT (room coordinates on each floor)
    # ---------------------------------------------------------
    generator    = LayoutGenerator(prefer_cpsat=True, cpsat_timeout_s=20.0)
    layout_plan  = generator.generate(enriched_plan, run_id=run_id, source_json=str(enriched_path))

    # ── Save Step 4 layout plan ──
    layout_path = _save_json(
        layout_plan.model_dump(), "step4_layout_plan", run_id
    )

    layout_summary = layout_plan.summary()
    logger.info(
        "Step 4 complete — %d rooms placed, %.1f sqft, solver=%s (%s), "
        "adj=%.3f zone=%.3f quality=%.3f",
        layout_summary["total_rooms_placed"],
        layout_summary["total_area_sqft"],
        layout_summary["solver"],
        layout_summary["solver_status"],
        layout_summary["adjacency_score"],
        layout_summary["zone_score"],
        layout_summary["layout_quality_score"],
    )
    if layout_plan.layout_warnings:
        for w in layout_plan.layout_warnings:
            logger.warning("  ⚠  %s", w)

    # ---------------------------------------------------------
    # STEP 5: RENDER SVG FLOOR PLANS
    # ---------------------------------------------------------
    svg_paths = []
    svg_output_dir = _PROJECT_ROOT / "output" / run_id
    try:
        from modules.step4_generate.renderer import FloorPlanRenderer

        renderer = FloorPlanRenderer(
            plan         = layout_plan,
            output_dir   = str(svg_output_dir),
            project_name = f"PlanGen — {run_id}",
        )
        svg_paths = renderer.render_all()
        logger.info("Step 5 complete — %d SVG floor plans rendered", len(svg_paths))
    except ImportError as e:
        logger.warning("Step 5 skipped — renderer dependency missing: %s", e)
    except Exception as e:
        logger.warning("Step 5 failed — SVG rendering error: %s", e)

    logger.info("Pipeline Execution Complete!")
    print(f"\n✅ All output JSONs saved to: {OUTPUT_DIR}")
    print(f"   Step 1 final:     {run_id}_step1_final.json")
    print(f"   Step 2 bundle:    {bundle_path.name}")
    print(f"   Step 3 enriched:  {enriched_path.name}")
    print(f"   Step 4 layout:    {layout_path.name}")
    if svg_paths:
        print(f"\n✅ SVG floor plans rendered to: {svg_output_dir}")
        for sp in svg_paths:
            print(f"   🏠  {Path(sp).name}")
    print(f"\n── Enriched Plan Summary ──────────────────────────")
    print(f"   Rooms:            {summary['total_rooms']} "
          f"({summary['implicit_rooms_added']} added by enricher)")
    print(f"   Floors:           {summary['total_floors']}")
    print(f"   Plot:             {summary['plot_ft']} ft")
    print(f"   Net buildable:    {summary['net_buildable_ft']} ft")
    print(f"   Total area:       {summary['total_target_area_sqft']} sqft")
    print(f"   Area budget:      {'✅ OK' if summary['area_budget_ok'] else '⚠️  Exceeds FAR'}")
    print(f"   Vastu:            {'enabled' if summary['vastu_enabled'] else 'disabled'}")
    print(f"   Match quality:    {summary['match_quality']:.3f}")
    if enriched_plan.implicit_rooms_added:
        print(f"   Added implicitly: {', '.join(enriched_plan.implicit_rooms_added)}")
    print(f"─────────────────────────────────────────────────")
    print(f"\n── Layout Plan Summary ────────────────────────────")
    print(f"   Rooms placed:     {layout_summary['total_rooms_placed']}")
    print(f"   Total area:       {layout_summary['total_area_sqft']} sqft")
    print(f"   Solver:           {layout_summary['solver']} ({layout_summary['solver_status']})")
    print(f"   Solve time:       {layout_summary['solve_time_ms']} ms")
    print(f"   Adjacency score:  {layout_summary['adjacency_score']:.3f}")
    print(f"   Zone score:       {layout_summary['zone_score']:.3f}")
    print(f"   Layout quality:   {layout_summary['layout_quality_score']:.3f}")
    for f_plan in layout_plan.floors:
        print(f"   {f_plan.floor_label:<16}  {f_plan.room_count} rooms, "
              f"{f_plan.floor_area_placed_sqft:.0f} sqft "
              f"({f_plan.floor_coverage_pct:.0f}% coverage)")
    print(f"─────────────────────────────────────────────────")

    return {"layout_plan": layout_plan, "svg_paths": svg_paths}


def run_interactive_demo():
    """
    Interactive demo mode — shows the full conversational flow.
    Supports text, image, or both as input.
    """
    print("=" * 60)
    print("  PlanGen — AI Floor Plan Generator")
    print("  Interactive Consultation Mode")
    print("=" * 60)
    print("\nWelcome! You can describe your dream home in text,")
    print("provide an image of your plot/plan, or both.\n")

    # Get text input
    text_input = input("Describe your requirements (or press Enter to skip): ").strip()
    if not text_input:
        text_input = None

    # Get image input
    image_input = input("Image path (or press Enter to skip): ").strip()
    if not image_input:
        image_input = None

    if not text_input and not image_input:
        print("\nPlease provide at least a text description or an image.")
        return

    # Run the pipeline
    bundle = run_pipeline(text_input, image_input)

    if bundle:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"All output JSONs have been saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            run_interactive_demo()
        elif sys.argv[1] == "--image":
            # Image-only or image+text mode
            image_path = sys.argv[2] if len(sys.argv) > 2 else None
            text = sys.argv[3] if len(sys.argv) > 3 else None
            if not image_path:
                print("Usage: python main.py --image <path> [optional text]")
            else:
                bundle = run_pipeline(text, image_path)
                if bundle:
                    print("\n" + "=" * 60)
                    print("PIPELINE COMPLETE")
                    print("=" * 60)
                    print(f"All output JSONs saved to:\n  {OUTPUT_DIR}")
        else:
            # Treat all remaining args as the text prompt
            test_prompt = " ".join(sys.argv[1:])
            bundle = run_pipeline(test_prompt)
            if bundle:
                print("\n" + "=" * 60)
                print("PIPELINE COMPLETE")
                print("=" * 60)
                print(f"All output JSONs saved to:\n  {OUTPUT_DIR}")
    else:
        # Default: run interactive demo
        run_interactive_demo()
