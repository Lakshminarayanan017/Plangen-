import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from modules.step1_parse.parser import Module1Pipeline
from modules.step2_match.matcher import PatternMatcher

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
    
    logger.info("Pipeline Execution Complete!")
    print(f"\n✅ All output JSONs saved to: {OUTPUT_DIR}")
    print(f"   Final bundle: {bundle_path.name}")
    return knowledge_bundle


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
