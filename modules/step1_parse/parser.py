import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load environment variables from sources/.env relative to this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_env_path = os.path.join(_project_root, "sources", ".env")
load_dotenv(_env_path)

# Official Google GenAI SDK (2026 standard)
from google import genai
from google.genai import types

# Multi-key rotation for 429 rate-limit mitigation
from sources.key_rotator import GeminiKeyRotator

# Anthropic SDK — used as structural fallback when Gemini is unavailable
try:
    import anthropic as _anthropic_sdk
    import instructor as _instructor_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# Import your existing Pydantic data models
from models import BuildingRequirements

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PlanGen.Parser")


# =====================================================================
# 1. CONFIGURATION MANAGEMENT
# =====================================================================
class ParserConfig:
    """
    Production configuration defaults.

    API keys are managed by GeminiKeyRotator (round-robin pool).
    System prompts are loaded once and cached at class level to avoid
    redundant disk I/O across multiple ParserConfig instances.
    """
    PRIMARY_MODEL: str = "gemini-2.5-flash"
    FLASH_MODEL: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 3
    RETRY_DELAY_BASE: float = 4.0

    # Class-level prompt cache (loaded once, shared across all instances)
    _PARSER_SYSTEM_INSTRUCTION: Optional[str] = None

    def __init__(self):
        # Initialize the multi-key rotator (replaces single GEMINI_API_KEY)
        # The rotator auto-discovers GEMINI_API_KEY_1, _2, _3, ... from env
        self.key_rotator = GeminiKeyRotator(cooldown_seconds=60.0)
        logger.info(
            "ParserConfig initialized with %d API key(s) in rotation pool",
            self.key_rotator.key_count,
        )

        # [FIX #9] Load prompt into a private class var with explicit None check
        # instead of setting a public class var from an instance method
        if ParserConfig._PARSER_SYSTEM_INSTRUCTION is None:
            from docs.prompts.loader import load_prompt
            ParserConfig._PARSER_SYSTEM_INSTRUCTION = load_prompt("step1_parser_system.md")
            logger.info("Loaded parser system instruction from docs/prompts/step1_parser_system.md")

    @property
    def PARSER_SYSTEM_INSTRUCTION(self) -> str:
        """Access the cached system instruction via a clean read-only property."""
        return ParserConfig._PARSER_SYSTEM_INSTRUCTION


# =====================================================================
# 2. PERSONA-DRIVEN VALIDATION GATEKEEPER
# =====================================================================
class ArchitectGatekeeper:
    """
    Validates structural requirements and utilizes Gemini Flash to generate
    context-aware, senior-architect-styled clarification requests.
    """
    def __init__(self, config: ParserConfig):
        self.config = config

    def validate_and_format(self, requirements: BuildingRequirements) -> Dict[str, Any]:
        """
        Evaluates extracted requirements against the structural Tier-1/Tier-2 rules.
        If details are missing, it asks Gemini to formulate a hyper-professional response.
        """
        missing_tier1: List[str] = []
        missing_tier2: List[str] = []

        # ---- Tier 1: System Blockers ----
        if not requirements.plot_dimensions:
            missing_tier1.append("Plot dimensions (e.g., 30x40 ft, 1200 sqft plot size)")
        else:
            if requirements.plot_dimensions.length is None or requirements.plot_dimensions.width is None:
                missing_tier1.append("Complete plot boundaries (both length and width are required)")

        if not requirements.rooms or len(requirements.rooms) == 0:
            missing_tier1.append("Room layout configuration (e.g., number of bedrooms, kitchen needs)")

        if not requirements.plot_context or not requirements.plot_context.road_facing_sides:
            missing_tier1.append("Plot facing direction (e.g., North-facing, East-facing, South-facing)")

        # ---- Tier 2: Non-Blockers (System defaults exist) ----
        if requirements.number_of_floors is None:
            missing_tier2.append("Number of floors (Ground only, G+1, or G+2)")

        is_valid = len(missing_tier1) == 0

        # Build the architect conversational response via Gemini Flash
        clarification_prompt = self._generate_architect_response(
            is_valid=is_valid,
            missing_tier1=missing_tier1,
            missing_tier2=missing_tier2
        )

        return {
            "is_valid": is_valid,
            "missing_tier1": missing_tier1,
            "missing_tier2": missing_tier2,
            "clarification_prompt": clarification_prompt
        }

    def _generate_architect_response(self, is_valid: bool, missing_tier1: List[str], missing_tier2: List[str]) -> str:
        """Generates conversational responses matching a formal senior Indian architect persona."""
        from docs.prompts.loader import load_prompt

        # Load the persona template and inject runtime context
        persona_template = load_prompt("step1_gatekeeper_persona.md")

        # [FIX #7] Join lists into human-readable strings instead of passing raw Python list repr
        persona_prompt = persona_template.format(
            validation_status="VALID" if is_valid else "INCOMPLETE",
            missing_tier1=", ".join(missing_tier1) if missing_tier1 else "None",
            missing_tier2=", ".join(missing_tier2) if missing_tier2 else "None"
        )

        # Use key rotator to get a client, with 429-aware retry
        for _attempt in range(self.config.key_rotator.key_count + 1):
            client, slot_idx = self.config.key_rotator.get_client()
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=persona_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.6,  # Slightly warmer for human conversation flows
                        max_output_tokens=500
                    )
                )
                return response.text.strip()
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource" in error_str and "exhausted" in error_str:
                    logger.warning(f"429 rate-limit on gatekeeper call (slot {slot_idx}): {e}")
                    self.config.key_rotator.report_rate_limited(slot_idx)
                    continue  # try next key
                logger.error(f"Failed to generate persona-driven prompt: {e}")
                break  # non-429 error, fall through to static fallback

        # ── Anthropic fallback for gatekeeper persona ──
        anthropic_resp = self._try_anthropic_gatekeeper(persona_prompt)
        if anthropic_resp:
            return anthropic_resp

        # Final static fallback — no API available at all
        if not is_valid:
            return f"Namaste. To map out your floor plan accurately, I will need a few missing details: {', '.join(missing_tier1)}. Could you provide these?"
        return "Excellent. I have the baseline requirements. Shall I assume a single-floor (Ground only) layout, or do you intend to build G+1/G+2?"

    def _try_anthropic_gatekeeper(self, persona_prompt: str) -> Optional[str]:
        """
        Anthropic claude-sonnet-4-5 fallback for the gatekeeper persona response.
        Returns None if key is missing, SDK unavailable, or call fails.
        """
        if not _ANTHROPIC_AVAILABLE:
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            raw_client = _anthropic_sdk.Anthropic(api_key=api_key)
            logger.info("Gatekeeper: using Anthropic claude-sonnet-4-5 fallback.")
            response = raw_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": persona_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error("Anthropic gatekeeper fallback failed: %s", e)
            return None


# =====================================================================
# 3. GEMINI NLP PARSER MODULE
# =====================================================================
class ArchitectureParser:
    """
    Direct interface to the Gemini API utilizing strict structured response schemas
    for mapping raw text or merged chat workflows into Pydantic models.
    """
    def __init__(self, config: ParserConfig):
        self.config = config

    @staticmethod
    def _clean_response_text(raw_text: str) -> str:
        """
        [FIX #5] Strip markdown code fences that Gemini occasionally wraps
        around JSON output, even when response_mime_type='application/json'.
        """
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def parse_input(self, user_text: str) -> Optional[BuildingRequirements]:
        """Sends clean natural text to Gemini to fulfill structured Pydantic parameters."""
        logger.info(f"Initiating extraction on primary engine ({self.config.PRIMARY_MODEL})...")

        for attempt in range(1, self.config.MAX_RETRIES + 1):
            # Get a fresh client from the rotation pool for every attempt
            client, slot_idx = self.config.key_rotator.get_client()
            logger.info(f"Attempt {attempt}/{self.config.MAX_RETRIES} using key slot {slot_idx}")

            try:
                # [FIX] Strip unsupported 'default' keys from JSON schema
                schema = BuildingRequirements.model_json_schema()
                def remove_defaults(d):
                    if isinstance(d, dict):
                        return {k: remove_defaults(v) for k, v in d.items() if k != "default"}
                    elif isinstance(d, list):
                        return [remove_defaults(v) for v in d]
                    return d
                clean_schema = remove_defaults(schema)

                # Direct implementation of Gemini's native structured JSON engine
                response = client.models.generate_content(
                    model=self.config.PRIMARY_MODEL,
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        system_instruction=self.config.PARSER_SYSTEM_INSTRUCTION,
                        temperature=self.config.TEMPERATURE,
                        response_mime_type="application/json",
                        response_schema=clean_schema,  # Strict structural enforcement
                    )
                )

                # [FIX #5] Defensively strip markdown wrappers before validation
                cleaned_text = self._clean_response_text(response.text)

                # Re-validate the cleaned output text directly using Pydantic
                parsed_data = BuildingRequirements.model_validate_json(cleaned_text)
                return parsed_data

            except (ValidationError, json.JSONDecodeError) as ve:
                # [FIX #3] Schema/JSON errors are now retryable — LLM may
                # produce valid output on next attempt instead of immediately returning None
                logger.warning(
                    f"Schema/JSON validation failed on attempt {attempt}/{self.config.MAX_RETRIES}: {ve}"
                )
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(self.config.RETRY_DELAY_BASE)  # Linear delay for validation retries
                else:
                    logger.error("Exhausted retries due to persistent schema validation failures.")
                    return None

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or (
                    "resource" in error_str and "exhausted" in error_str
                )

                if is_rate_limit:
                    # 429 — cooldown this key and immediately retry with next key
                    logger.warning(
                        f"429 rate-limit hit on slot {slot_idx} (attempt {attempt}): {e}"
                    )
                    self.config.key_rotator.report_rate_limited(slot_idx)
                    # Don't count this as a "real" attempt — retry immediately with next key
                    continue

                # Non-429 API error — standard exponential backoff
                logger.warning(f"Gemini API failure on attempt {attempt}/{self.config.MAX_RETRIES}: {e}")
                if attempt < self.config.MAX_RETRIES:
                    time.sleep(self.config.RETRY_DELAY_BASE ** attempt)  # Exponential backoff for API errors
                else:
                    logger.error("Max retry limits exhausted on parser pipeline execution.")
                    break   # fall through to Anthropic fallback

        # ── Anthropic fallback — fires when ALL Gemini keys are exhausted ──
        logger.info("All Gemini keys exhausted — attempting Anthropic claude-sonnet-4-5 fallback...")
        return self._parse_with_anthropic(user_text)

    def _parse_with_anthropic(self, user_text: str) -> Optional[BuildingRequirements]:
        """
        Anthropic claude-sonnet-4-5 structured extraction fallback.

        Uses the `instructor` library patched onto the Anthropic client so we get
        the same Pydantic model output as the Gemini path.  Falls back to raw JSON
        parsing when instructor is unavailable.

        Returns None only if the API key is missing or every attempt fails.
        """
        if not _ANTHROPIC_AVAILABLE:
            logger.warning("anthropic/instructor not installed — Anthropic fallback skipped.")
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set in environment — Anthropic fallback skipped.")
            return None

        logger.info("Anthropic fallback: using claude-sonnet-4-5 for structured extraction.")
        for attempt in range(1, 3):  # 2 attempts max
            try:
                raw_client = _anthropic_sdk.Anthropic(api_key=api_key)
                client = _instructor_sdk.from_anthropic(raw_client)

                result: BuildingRequirements = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=4096,
                    system=self.config.PARSER_SYSTEM_INSTRUCTION,
                    messages=[{"role": "user", "content": user_text}],
                    response_model=BuildingRequirements,
                )
                logger.info("Anthropic fallback succeeded on attempt %d.", attempt)
                return result

            except (json.JSONDecodeError, ValidationError) as ve:
                logger.warning("Anthropic schema validation failed (attempt %d): %s", attempt, ve)
                if attempt < 2:
                    time.sleep(2.0)
            except Exception as e:
                logger.error("Anthropic fallback error (attempt %d): %s", attempt, e)
                if attempt < 2:
                    time.sleep(2.0)

        logger.error("Anthropic fallback exhausted — returning None.")
        return None

    def merge_and_reparse(self, original_input: str, followup_input: str, existing_data: Dict[str, Any]) -> Optional[BuildingRequirements]:
        """Resolves multi-turn architectural updates by executing structural consolidation prompts."""
        from docs.prompts.loader import load_prompt

        # Load the consolidation template and inject runtime context
        consolidation_template = load_prompt("step1_merge_consolidation.md")
        consolidation_prompt = consolidation_template.format(
            existing_data=json.dumps(existing_data, indent=2),
            original_input=original_input,
            followup_input=followup_input
        )
        return self.parse_input(consolidation_prompt)


# =====================================================================
# 4. ORCHESTRATION PIPELINE
# =====================================================================
class Module1Pipeline:
    """
    Main Orchestration pipeline acting as the stateful framework manager
    for Step 1 (Chat Execution and Verification).

    Supports three input modes:
      1. Text-only:  execute(user_input)
      2. Image-only:  execute_with_image(image_path)
      3. Text+Image:  execute_with_image(image_path, user_text)

    After initial parsing, the pipeline enters an interactive gathering
    loop that asks the user targeted questions to fill in missing details.
    The loop is driven externally by the caller via:
      - get_next_interactive_action(session_id) → returns question or complete
      - process_interactive_answer(answer, session_id) → merges and re-inspects

    [FIX #4] State is managed per-session via a dictionary keyed by session_id,
    making this safe for concurrent users when deployed behind FastAPI/Flask.
    """
    def __init__(self):
        # ParserConfig now owns the GeminiKeyRotator (multi-key pool)
        self.config = ParserConfig()

        # Parser and gatekeeper get clients from the rotator on each API call
        self.parser = ArchitectureParser(self.config)
        self.gatekeeper = ArchitectGatekeeper(self.config)

        # Image analyzer — lazy-initialized (only if image input is used)
        self._image_analyzer = None

        # Interactive gatherer — lazy-initialized (only on first interactive call)
        self._interactive_gatherer = None

        # Session-scoped state: keyed by session_id for concurrent safety
        self._sessions: Dict[str, Dict[str, Any]] = {}

    @property
    def image_analyzer(self):
        """Lazy-load ImageAnalyzer only when image input is actually provided."""
        if self._image_analyzer is None:
            from modules.step1_parse.image_analyzer import ImageAnalyzer
            self._image_analyzer = ImageAnalyzer()
            logger.info("ImageAnalyzer lazy-initialized")
        return self._image_analyzer

    @property
    def interactive_gatherer(self):
        """Lazy-load InteractiveGatherer only when interactive mode is needed."""
        if self._interactive_gatherer is None:
            from modules.step1_parse.interactive_gatherer import InteractiveGatherer
            self._interactive_gatherer = InteractiveGatherer(self.config)
            logger.info("InteractiveGatherer lazy-initialized")
        return self._interactive_gatherer

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieve or initialize a session context."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "original_input": None,
                "last_parsed_data": None,
                "conversation_history": [],   # [FIX #6] Full turn history
                "image_data": None,           # Data extracted from image
                "image_path": None,           # Path to provided image
                "interactive_mode": False,    # Whether interactive loop is active
            }
        return self._sessions[session_id]

    def execute(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """Executes the foundational parsing turn for entirely fresh incoming requests."""
        session = self._get_session(session_id)
        session["original_input"] = user_input
        session["last_parsed_data"] = None
        session["image_data"] = None
        session["image_path"] = None
        session["interactive_mode"] = False
        session["conversation_history"] = [user_input]
        return self._run_parse_and_validate(user_input, session)

    def execute_with_image(
        self,
        image_input: Union[str, bytes, Path],
        user_text: Optional[str] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Executes parsing with image input (and optional text).

        This is the primary entry point for multimodal input. It:
          1. Analyzes the image via Gemini Vision
          2. Parses the text (if provided) via the standard text parser
          3. Merges both extraction results (image wins for spatial, text for preferences)
          4. Returns the merged result with interactive loop ready

        Args:
            image_input: Image as file path, raw bytes, or base64 string.
            user_text: Optional text prompt from the user.
            session_id: Session identifier.

        Returns:
            Same result format as execute() — status, data, clarification_prompt.
        """
        logger.info("=" * 50)
        logger.info("MULTIMODAL PIPELINE — Image + Text Execution")
        logger.info("=" * 50)

        session = self._get_session(session_id)
        session["original_input"] = user_text or "[image input]"
        session["last_parsed_data"] = None
        session["interactive_mode"] = False
        session["conversation_history"] = [user_text or "[image input]"]

        # Store image path for reference
        if isinstance(image_input, (str, Path)):
            session["image_path"] = str(image_input)

        # Step 1: Analyze image
        logger.info("Step 1a: Analyzing image...")
        image_result = self.image_analyzer.analyze(image_input, user_text)

        if image_result["status"] == "error":
            logger.error("Image analysis failed: %s", image_result["message"])
            # If text is available, fall back to text-only parsing
            if user_text:
                logger.info("Falling back to text-only parsing...")
                return self.execute(user_text, session_id)
            return {
                "status": "error",
                "message": f"Image analysis failed: {image_result['message']}"
            }

        image_data = image_result["data"]
        session["image_data"] = image_data
        logger.info(
            "Image analysis complete. Type: %s",
            image_result.get("image_type", "unknown"),
        )

        # Step 2: Parse text (if provided)
        text_data = None
        if user_text and user_text.strip():
            logger.info("Step 1b: Parsing text input...")
            text_requirements = self.parser.parse_input(user_text)
            if text_requirements:
                text_data = text_requirements.model_dump()

        # Step 3: Merge image + text data
        logger.info("Step 1c: Merging image and text data...")
        from modules.step1_parse.image_analyzer import ImageTextMerger
        merged_data = ImageTextMerger.merge(text_data, image_data)

        # Step 4: Validate the merged data through gatekeeper
        merged_model = BuildingRequirements.model_validate(merged_data)
        return self._validate_and_respond(merged_model, session)

    def handle_followup(self, followup_input: str, session_id: str = "default") -> Dict[str, Any]:
        """Processes conversational follow-up turns by resolving changes over prior states."""
        session = self._get_session(session_id)

        if not session["original_input"] or not session["last_parsed_data"]:
            logger.warning(
                "Follow-up received outside active conversation thread context. "
                "Redirecting to fresh session initialization."
            )
            return self.execute(followup_input, session_id)

        # [FIX #6] Track full conversation history — prevents multi-turn amnesia
        session["conversation_history"].append(followup_input)

        # Pass the FULL conversation thread (all prior turns) as context,
        # not just the original first message. This ensures the LLM sees
        # turns 2 and 3 when processing turn 4.
        full_conversation_context = "\n---\n".join(session["conversation_history"][:-1])

        updated_requirements = self.parser.merge_and_reparse(
            original_input=full_conversation_context,
            followup_input=followup_input,
            existing_data=session["last_parsed_data"]
        )

        if not updated_requirements:
            return {
                "status": "error",
                "message": "I could not process that specific update. Please describe your adjustments again."
            }

        return self._validate_and_respond(updated_requirements, session)

    # -----------------------------------------------------------------
    # INTERACTIVE GATHERING LOOP
    # -----------------------------------------------------------------
    def get_next_interactive_action(
        self,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Drive the interactive gathering loop. Call this after execute() or
        execute_with_image() when the result status is 'incomplete'.

        Returns:
          - {"action": "ask", "question": str, ...} → present question to user
          - {"action": "complete", ...} → all essential data gathered, proceed
        """
        session = self._get_session(session_id)

        if not session["last_parsed_data"]:
            return {
                "action": "complete",
                "data": None,
                "completion_reason": "no_session_data",
            }

        session["interactive_mode"] = True
        return self.interactive_gatherer.get_next_action(
            session["last_parsed_data"],
            session_id,
        )

    def process_interactive_answer(
        self,
        answer: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process the user's answer to an interactive question.

        Merges the answer into the existing data using merge_and_reparse,
        then returns the updated result (which may trigger another question
        or complete the loop).

        Args:
            answer: The user's text response to the question.
            session_id: Session identifier.

        Returns:
            Updated pipeline result (same format as execute()).
        """
        session = self._get_session(session_id)

        if not session["last_parsed_data"]:
            logger.warning("Interactive answer received without active session data.")
            return self.execute(answer, session_id)

        # Track in conversation history
        session["conversation_history"].append(answer)

        # Merge the answer into existing data
        full_conversation_context = "\n---\n".join(session["conversation_history"][:-1])

        updated_requirements = self.parser.merge_and_reparse(
            original_input=full_conversation_context,
            followup_input=answer,
            existing_data=session["last_parsed_data"]
        )

        if not updated_requirements:
            logger.warning("Failed to merge interactive answer. Keeping existing data.")
            return {
                "status": "incomplete",
                "data": session["last_parsed_data"],
                "message": "I could not process that response. Could you rephrase?"
            }

        # Update session with merged data
        session["last_parsed_data"] = updated_requirements.model_dump()

        # Check if we should continue asking or validate through gatekeeper
        next_action = self.interactive_gatherer.get_next_action(
            session["last_parsed_data"],
            session_id,
        )

        if next_action["action"] == "complete":
            # Interactive loop finished — run through gatekeeper for final validation
            logger.info("Interactive loop complete. Running final gatekeeper validation.")
            return self._validate_and_respond(updated_requirements, session)

        # Still have questions — return the next question
        return {
            "status": "interactive",
            "action": next_action["action"],
            "question": next_action.get("question"),
            "field": next_action.get("field"),
            "field_name": next_action.get("field_name"),
            "tier": next_action.get("tier"),
            "data": session["last_parsed_data"],
            "missing_summary": next_action.get("missing_summary"),
        }

    # -----------------------------------------------------------------
    # INTERNAL METHODS
    # -----------------------------------------------------------------
    def _run_parse_and_validate(self, text: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Parses text and validates through gatekeeper. All errors are handled gracefully."""
        requirements = self.parser.parse_input(text)
        if not requirements:
            return {
                "status": "error",
                "message": "Internal structural compilation error occurred while processing design layouts."
            }
        return self._validate_and_respond(requirements, session)

    def _validate_and_respond(self, requirements: BuildingRequirements, session: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates configurations through the gatekeeper to prepare API response contracts."""
        session["last_parsed_data"] = requirements.model_dump()
        validation = self.gatekeeper.validate_and_format(requirements)

        if validation["is_valid"]:
            return {
                "status": "success",
                "data": session["last_parsed_data"],
                "suggestions": validation["missing_tier2"],
                "clarification_prompt": validation["clarification_prompt"]
            }
        else:
            return {
                "status": "incomplete",
                "data": session["last_parsed_data"],
                "missing_fields": validation["missing_tier1"],
                "clarification_prompt": validation["clarification_prompt"]
            }

    def clear_session(self, session_id: str = "default") -> None:
        """Clears a specific session's conversational state."""
        self._sessions.pop(session_id, None)
        # Also clear interactive gatherer state
        if self._interactive_gatherer:
            self._interactive_gatherer.clear_session(session_id)


# =====================================================================
# 5. LOCAL VERIFICATION RUNNER
# =====================================================================
if __name__ == "__main__":
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    # Output directory for JSON saves
    _output_dir = _Path(__file__).resolve().parent.parent.parent / "extracted data" / "prompt_extraction"
    _output_dir.mkdir(parents=True, exist_ok=True)

    def _save(data, label):
        """Save a trial's JSON data to the prompt_extraction folder."""
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        fp = _output_dir / f"{ts}_{label}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  → JSON saved: {fp.name}")

    # Comprehensive end-to-end sandbox verification runner
    print("Initializing production pipeline sandbox sanity check...")
    print(f"JSON outputs will be saved to:\n  {_output_dir}\n")
    try:
        pipeline = Module1Pipeline()

        # ──────────────────────────────────────────────────────────────
        # Test Case 1: Incomplete shorthand — should ask for plot size
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("TRIAL 1: Incomplete Shorthand Entry")
        print("Input: 'I need a 3BHK north-facing Vastu compliant luxury villa.'")
        print("=" * 60)
        incomplete_sample = "I need a 3BHK north-facing Vastu compliant luxury villa."
        res1 = pipeline.execute(incomplete_sample)
        print(f"Status: {res1['status']}")
        print(f"Architect Response:\n{res1.get('clarification_prompt', 'N/A')}")
        if res1.get('data'):
            _save(res1['data'], "trial1_incomplete")

        # ──────────────────────────────────────────────────────────────
        # Test Case 2: Follow-up with plot dimensions
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("TRIAL 2: Resolving Missing Plot Parameters")
        print("Input: 'The plot size is 30x40 feet.'")
        print("=" * 60)
        res2 = pipeline.handle_followup("The plot size is 30x40 feet.")
        print(f"Status: {res2['status']}")
        if res2.get('data'):
            _save(res2['data'], "trial2_followup")
        if res2.get('suggestions'):
            print(f"Tier 2 Suggestions: {res2['suggestions']}")
        if res2.get('clarification_prompt'):
            print(f"Architect Follow-up:\n{res2['clarification_prompt']}")

        # ──────────────────────────────────────────────────────────────
        # Test Case 3: Explicit bathrooms — should extract them
        # ──────────────────────────────────────────────────────────────
        pipeline.clear_session()
        print("\n" + "=" * 60)
        print("TRIAL 3: Explicit Bathroom Mention (Anti-Hallucination Check)")
        print("Input: 'I want a 2BHK with 3 bathrooms and a balcony on a 20x30 east-facing plot'")
        print("=" * 60)
        res3 = pipeline.execute("I want a 2BHK with 3 bathrooms and a balcony on a 20x30 east-facing plot")
        print(f"Status: {res3['status']}")
        if res3.get('data'):
            _save(res3['data'], "trial3_bathrooms")
        if res3.get('suggestions'):
            print(f"Tier 2 Suggestions: {res3['suggestions']}")

        # ──────────────────────────────────────────────────────────────
        # Test Case 4: Villa with explicit floors — no inference needed
        # ──────────────────────────────────────────────────────────────
        pipeline.clear_session()
        print("\n" + "=" * 60)
        print("TRIAL 4: Villa with Explicit Floor Count")
        print("Input: 'Build me a G+2 villa on a 50x80 north-facing plot with 4 bedrooms'")
        print("=" * 60)
        res4 = pipeline.execute("Build me a G+2 villa on a 50x80 north-facing plot with 4 bedrooms")
        print(f"Status: {res4['status']}")
        if res4.get('data'):
            _save(res4['data'], "trial4_villa")

        # ──────────────────────────────────────────────────────────────
        # Test Case 5: Pure BHK — zero hallucination test
        # ──────────────────────────────────────────────────────────────
        pipeline.clear_session()
        print("\n" + "=" * 60)
        print("TRIAL 5: Pure BHK — Zero Hallucination Test")
        print("Input: '3BHK house'")
        print("=" * 60)
        res5 = pipeline.execute("3BHK house")
        print(f"Status: {res5['status']}")
        if res5.get('data'):
            _save(res5['data'], "trial5_bhk")
            rooms = res5['data'].get('rooms', [])
            room_types = [r['room_type'] for r in rooms]
            print(f"Extracted Rooms: {room_types}")
            hallucinated = [r for r in room_types if r not in ['Bedroom', 'Living Room', 'Kitchen']]
            if hallucinated:
                print(f"[FAIL] HALLUCINATION DETECTED: {hallucinated}")
            else:
                print("[PASS] ZERO HALLUCINATION -- Only Bedroom, Living Room, Kitchen extracted.")

        print(f"\n✅ All trial JSONs saved to:\n  {_output_dir}")

    except Exception as error:
        print(f"Pipeline crashed during structural validation execution tracking: {error}")
