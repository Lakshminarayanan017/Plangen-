"""
Image Analyzer Module — Gemini Vision Multimodal Analysis Engine
================================================================
Production-grade module for extracting architectural data from plot/plan images
using Google Gemini's multimodal vision capabilities.

Supports:
  - Hand-drawn sketches (pencil/pen on paper)
  - Professional architectural drawings (CAD exports, dimensioned plans)
  - Existing floor plans (user wants modifications)
  - Site photographs (physical plot photos)
  - Plot maps / survey plans
  - Digital screenshots (from apps, websites, tools)
  - Incomplete/partial plans (blank plots, single floors)

Design
------
* Uses a DEDICATED Gemini API key (GEMINI_IMAGE_API_KEY) to avoid
  contention with the text parser's key rotation pool.
* Accepts input as file path, raw bytes, or base64 string — handles
  all three for flexibility across CLI, API, and web upload flows.
* Outputs the same BuildingRequirements Pydantic schema as the text parser,
  making downstream merge trivial.
* Includes robust retry logic with exponential backoff for transient failures.
* Validates and normalizes image input before sending to the API.
"""

import os
import io
import json
import time
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from pydantic import ValidationError
from dotenv import load_dotenv

# Load environment variables
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_env_path = os.path.join(_project_root, "sources", ".env")
load_dotenv(_env_path)

# Official Google GenAI SDK
from google import genai
from google.genai import types

# Project models
from models import BuildingRequirements

logger = logging.getLogger("PlanGen.ImageAnalyzer")


# =====================================================================
# 1. CONFIGURATION
# =====================================================================
class ImageAnalyzerConfig:
    """
    Configuration for the Image Analyzer module.

    Uses a dedicated API key (GEMINI_IMAGE_API_KEY) separate from the
    text parser's key rotation pool. This prevents vision API calls
    from consuming text-parsing quota and vice versa.
    """

    # Gemini model for vision tasks — Flash supports multimodal input
    VISION_MODEL: str = "gemini-2.5-flash"

    # Conservative temperature for precise extraction
    TEMPERATURE: float = 0.1

    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_BASE: float = 4.0

    # Maximum image size in bytes (20 MB — Gemini's limit)
    MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024

    # Supported MIME types for Gemini Vision
    SUPPORTED_MIME_TYPES: set = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }

    # Class-level prompt cache
    _IMAGE_SYSTEM_INSTRUCTION: Optional[str] = None

    def __init__(self):
        # Load dedicated image analysis API key
        self._api_key = os.getenv("GEMINI_IMAGE_API_KEY", "")

        if not self._api_key:
            # Fallback: try the legacy key if dedicated key is missing
            self._api_key = os.getenv("GEMINI_API_KEY", "")
            if self._api_key:
                logger.warning(
                    "GEMINI_IMAGE_API_KEY not found. Falling back to GEMINI_API_KEY. "
                    "For production, set a dedicated GEMINI_IMAGE_API_KEY."
                )
            else:
                raise ValueError(
                    "No API key found for image analysis. "
                    "Set GEMINI_IMAGE_API_KEY in your environment."
                )

        # Create a dedicated Gemini client for vision
        self.client = genai.Client(api_key=self._api_key)
        logger.info("ImageAnalyzerConfig initialized with dedicated vision API key")

        # Load system prompt (cached at class level)
        if ImageAnalyzerConfig._IMAGE_SYSTEM_INSTRUCTION is None:
            from docs.prompts.loader import load_prompt

            ImageAnalyzerConfig._IMAGE_SYSTEM_INSTRUCTION = load_prompt(
                "step1_image_analyzer_system.md"
            )
            logger.info(
                "Loaded image analyzer system instruction from "
                "docs/prompts/step1_image_analyzer_system.md"
            )

    @property
    def IMAGE_SYSTEM_INSTRUCTION(self) -> str:
        """Access the cached image analysis system instruction."""
        return ImageAnalyzerConfig._IMAGE_SYSTEM_INSTRUCTION


# =====================================================================
# 2. IMAGE INPUT VALIDATOR & PREPROCESSOR
# =====================================================================
class ImagePreprocessor:
    """
    Validates and normalizes image input into a format suitable for
    the Gemini Vision API.

    Accepts three input modes:
      1. File path (str or Path) — reads and validates the file
      2. Raw bytes — validates MIME type from magic bytes
      3. Base64 string — decodes and validates
    """

    # Magic byte signatures for image format detection
    _MAGIC_BYTES = {
        b"\xff\xd8\xff": "image/jpeg",
        b"\x89PNG": "image/png",
        b"RIFF": "image/webp",  # WebP starts with RIFF
        b"GIF8": "image/gif",
        b"BM": "image/bmp",
        b"II": "image/tiff",  # TIFF little-endian
        b"MM": "image/tiff",  # TIFF big-endian
    }

    @staticmethod
    def detect_mime_type(data: bytes) -> Optional[str]:
        """Detect MIME type from the first few bytes of image data."""
        for magic, mime in ImagePreprocessor._MAGIC_BYTES.items():
            if data[: len(magic)] == magic:
                return mime
        return None

    @staticmethod
    def validate_and_load(
        image_input: Union[str, bytes, Path],
        config: ImageAnalyzerConfig,
    ) -> Tuple[bytes, str]:
        """
        Validates and loads image input into (bytes, mime_type) tuple.

        Args:
            image_input: File path, raw bytes, or base64 string.
            config: ImageAnalyzerConfig for validation constraints.

        Returns:
            Tuple of (image_bytes, mime_type).

        Raises:
            ValueError: If input is invalid, too large, or unsupported format.
            FileNotFoundError: If file path doesn't exist.
        """
        image_bytes: bytes
        mime_type: str

        # --- MODE 1: File path ---
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)

            # Check if it's a base64 string (not a file path)
            if isinstance(image_input, str) and not path.exists():
                # Try to decode as base64
                try:
                    image_bytes = base64.b64decode(image_input)
                    detected_mime = ImagePreprocessor.detect_mime_type(image_bytes)
                    if detected_mime is None:
                        raise ValueError(
                            "Could not determine image format from base64 data. "
                            "Ensure the data is a valid image."
                        )
                    mime_type = detected_mime
                    logger.info("Image loaded from base64 string (%d bytes)", len(image_bytes))

                except Exception as e:
                    if "Could not determine" in str(e):
                        raise
                    raise FileNotFoundError(
                        f"Image path not found and input is not valid base64: {image_input}"
                    ) from e
            else:
                # It's a real file path
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")

                if not path.is_file():
                    raise ValueError(f"Path is not a file: {path}")

                # Detect MIME type from extension first
                mime_type_from_ext = mimetypes.guess_type(str(path))[0]
                image_bytes = path.read_bytes()

                # Also detect from magic bytes for verification
                detected_mime = ImagePreprocessor.detect_mime_type(image_bytes)

                # Prefer magic-byte detection, fall back to extension
                mime_type = detected_mime or mime_type_from_ext or "image/jpeg"

                logger.info(
                    "Image loaded from file: %s (%d bytes, %s)",
                    path.name,
                    len(image_bytes),
                    mime_type,
                )

        # --- MODE 2: Raw bytes ---
        elif isinstance(image_input, bytes):
            image_bytes = image_input
            detected_mime = ImagePreprocessor.detect_mime_type(image_bytes)
            if detected_mime is None:
                raise ValueError(
                    "Could not determine image format from raw bytes. "
                    "Ensure the data starts with valid image headers."
                )
            mime_type = detected_mime
            logger.info("Image loaded from raw bytes (%d bytes, %s)", len(image_bytes), mime_type)

        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input).__name__}. "
                f"Expected str (path/base64), bytes, or Path."
            )

        # --- VALIDATION ---
        # Check size
        if len(image_bytes) > config.MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image size ({len(image_bytes) / (1024 * 1024):.1f} MB) exceeds "
                f"maximum allowed size ({config.MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f} MB)."
            )

        # Check MIME type
        if mime_type not in config.SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported image format: {mime_type}. "
                f"Supported formats: {', '.join(config.SUPPORTED_MIME_TYPES)}"
            )

        # Check minimum size (at least 1KB — likely corrupted if smaller)
        if len(image_bytes) < 1024:
            logger.warning(
                "Image is very small (%d bytes) — may be corrupted or empty.",
                len(image_bytes),
            )

        return image_bytes, mime_type


# =====================================================================
# 3. CORE IMAGE ANALYZER
# =====================================================================
class ImageAnalyzer:
    """
    Production-grade Gemini Vision analyzer for architectural images.

    Extracts building requirements from plot photos, floor plan sketches,
    site photographs, and architectural drawings into the standard
    BuildingRequirements JSON schema.

    Usage:
        analyzer = ImageAnalyzer()

        # From file path
        result = analyzer.analyze("path/to/plot_sketch.jpg")

        # From file path + user text
        result = analyzer.analyze("path/to/sketch.jpg", user_text="3BHK villa")

        # From raw bytes (e.g., web upload)
        result = analyzer.analyze(uploaded_file.read())

        # From base64 string
        result = analyzer.analyze(base64_string)
    """

    def __init__(self, config: Optional[ImageAnalyzerConfig] = None):
        """
        Initialize the ImageAnalyzer.

        Args:
            config: Optional ImageAnalyzerConfig. Creates default if None.
        """
        self.config = config or ImageAnalyzerConfig()
        self.preprocessor = ImagePreprocessor()
        logger.info("ImageAnalyzer initialized and ready for multimodal analysis")

    @staticmethod
    def _clean_response_text(raw_text: str) -> str:
        """
        Strip markdown code fences that Gemini occasionally wraps
        around JSON output, even with response_mime_type='application/json'.
        """
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _build_content_parts(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_text: Optional[str] = None,
    ) -> list:
        """
        Build the multimodal content parts for the Gemini API call.

        The order matters: image first, then text context. This tells
        Gemini to prioritize visual analysis while using text as supplementary.
        """
        parts = []

        # Image part — always first
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        parts.append(image_part)

        # Text context part — provides additional context from user
        if user_text and user_text.strip():
            context_prompt = (
                "The client has also provided this text description alongside the image. "
                "Use BOTH the image and this text to extract building requirements. "
                "When they conflict, follow the CONFLICT RESOLUTION rules in your instructions.\n\n"
                f"CLIENT'S TEXT: \"{user_text.strip()}\""
            )
            parts.append(context_prompt)
        else:
            # No text — image-only analysis
            parts.append(
                "Analyze this architectural image and extract all visible building "
                "requirements into the BuildingRequirements JSON schema. The image is "
                "the ONLY source of information — extract everything you can see."
            )

        return parts

    def analyze(
        self,
        image_input: Union[str, bytes, Path],
        user_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze an image and extract building requirements.

        Args:
            image_input: Image as file path, raw bytes, or base64 string.
            user_text: Optional accompanying text from the user.

        Returns:
            Dict containing the extraction result:
            {
                "status": "success" | "error",
                "data": BuildingRequirements dict (if success),
                "image_type": str description of detected image type,
                "message": str error message (if error),
            }
        """
        logger.info("=" * 50)
        logger.info("IMAGE ANALYSIS — Starting multimodal extraction")
        logger.info("=" * 50)

        # Step 1: Validate and load image
        try:
            image_bytes, mime_type = self.preprocessor.validate_and_load(
                image_input, self.config
            )
        except (FileNotFoundError, ValueError, TypeError) as e:
            logger.error("Image validation failed: %s", e)
            return {
                "status": "error",
                "data": None,
                "image_type": "unknown",
                "message": f"Image validation error: {str(e)}",
            }

        # Step 2: Build multimodal content parts
        content_parts = self._build_content_parts(image_bytes, mime_type, user_text)

        # Step 3: Call Gemini Vision API with retry logic
        parsed_data = self._call_vision_api(content_parts)

        if parsed_data is None:
            return {
                "status": "error",
                "data": None,
                "image_type": "unknown",
                "message": "Failed to extract data from image after all retries.",
            }

        # Step 4: Extract image type from the parsed data
        image_type = "unknown"
        if parsed_data.plot_context and parsed_data.plot_context.image_source_notes:
            image_type = parsed_data.plot_context.image_source_notes

        logger.info("IMAGE ANALYSIS — Extraction complete. Image type: %s", image_type)

        return {
            "status": "success",
            "data": parsed_data.model_dump(),
            "image_type": image_type,
            "message": "Image analysis successful.",
        }

    def _call_vision_api(self, content_parts: list) -> Optional[BuildingRequirements]:
        """
        Execute the Gemini Vision API call with exponential backoff retry.

        Args:
            content_parts: List of multimodal content parts (image + text).

        Returns:
            BuildingRequirements object or None on failure.
        """
        for attempt in range(1, self.config.MAX_RETRIES + 1):
            logger.info(
                "Vision API attempt %d/%d using model %s",
                attempt,
                self.config.MAX_RETRIES,
                self.config.VISION_MODEL,
            )

            try:
                response = self.config.client.models.generate_content(
                    model=self.config.VISION_MODEL,
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        system_instruction=self.config.IMAGE_SYSTEM_INSTRUCTION,
                        temperature=self.config.TEMPERATURE,
                        response_mime_type="application/json",
                        response_schema=BuildingRequirements,
                    ),
                )

                # Clean and validate response
                cleaned_text = self._clean_response_text(response.text)
                parsed_data = BuildingRequirements.model_validate_json(cleaned_text)

                logger.info("Vision API — Successfully parsed response on attempt %d", attempt)
                return parsed_data

            except (ValidationError, json.JSONDecodeError) as ve:
                logger.warning(
                    "Schema/JSON validation failed on attempt %d/%d: %s",
                    attempt,
                    self.config.MAX_RETRIES,
                    ve,
                )
                if attempt < self.config.MAX_RETRIES:
                    delay = self.config.RETRY_DELAY_BASE
                    logger.info("Retrying in %.1f seconds...", delay)
                    time.sleep(delay)
                else:
                    logger.error(
                        "Exhausted retries due to persistent schema validation failures."
                    )
                    return None

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or (
                    "resource" in error_str and "exhausted" in error_str
                )

                if is_rate_limit:
                    logger.warning(
                        "429 rate-limit hit on vision API (attempt %d): %s",
                        attempt,
                        e,
                    )
                    # Exponential backoff for rate limits
                    delay = self.config.RETRY_DELAY_BASE * (2 ** (attempt - 1))
                    logger.info("Rate-limited. Waiting %.1f seconds...", delay)
                    time.sleep(delay)
                    continue

                # Non-429 API error
                logger.warning(
                    "Vision API failure on attempt %d/%d: %s",
                    attempt,
                    self.config.MAX_RETRIES,
                    e,
                )
                if attempt < self.config.MAX_RETRIES:
                    delay = self.config.RETRY_DELAY_BASE ** attempt
                    logger.info("Retrying in %.1f seconds...", delay)
                    time.sleep(delay)
                else:
                    logger.error("Max retry limits exhausted on vision API.")
                    return None

        return None


# =====================================================================
# 4. IMAGE-TEXT DATA MERGER
# =====================================================================
class ImageTextMerger:
    """
    Merges extraction results from image analysis and text parsing
    into a single unified BuildingRequirements dict.

    Conflict Resolution Priority:
      - TEXT wins for: rooms, number_of_floors, vastu, parking, style, building_type
      - IMAGE wins for: plot_dimensions, plot_context.shape, road_facing_sides, setbacks
      - MERGE for: rooms (union), additional_notes (concatenate), specific_requirements
    """

    @staticmethod
    def merge(
        text_data: Optional[Dict[str, Any]],
        image_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge text-parsed and image-parsed BuildingRequirements dicts.

        Args:
            text_data: Parsed output from text parser (may be None).
            image_data: Parsed output from image analyzer (may be None).

        Returns:
            Unified BuildingRequirements dict.
        """
        # Handle cases where one source is missing
        if text_data is None and image_data is None:
            return BuildingRequirements().model_dump()

        if text_data is None:
            logger.info("Merge: Image-only input — using image data directly")
            return image_data

        if image_data is None:
            logger.info("Merge: Text-only input — using text data directly")
            return text_data

        logger.info("Merge: Both image and text data present — merging...")

        # Start with image data as base (image wins for spatial data)
        merged = json.loads(json.dumps(image_data))  # Deep copy

        # --- PLOT DIMENSIONS: Image wins (more reliable measurements) ---
        # But if text explicitly provides dimensions and image doesn't, use text
        if merged.get("plot_dimensions") is None and text_data.get("plot_dimensions"):
            merged["plot_dimensions"] = text_data["plot_dimensions"]
        elif merged.get("plot_dimensions") and text_data.get("plot_dimensions"):
            # Both have dimensions — image wins unless text is more complete
            img_dims = merged["plot_dimensions"]
            txt_dims = text_data["plot_dimensions"]
            if img_dims.get("length") is None and txt_dims.get("length") is not None:
                img_dims["length"] = txt_dims["length"]
            if img_dims.get("width") is None and txt_dims.get("width") is not None:
                img_dims["width"] = txt_dims["width"]
            if img_dims.get("total_area_sqft") is None and txt_dims.get("total_area_sqft"):
                img_dims["total_area_sqft"] = txt_dims["total_area_sqft"]

        # --- PLOT CONTEXT: Image wins for spatial, merge provenance ---
        if merged.get("plot_context") is None and text_data.get("plot_context"):
            merged["plot_context"] = text_data["plot_context"]
        elif merged.get("plot_context") and text_data.get("plot_context"):
            img_ctx = merged["plot_context"]
            txt_ctx = text_data["plot_context"]
            # Image wins for shape if detected
            if img_ctx.get("shape") is None:
                img_ctx["shape"] = txt_ctx.get("shape")
            # Image wins for road_facing_sides if detected
            if not img_ctx.get("road_facing_sides") and txt_ctx.get("road_facing_sides"):
                img_ctx["road_facing_sides"] = txt_ctx["road_facing_sides"]
            # Text wins for north_direction if image doesn't have compass
            if img_ctx.get("north_direction") is None:
                img_ctx["north_direction"] = txt_ctx.get("north_direction")
            # Text wins for entrance_side
            if img_ctx.get("entrance_side") is None:
                img_ctx["entrance_side"] = txt_ctx.get("entrance_side")

        # --- SETBACKS: Image wins (measured from drawing) ---
        if merged.get("setbacks") is None and text_data.get("setbacks"):
            merged["setbacks"] = text_data["setbacks"]

        # --- ROOMS: Text wins, but merge unique rooms from image ---
        text_rooms = text_data.get("rooms", [])
        image_rooms = merged.get("rooms", [])

        if text_rooms:
            # Text rooms take priority
            merged_rooms = list(text_rooms)
            text_room_types = {r["room_type"] for r in text_rooms}

            # Add image-only rooms (rooms visible in image but not mentioned in text)
            for img_room in image_rooms:
                if img_room["room_type"] not in text_room_types:
                    # Mark as image-sourced
                    if img_room.get("specific_requirements"):
                        img_room["specific_requirements"] += "; from image"
                    else:
                        img_room["specific_requirements"] = "from image"
                    merged_rooms.append(img_room)

            merged["rooms"] = merged_rooms
        elif image_rooms:
            # Only image has rooms
            merged["rooms"] = image_rooms

        # --- TEXT WINS for preference fields ---
        text_pref_fields = [
            "number_of_floors",
            "vastu_compliant",
            "parking_type",
            "include_furniture",
            "architectural_style",
            "building_type",
        ]
        for field in text_pref_fields:
            if text_data.get(field) is not None:
                merged[field] = text_data[field]

        # --- ADDITIONAL NOTES: Concatenate both ---
        text_notes = text_data.get("additional_notes", "")
        image_notes = merged.get("additional_notes", "")
        combined_notes = "; ".join(filter(None, [image_notes, text_notes]))
        merged["additional_notes"] = combined_notes if combined_notes else None

        logger.info("Merge complete — unified data assembled")
        return merged


# =====================================================================
# 5. LOCAL VERIFICATION RUNNER
# =====================================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("IMAGE ANALYZER — Production Verification")
    print("=" * 60)

    try:
        # Initialize
        analyzer = ImageAnalyzer()
        print("[OK] ImageAnalyzer initialized successfully")
        print(f"[OK] Vision model: {analyzer.config.VISION_MODEL}")
        print(f"[OK] System prompt loaded: {len(analyzer.config.IMAGE_SYSTEM_INSTRUCTION)} chars")

        # Test preprocessor with a dummy check
        print("\n--- Preprocessor Validation Tests ---")
        try:
            ImagePreprocessor.validate_and_load("nonexistent_file.jpg", analyzer.config)
            print("[FAIL] Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("[PASS] FileNotFoundError raised for missing file")

        try:
            ImagePreprocessor.validate_and_load(b"not_an_image", analyzer.config)
            print("[FAIL] Should have raised ValueError")
        except ValueError:
            print("[PASS] ValueError raised for invalid bytes")

        try:
            ImagePreprocessor.validate_and_load(12345, analyzer.config)
            print("[FAIL] Should have raised TypeError")
        except TypeError:
            print("[PASS] TypeError raised for invalid input type")

        # Test merger
        print("\n--- ImageTextMerger Tests ---")
        merger = ImageTextMerger()

        # Test: merge with only text
        text_only = {"plot_dimensions": {"length": 40, "width": 30, "unit": "ft"}, "rooms": []}
        result = merger.merge(text_only, None)
        assert result["plot_dimensions"]["length"] == 40
        print("[PASS] Text-only merge works")

        # Test: merge with only image
        image_only = {"plot_dimensions": {"length": 50, "width": 40, "unit": "ft"}, "rooms": []}
        result = merger.merge(None, image_only)
        assert result["plot_dimensions"]["length"] == 50
        print("[PASS] Image-only merge works")

        # Test: merge with both (image dims should win)
        result = merger.merge(text_only, image_only)
        assert result["plot_dimensions"]["length"] == 50  # Image wins
        print("[PASS] Image dimensions take priority in merge")

        # Test: text preferences win
        text_with_vastu = {
            "plot_dimensions": None,
            "rooms": [],
            "vastu_compliant": True,
            "number_of_floors": 2,
        }
        image_with_vastu = {
            "plot_dimensions": {"length": 40, "width": 30, "unit": "ft"},
            "rooms": [],
            "vastu_compliant": False,
            "number_of_floors": 1,
        }
        result = merger.merge(text_with_vastu, image_with_vastu)
        assert result["vastu_compliant"] is True  # Text wins
        assert result["number_of_floors"] == 2  # Text wins
        assert result["plot_dimensions"]["length"] == 40  # Image wins
        print("[PASS] Text preferences override image preferences")

        print("\n" + "=" * 60)
        print("ALL VERIFICATION CHECKS PASSED")
        print("=" * 60)

        # If an image path is provided as CLI argument, run a real analysis
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            user_text = sys.argv[2] if len(sys.argv) > 2 else None

            print(f"\n--- Live Analysis: {image_path} ---")
            if user_text:
                print(f"User text: {user_text}")

            result = analyzer.analyze(image_path, user_text)
            print(f"\nStatus: {result['status']}")
            print(f"Image Type: {result['image_type']}")
            if result["data"]:
                print(f"Extracted Data:\n{json.dumps(result['data'], indent=2)}")
            else:
                print(f"Error: {result['message']}")

    except Exception as error:
        print(f"\n[FATAL] Verification crashed: {error}")
        import traceback
        traceback.print_exc()
