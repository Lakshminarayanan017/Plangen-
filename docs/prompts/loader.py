"""
Prompt Loader Utility
=====================
Centralized loader for all system prompts stored as external .md/.txt files
under the docs/prompts/ directory.

Usage in any module:
    from docs.prompts.loader import load_prompt
    prompt = load_prompt("step1_parser_system.md")

Convention:
    - All prompt files live in: docs/prompts/
    - Naming pattern: step{N}_{module_name}_system.md
    - Examples:
        step1_parser_system.md
        step1_gatekeeper_persona.md
        step2_matcher_system.md
        step3_generator_system.md
"""

import os
from pathlib import Path
from functools import lru_cache

# Resolve the prompts directory relative to THIS file's location
_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(filename: str) -> str:
    """
    Load a prompt file from the docs/prompts/ directory and return its contents
    as a single string. Results are cached so repeated calls don't hit disk.

    Args:
        filename: The prompt file name (e.g., 'step1_parser_system.md').

    Returns:
        The full text content of the prompt file.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    filepath = _PROMPTS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {filepath}\n"
            f"Available prompts: {list_available_prompts()}"
        )

    return filepath.read_text(encoding="utf-8").strip()


def list_available_prompts() -> list:
    """List all available prompt files in the prompts directory."""
    return [
        f.name for f in _PROMPTS_DIR.iterdir()
        if f.is_file() and f.suffix in ('.md', '.txt')
    ]
