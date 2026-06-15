"""
API Key Rotator for Gemini — handles multi-key round-robin rotation
with automatic 429 back-off per key.

Design
------
* Keys are loaded at startup from GEMINI_API_KEY_1, _2, _3, ... env vars.
* Each key gets its own `genai.Client` instance (Google SDK binds key to client).
* On every `get_client()` call the rotator hands out the next available key
  via round-robin.  Keys that recently received a 429 are placed in a cooldown
  window (default 60 s) and skipped until they're eligible again.
* `report_rate_limited(index)` lets callers signal a 429 so the rotator knows
  to cooldown that key.
* Thread-safe via a simple lock.
"""

import os
import time
import logging
import threading
from typing import List, Tuple, Optional

from google import genai

logger = logging.getLogger("PlanGen.KeyRotator")


class KeySlot:
    """Tracks a single API key's client and rate-limit cooldown state."""

    def __init__(self, api_key: str, label: str):
        self.api_key = api_key
        self.label = label
        self.client = genai.Client(api_key=api_key)
        self.cooldown_until: float = 0.0  # epoch time when key becomes usable again

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def set_cooldown(self, seconds: float) -> None:
        self.cooldown_until = time.time() + seconds
        logger.warning(
            "Key [%s] placed in cooldown for %.0f s (until %s)",
            self.label,
            seconds,
            time.strftime("%H:%M:%S", time.localtime(self.cooldown_until)),
        )


class GeminiKeyRotator:
    """
    Round-robin rotator over multiple Gemini API keys.

    Usage
    -----
    rotator = GeminiKeyRotator()
    client, idx = rotator.get_client()       # get next available client
    try:
        response = client.models.generate_content(...)
    except google.api_core.exceptions.ResourceExhausted:
        rotator.report_rate_limited(idx)     # mark key as rate-limited
        client, idx = rotator.get_client()   # try next key
    """

    DEFAULT_COOLDOWN_SECONDS: float = 60.0   # per-key cooldown after a 429

    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self._slots: List[KeySlot] = []
        self._current_index: int = 0
        self._lock = threading.Lock()
        self._cooldown_seconds = cooldown_seconds

        # Auto-discover keys from env: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        idx = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{idx}", "")
            if not key:
                break
            self._slots.append(KeySlot(api_key=key, label=f"KEY_{idx}"))
            idx += 1

        # Fallback: if no numbered keys found, try legacy GEMINI_API_KEY
        if not self._slots:
            legacy_key = os.getenv("GEMINI_API_KEY", "")
            if legacy_key:
                self._slots.append(KeySlot(api_key=legacy_key, label="KEY_LEGACY"))

        if not self._slots:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... "
                "or at minimum GEMINI_API_KEY in your environment."
            )

        logger.info(
            "Key rotator initialized with %d key(s): %s",
            len(self._slots),
            ", ".join(s.label for s in self._slots),
        )

    @property
    def key_count(self) -> int:
        return len(self._slots)

    def get_client(self) -> Tuple[genai.Client, int]:
        """
        Returns the next available (client, slot_index) tuple via round-robin.

        Skips keys that are in cooldown.  If ALL keys are in cooldown, waits
        for the soonest one to become available.
        """
        with self._lock:
            # First pass: look for an immediately-available key
            for _ in range(len(self._slots)):
                slot = self._slots[self._current_index]
                idx = self._current_index
                self._current_index = (self._current_index + 1) % len(self._slots)
                if slot.is_available:
                    logger.debug("Handing out %s", slot.label)
                    return slot.client, idx

            # All keys are in cooldown — wait for the soonest one
            soonest_slot = min(self._slots, key=lambda s: s.cooldown_until)
            wait_time = max(0.0, soonest_slot.cooldown_until - time.time())

        # Release lock while sleeping
        if wait_time > 0:
            logger.warning(
                "All keys in cooldown. Waiting %.1f s for %s ...",
                wait_time,
                soonest_slot.label,
            )
            time.sleep(wait_time)

        # Recurse (now at least one key should be available)
        return self.get_client()

    def report_rate_limited(self, slot_index: int, cooldown_override: Optional[float] = None) -> None:
        """Mark a key as rate-limited so the rotator skips it for a cooldown period."""
        cd = cooldown_override if cooldown_override is not None else self._cooldown_seconds
        with self._lock:
            self._slots[slot_index].set_cooldown(cd)

    def get_any_api_key(self) -> str:
        """Returns the raw API key string of the next available slot (for legacy compat)."""
        _, idx = self.get_client()
        return self._slots[idx].api_key
