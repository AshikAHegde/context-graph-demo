"""
Gemini API Key Pool and Failover Client Manager.
Supports a pool of multiple Gemini API keys with automatic fallback and rotation.
If an API key fails (due to quota limits, rate limits 429, auth errors 403, or service errors),
it automatically fails over to the next key until all keys in the pool are exhausted.
"""

import logging
import os
import threading
from typing import Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)


def parse_gemini_api_keys() -> list[str]:
    """
    Extract and clean all available Gemini API keys from environment variables.
    Supports:
      - Comma / semicolon / newline separated keys in GOOGLE_API_KEY or GOOGLE_API_KEYS
      - GEMINI_API_KEY or GEMINI_API_KEYS
      - Numbered variables: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
    """
    raw_keys: list[str] = []

    # Check common env variable names
    for env_var in ["GOOGLE_API_KEYS", "GOOGLE_API_KEY", "GEMINI_API_KEYS", "GEMINI_API_KEY"]:
        val = os.getenv(env_var, "").strip()
        if val:
            for part in val.replace(";", ",").replace("\n", ",").split(","):
                cleaned = part.strip().strip("'\"")
                if cleaned:
                    raw_keys.append(cleaned)

    # Check numbered env variables (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...)
    for key, val in os.environ.items():
        if (key.startswith("GOOGLE_API_KEY_") or key.startswith("GEMINI_API_KEY_")) and val.strip():
            cleaned = val.strip().strip("'\"")
            if cleaned:
                raw_keys.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for k in raw_keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


class GeminiKeyPool:
    """
    Thread-safe Gemini API Key Pool with automatic rotation and failover.
    """

    def __init__(self, keys: Optional[list[str]] = None):
        self._keys = keys if keys is not None else parse_gemini_api_keys()
        self._current_index = 0
        self._clients: dict[str, genai.Client] = {}
        self._lock = threading.Lock()

        if self._keys:
            masked = [self._mask_key(k) for k in self._keys]
            logger.info(f"Initialized GeminiKeyPool with {len(self._keys)} key(s): {', '.join(masked)}")
        else:
            logger.warning("GeminiKeyPool initialized with 0 API keys!")

    @staticmethod
    def _mask_key(key: str) -> str:
        """Mask API key for safe logging (e.g., 'AIzaSy...4xQ')."""
        if len(key) <= 8:
            return "***"
        return f"{key[:6]}...{key[-4:]}"

    @property
    def keys(self) -> list[str]:
        return self._keys

    @property
    def count(self) -> int:
        return len(self._keys)

    def add_key(self, key: str):
        """Add a key dynamically to the pool."""
        cleaned = key.strip().strip("'\"")
        with self._lock:
            if cleaned and cleaned not in self._keys:
                self._keys.append(cleaned)
                logger.info(f"Added API key {self._mask_key(cleaned)} to pool. Total keys: {len(self._keys)}")

    def _get_client_for_key(self, key: str) -> genai.Client:
        """Get or create cached genai.Client for a specific API key."""
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def generate_content(
        self,
        model: str,
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
    ) -> Any:
        """
        Execute generate_content with automatic key failover.
        Iterates through available keys if rate-limited (429), quota exceeded, or errored.
        """
        if not self._keys:
            raise ValueError("No Google Gemini API keys configured in environment.")

        total_keys = len(self._keys)
        errors = []

        with self._lock:
            start_index = self._current_index

        for attempt in range(total_keys):
            key_index = (start_index + attempt) % total_keys
            api_key = self._keys[key_index]
            masked_key = self._mask_key(api_key)

            try:
                client = self._get_client_for_key(api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                # On success, update current active index
                with self._lock:
                    self._current_index = key_index
                return response

            except Exception as e:
                err_msg = f"Key [{key_index + 1}/{total_keys}] {masked_key} failed: {str(e)}"
                logger.warning(f"[GeminiKeyPool Failover] {err_msg}. Rotating to next API key...")
                errors.append(err_msg)

        # If all keys failed
        error_summary = "\n".join(f"  - {err}" for err in errors)
        logger.error(f"[GeminiKeyPool] All {total_keys} Gemini API keys exhausted!\n{error_summary}")
        raise RuntimeError(
            f"All {total_keys} Gemini API keys in the pool failed to generate content:\n{error_summary}"
        )

    def embed_content(
        self,
        model: str,
        contents: Any,
        config: Optional[types.EmbedContentConfig] = None,
    ) -> Any:
        """
        Execute embed_content with automatic key failover.
        Iterates through available keys if rate-limited, quota exceeded, or errored.
        """
        if not self._keys:
            raise ValueError("No Google Gemini API keys configured in environment.")

        total_keys = len(self._keys)
        errors = []

        with self._lock:
            start_index = self._current_index

        for attempt in range(total_keys):
            key_index = (start_index + attempt) % total_keys
            api_key = self._keys[key_index]
            masked_key = self._mask_key(api_key)

            try:
                client = self._get_client_for_key(api_key)
                response = client.models.embed_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                with self._lock:
                    self._current_index = key_index
                return response

            except Exception as e:
                err_msg = f"Key [{key_index + 1}/{total_keys}] {masked_key} failed: {str(e)}"
                logger.warning(f"[GeminiKeyPool Failover] {err_msg}. Rotating to next API key...")
                errors.append(err_msg)

        error_summary = "\n".join(f"  - {err}" for err in errors)
        logger.error(f"[GeminiKeyPool] All {total_keys} Gemini API keys exhausted during embedding!\n{error_summary}")
        raise RuntimeError(
            f"All {total_keys} Gemini API keys in the pool failed to generate embedding:\n{error_summary}"
        )


# Global singleton instance
gemini_pool = GeminiKeyPool()
