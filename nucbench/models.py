"""
nucbench/models.py
------------------
Model discovery and API-key validation utilities for NucBench.

All interactions with LiteLLM are isolated here so the rest of the
application can be tested without a live provider connection.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Vision-model discovery
# ---------------------------------------------------------------------------

def get_vision_models() -> List[str]:
    """Return every LiteLLM-registered model that supports image inputs.

    Iterates over ``litellm.model_cost`` — the canonical per-model metadata
    registry — and retains entries where ``supports_vision`` is ``True``.
    The result is deduplicated and alphabetically sorted.

    Returns an empty list (instead of raising) when LiteLLM is not installed
    or the registry cannot be read, so the UI can degrade gracefully.
    """
    try:
        import litellm  # deferred so module loads even without litellm
    except ImportError:
        return []

    vision_models: List[str] = []
    seen: set = set()

    try:
        for model_name, model_info in litellm.model_cost.items():
            if (
                isinstance(model_info, dict)
                and model_info.get("supports_vision", False)
                and model_name not in seen
            ):
                seen.add(model_name)
                vision_models.append(model_name)
    except Exception:  # noqa: BLE001
        pass

    return sorted(vision_models)


# ---------------------------------------------------------------------------
# API-key validation
# ---------------------------------------------------------------------------

def validate_api_key(model: str, api_key: str) -> Tuple[bool, str]:
    """Validate an API key by making a minimal, low-cost test completion call.

    A ``RateLimitError`` is treated as a *successful* validation because the
    provider authenticated the request before throttling it.

    Args:
        model:   LiteLLM model identifier (e.g. ``"openai/gpt-4o"``).
        api_key: Provider API key supplied by the user.

    Returns:
        A ``(success, message)`` tuple where *success* is ``True`` when the
        key appears to be valid and *message* is a human-readable explanation.
    """
    try:
        import litellm
    except ImportError:
        return False, "LiteLLM is not installed. Run: pip install litellm"

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            api_key=api_key,
            max_tokens=5,
        )
        return True, "API key validated successfully."

    except litellm.AuthenticationError as exc:
        return False, f"Authentication failed — check your API key. ({exc})"

    except litellm.RateLimitError:
        # Authenticated but over quota; the key itself is valid.
        return True, "API key is valid (rate-limited, but authenticated)."

    except litellm.APIError as exc:
        return False, f"API error during validation: {exc}"

    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error during validation: {exc}"
