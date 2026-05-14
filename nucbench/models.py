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
# Model-identifier validation
# ---------------------------------------------------------------------------

def validate_model_identifier(model: str) -> Tuple[bool, str]:
    """Check that *model* is a plausible local model identifier.

    Catches the most common format mistakes before a network call is made.

    Args:
        model: The raw string the user typed (may be untrimmed).

    Returns:
        ``(True, "")`` when the identifier looks valid, or
        ``(False, human_readable_error)`` otherwise.
    """
    if not model or not model.strip():
        return False, (
            "Model identifier is empty — enter a name such as "
            "'llama3:8b', 'ollama/mistral', or 'openai/phi3'."
        )

    m = model.strip()

    if m.startswith(("http://", "https://")):
        return False, (
            f"'{m}' looks like a URL — paste it into **Server URL** above, "
            "not here.  Model identifier should be a name like 'llama3:8b'."
        )

    if " " in m:
        return False, (
            f"Model identifier must not contain spaces — got: '{m}'.  "
            "Did you mean to use a colon for the tag, e.g. 'llama3:8b'?"
        )

    if m.startswith("/"):
        return False, (
            f"Model identifier must not start with '/' — got: '{m}'.  "
            "Example: 'ollama/llama3:8b' (no leading slash)."
        )

    if m.endswith("/"):
        return False, (
            f"Incomplete identifier '{m}' — add the model name after the "
            "slash, e.g. 'ollama/llama3:8b'."
        )

    if "//" in m:
        return False, (
            f"Model identifier contains consecutive slashes — got: '{m}'.  "
            "Use a single slash, e.g. 'ollama/llama3:8b'."
        )

    return True, ""


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


def get_all_models() -> List[str]:
    """Return all LiteLLM-registered models, regardless of capability.

    Iterates over ``litellm.model_cost`` and returns every model name,
    deduplicated and alphabetically sorted.

    Returns an empty list (instead of raising) when LiteLLM is not installed
    or the registry cannot be read.
    """
    try:
        import litellm
    except ImportError:
        return []

    models: List[str] = []
    seen: set = set()

    try:
        for model_name in litellm.model_cost:
            if model_name not in seen:
                seen.add(model_name)
                models.append(model_name)
    except Exception:  # noqa: BLE001
        pass

    return sorted(models)


def model_supports_vision(model: str) -> bool:
    """Return ``True`` if *model* is registered in LiteLLM as vision-capable.

    Args:
        model: LiteLLM model identifier (e.g. ``"openai/gpt-4o"``).

    Returns:
        ``True`` when the model's metadata contains ``supports_vision: true``,
        ``False`` otherwise (including when LiteLLM is not installed or the
        model is not found in the registry).
    """
    try:
        import litellm
    except ImportError:
        return False

    try:
        info = litellm.model_cost.get(model, {})
        return isinstance(info, dict) and bool(info.get("supports_vision", False))
    except Exception:  # noqa: BLE001
        return False


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


# ---------------------------------------------------------------------------
# Local / open-source model discovery
# ---------------------------------------------------------------------------

def get_local_models(api_base: str = "http://localhost:11434") -> List[str]:
    """Discover models available at a local inference server.

    Tries the Ollama REST API (``/api/tags``) first, then falls back to the
    OpenAI-compatible ``/v1/models`` endpoint used by llama.cpp, vLLM, and
    LM Studio.

    Args:
        api_base: Base URL of the local inference server.

    Returns:
        A sorted list of LiteLLM-compatible model identifiers, e.g.
        ``["ollama/llama3:8b", "ollama/mistral"]`` for Ollama or plain model
        IDs for OpenAI-compatible servers.  Returns an empty list when the
        server is unreachable or reports no models.
    """
    import json as _json
    import urllib.error
    import urllib.request

    base = api_base.rstrip("/")

    # -- Ollama /api/tags ----------------------------------------------------
    try:
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        models = [
            f"ollama/{m['name']}"
            for m in data.get("models", [])
            if m.get("name")
        ]
        if models:
            return sorted(models)
    except Exception:  # noqa: BLE001
        pass

    # -- OpenAI-compatible /v1/models ----------------------------------------
    try:
        req = urllib.request.Request(f"{base}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        models = [f"openai/{m['id']}" for m in data.get("data", []) if m.get("id")]
        if models:
            return sorted(models)
    except Exception:  # noqa: BLE001
        pass

    return []


def _detect_server_prefix(api_base: str) -> str:
    """Return 'ollama' or 'openai' by probing the server."""
    import json as _json
    import urllib.request

    base = api_base.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
            _json.loads(r.read())
        return "ollama"
    except Exception:  # noqa: BLE001
        return "openai"


# Recognised LiteLLM provider prefixes for local inference.
# Anything outside this set is treated as a bare model name and wrapped with
# the openai/ provider so LiteLLM routes it to the correct /v1 endpoint.
_LOCAL_PROVIDER_PREFIXES = frozenset({"ollama", "ollama_chat", "openai"})


def resolve_local_model(model: str, api_base: str) -> Tuple[str, str]:
    """Normalise *(model, api_base)* for a LiteLLM local-inference call.

    All local servers — Ollama, llama.cpp, vLLM, LM Studio, LocalAI, Jan,
    Text Generation WebUI (with OpenAI extension), Koboldcpp, etc. — expose
    an OpenAI-compatible endpoint at ``{base}/v1/chat/completions``.  Using
    LiteLLM's ``openai/`` provider against that endpoint is more reliable than
    the native ``ollama/`` provider path, so this function:

    * converts ``ollama/*`` and ``ollama_chat/*`` models to ``openai/*``;
    * ensures ``/v1`` is appended to *api_base* (idempotent);
    * wraps bare names (no ``/``) or unknown-provider names (e.g.
      ``meta-llama/Llama-3-8b``) with the ``openai/`` prefix so LiteLLM
      forwards the full string as the model name.

    Args:
        model:    User-supplied model identifier.
        api_base: Base URL of the local inference server.

    Returns:
        ``(litellm_model_string, normalised_api_base)`` ready for
        ``litellm.completion``.
    """
    base = api_base.rstrip("/")
    v1_base = base if base.endswith("/v1") else f"{base}/v1"

    if "/" not in model:
        # Bare name — no provider prefix at all.
        return f"openai/{model}", v1_base

    provider = model.split("/", 1)[0]

    if provider in ("ollama", "ollama_chat"):
        # Route through the OpenAI-compatible endpoint every modern Ollama
        # instance exposes; avoids LiteLLM's sometimes-fragile Ollama path.
        name = model.split("/", 1)[1]
        return f"openai/{name}", v1_base

    if provider == "openai":
        return model, v1_base

    # Unknown provider prefix (e.g. "meta-llama/Llama-3-8b" typed as a model
    # name for vLLM).  Treat the whole string as the model name under openai/.
    return f"openai/{model}", v1_base


def test_local_connection(
    model: str,
    api_base: str = "http://localhost:11434",
) -> Tuple[bool, str]:
    """Test a local inference server with a minimal completion call.

    Supports Ollama, llama.cpp, vLLM, LM Studio, LocalAI, Jan, and any other
    server that exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Args:
        model:    Model identifier (e.g. ``"llama3"``, ``"ollama/llama3:8b"``,
                  ``"openai/mistral"``).
        api_base: Base URL of the local inference server (e.g.
                  ``"http://localhost:11434"`` for Ollama or
                  ``"http://localhost:8080"`` for llama.cpp / vLLM).

    Returns:
        A ``(success, message)`` tuple.
    """
    try:
        import litellm
    except ImportError:
        return False, "LiteLLM is not installed. Run: pip install litellm"

    resolved_model, resolved_base = resolve_local_model(model, api_base)

    try:
        litellm.completion(
            model=resolved_model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            api_base=resolved_base,
            api_key="local",  # placeholder — Ollama and most local servers ignore this
            max_tokens=5,
        )
        note = (
            f" (routed as '{resolved_model}' → {resolved_base})"
            if resolved_model != model or resolved_base != api_base.rstrip("/")
            else ""
        )
        return True, f"Connected to '{model}' at {api_base}.{note}"

    except Exception as exc:  # noqa: BLE001
        err_lower = str(exc).lower()
        if any(w in err_lower for w in ("connection refused", "refused", "unreachable", "cannot connect")):
            return (
                False,
                f"Cannot reach {api_base} — is the server running? ({type(exc).__name__})",
            )
        # Surface the real error so users can diagnose model-name typos,
        # missing /v1 support, auth issues, etc.
        return False, f"Connection test failed ({type(exc).__name__}): {exc}"
