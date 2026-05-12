"""
nucbench/prompts.py
-------------------
Prompt templates and message-builder helpers for all three NucBench tasks.

Keeping prompts in one place makes it easy to audit, version, or swap them
without touching task execution logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union


# ---------------------------------------------------------------------------
# Task 1 — Two-phase flow regime image classification
# ---------------------------------------------------------------------------

#: System message that establishes the classification task and regime definitions.
FLOW_REGIME_SYSTEM = (
    "You are an expert in two-phase flow regime identification. "
    "You will be shown images from a vertical two-phase flow experiment and must "
    "classify each image into exactly one of four flow regimes:\n\n"
    "- Bubbly: Small, dispersed gas bubbles distributed throughout the liquid. "
    "Bubbles are roughly spherical and much smaller than the pipe diameter.\n"
    "- Slug: Large bullet-shaped gas pockets (Taylor bubbles) that nearly fill "
    "the pipe cross-section, separated by liquid slugs that may contain small bubbles.\n"
    "- Churn: Highly chaotic and oscillatory flow with large, distorted gas "
    "structures and heavy liquid entrainment. An unstable transition between slug and annular.\n"
    "- Taylor Bubble: A single large, well-defined elongated gas bubble occupying "
    "most of the pipe cross-section, with a thin liquid film on the wall and a "
    "liquid plug below.\n\n"
    "The experimental setup is a 140 ft tall vertical flow loop with a "
    "5.5-inch inner diameter pipe and a 2-3/8-inch drill pipe forming the annulus. "
    "Working fluids are air-water or CO2-water mixtures."
)

#: User instruction sent with each image.
FLOW_REGIME_INSTRUCTION = (
    "Classify the two-phase flow regime shown in this image. "
    "Choose exactly one of: Bubbly, Slug, Churn, Taylor Bubble. "
    "Respond with only the flow regime name, nothing else."
)


# ---------------------------------------------------------------------------
# Tasks 2 & 3 — Exam questions (qualitative and quantitative)
# ---------------------------------------------------------------------------

#: Instruction appended to open-ended (non-MCQ) prompts to elicit a confidence score.
OPEN_ENDED_CONFIDENCE_SUFFIX = (
    "\n\nFormat requirement: Conclude your response with a confidence score "
    "formatted exactly as [Confidence: X%]."
)


#: Template for qualitative (concept / multiple-choice) questions.
_QUALITATIVE_TEMPLATE = (
    "You are an undergraduate nuclear engineering student taking an exam on "
    "reactor physics, thermal-hydraulics, and nuclear fuel cycles. You are to "
    "answer the following question directly and accurately.\n\n{question}"
)

#: Template for quantitative (calculation) questions.
_QUANTITATIVE_TEMPLATE = (
    "You are an undergraduate nuclear engineering student taking an exam on "
    "reactor physics, thermal-hydraulics, and nuclear fuel cycles. You are to "
    "answer the following question directly and accurately with the final "
    "calculated value only (number + units only, no text or explanation).\n\n"
    "{question}"
)


def build_exam_prompt(question_text: str, question_type: str, is_mcq: bool = True) -> str:
    """Select and render the correct exam prompt template.

    Args:
        question_text: Full text of the exam question (may include answer
                       choices such as ``A) … B) …``).
        question_type: ``"Qualitative"`` or ``"Quantitative"`` (case-insensitive).
        is_mcq:        If ``False``, appends the open-ended confidence-score
                       format requirement to the prompt.

    Returns:
        Formatted prompt string ready to pass to the LLM.
    """
    if question_type.strip().lower() == "quantitative":
        prompt = _QUANTITATIVE_TEMPLATE.format(question=question_text)
    else:
        prompt = _QUALITATIVE_TEMPLATE.format(question=question_text)
    if not is_mcq:
        prompt += OPEN_ENDED_CONFIDENCE_SUFFIX
    return prompt


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

ContentBlock = Dict[str, Any]
MessageContent = Union[str, List[ContentBlock]]


def build_flow_regime_message(image_b64: str, mime_type: str) -> List[Dict[str, Any]]:
    """Build the LiteLLM messages list for a flow-regime classification call.

    Uses a two-turn structure: a system message that describes the four flow
    regimes, followed by a user message containing the image and classification
    instruction.  This ensures the model has regime definitions in context
    before it sees the image.

    Args:
        image_b64: Base64-encoded image bytes.
        mime_type: MIME type string, e.g. ``"image/png"``.

    Returns:
        A ``messages`` list suitable for ``litellm.completion()``.
    """
    return [
        {
            "role": "system",
            "content": FLOW_REGIME_SYSTEM,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
                {"type": "text", "text": FLOW_REGIME_INSTRUCTION},
            ],
        },
    ]


def build_exam_message(
    question_text: str,
    question_type: str,
    inline_images: List[tuple],  # list of (b64_str, mime_type)
    is_mcq: bool = True,
) -> List[Dict[str, Any]]:
    """Build the LiteLLM messages list for an exam question.

    If the question has associated images (stored in its ``"images"`` field),
    they are appended as image_url blocks after the prompt text.

    Args:
        question_text: Full question prompt string.
        question_type: ``"Qualitative"`` or ``"Quantitative"``.
        inline_images: List of ``(base64_string, mime_type)`` tuples for any
                       images embedded in the question.  Pass ``[]`` if none.
        is_mcq:        Forwarded to :func:`build_exam_prompt`; when ``False``
                       the confidence-score requirement is appended.

    Returns:
        A ``messages`` list suitable for ``litellm.completion()``.
    """
    prompt_text = build_exam_prompt(question_text, question_type, is_mcq=is_mcq)

    if not inline_images:
        # Plain-text question — use a simple string content for wider compat.
        return [{"role": "user", "content": prompt_text}]

    # Question includes one or more images — use multi-part content.
    content_parts: List[ContentBlock] = [{"type": "text", "text": prompt_text}]
    for b64, mime in inline_images:
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return [{"role": "user", "content": content_parts}]
