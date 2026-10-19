"""
gemini_engine.py
----------------
Handles all interaction with the Gemini API, including:
  - Structured output generation (FlashcardBatch via Pydantic)
  - System prompt injection
  - MathJax / Cloze regex sanitizer
"""

import re
from typing import List, Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class AnkiCard(BaseModel):
    card_type: Literal["Basic", "Cloze"]
    front: str
    back: str
    intuition: str
    source_reference: str
    tags: List[str]


class FlashcardBatch(BaseModel):
    deck_suggestion: str
    cards: List[AnkiCard]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert flashcard engineer and subject-matter tutor. Your job is to extract high-yield, atomic Anki flashcards from uploaded textbook pages, PDFs, or screenshots while strictly obeying the user's custom instructions.

### CORE FLASHCARD PRINCIPLES:
1. Minimum Information Principle:
   - Every card must test ONE atomic concept, relationship, condition, or formula.
   - Never put multi-paragraph summaries on a single card.
   - Deconstruct complex comparison tables, multi-step proofs, or lists into multiple distinct, self-contained cards.
2. Card Type Selection:
   - Use "Cloze" for definitions, key terms, necessary/sufficient conditions, formulas, or short enumerations. Format strictly using Anki syntax: {{c1::hidden text}} or {{c1::hidden text::hint}}. Use {{c2::...}} only when testing a tight bidirectional relationship in the same sentence.
   - Use "Basic" for conceptual "Why/How" questions, cause-and-effect mechanics, or step-by-step derivations.
3. Strict Anki MathJax & Cloze Formatting (CRITICAL):
   - NEVER use standard Markdown dollar signs ($or$$) for math.
   - ALWAYS wrap inline math, variables, Greek letters, and equations in Anki inline MathJax delimiters: \\(...\\).
   - ALWAYS wrap standalone display equations in Anki block MathJax delimiters: \\[...\\].
   - NEVER place raw LaTeX commands (like ^, _, \\frac, \\sqrt, \\cdots) inside a {{c1::...}} Cloze block without MathJax delimiters. Either place the Cloze deletion inside the outer MathJax block, e.g., \\([ABC] = {{c1::rs}}\\), OR wrap the math in its own MathJax delimiters inside the Cloze block, e.g., {{c1::\\((e_1 + 1)(e_2 + 1)\\cdots(e_k + 1)\\)}}.
   - DOUBLE-BRACE COLLISION FIX: Whenever a LaTeX expression inside a Cloze deletion ends with a curly brace `}`, ALWAYS insert a trailing space before the closing Cloze braces `}}` so three curly braces never touch (`}}}`). Correct: `\\( \\frac{AB}{BX} = {{c1::\\frac{AC}{CX} }} \\)`.
4. First-Principles Intuition:
   - In the `intuition` field, provide a concise 1–2 sentence plain-English explanation of *why* the concept or formula works from first principles, the underlying mechanism, or a common trap/edge case.
5. Source Traceability & Tagging:
   - Extract the exact chapter, section header, figure number, or page number into `source_reference`.
   - Format `tags` hierarchically using double colons and underscores (no spaces), e.g., `Subject::Module::Topic`.
""".strip()


# ---------------------------------------------------------------------------
# MathJax / Cloze Sanitizer
# ---------------------------------------------------------------------------

def sanitize_anki_mathjax(text: str) -> str:
    """
    Post-processing sanitizer that fixes common Anki MathJax / Cloze
    collision issues in generated card text.

    Rules applied:
    1. Replace any run of three-or-more `}` that are NOT part of a legitimate
       Cloze closing `}}` preceded by a space, with `} }}` to prevent
       triple-brace collisions.
    2. Replace bare Markdown dollar-sign math (`$...$` and `$$...$$`) with
       Anki MathJax delimiters.
    3. Collapse any accidental quadruple braces back to double-close `}}`.
    """
    if not text:
        return text

    # Fix 1: Triple brace collision — `}}}` → `} }}`
    # Only fix sequences of exactly three consecutive `}` not already separated
    text = re.sub(r'(?<!\})\}\}\}(?!\})', r'} }}', text)

    # Fix 2: Block math `$$...$$` → `\[...\]`
    text = re.sub(
        r'\$\$(.+?)\$\$',
        lambda m: r'\[' + m.group(1) + r'\]',
        text,
        flags=re.DOTALL,
    )

    # Fix 3: Inline math `$...$` → `\(...\)`  (skip already-replaced \[ \])
    text = re.sub(
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
        lambda m: r'\(' + m.group(1) + r'\)',
        text,
        flags=re.DOTALL,
    )

    return text


def sanitize_card(card: AnkiCard) -> AnkiCard:
    """Apply the MathJax sanitizer to the front and back fields of a card."""
    return card.model_copy(update={
        "front": sanitize_anki_mathjax(card.front),
        "back": sanitize_anki_mathjax(card.back),
    })


# ---------------------------------------------------------------------------
# Gemini Client Factory
# ---------------------------------------------------------------------------

def build_client(api_key: str) -> genai.Client:
    """Create and return a configured Gemini client."""
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Core Generation Function
# ---------------------------------------------------------------------------

def generate_flashcards(
    api_key: str,
    model_id: str,
    content_parts: List[types.Part],
    custom_instructions: str,
) -> FlashcardBatch:
    """
    Send content parts (image bytes / PDF bytes / text) plus custom
    instructions to Gemini and return a validated FlashcardBatch.

    Args:
        api_key:             Gemini API key.
        model_id:            Model string, e.g. "gemini-3.8-flash".
        content_parts:       List of types.Part objects (image, PDF, text).
        custom_instructions: Free-text instructions appended to the system prompt.

    Returns:
        A FlashcardBatch with sanitized cards.

    Raises:
        ValueError: If the model returns an unparseable response.
        Exception:  Propagates Gemini API errors.
    """
    client = build_client(api_key)

    # Merge system prompt + user custom instructions
    full_system = SYSTEM_PROMPT
    if custom_instructions.strip():
        full_system += (
            "\n\n### ADDITIONAL USER INSTRUCTIONS (follow these exactly):\n"
            + custom_instructions.strip()
        )

    # Build the user turn: content parts + instruction reminder
    user_parts = list(content_parts)
    user_parts.append(
        types.Part.from_text(
            text=(
                "Please generate Anki flashcards from the uploaded content "
                "following all the principles in the system instructions. "
                "Return a complete FlashcardBatch JSON object."
            )
        )
    )

    response = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=user_parts)],
        config=types.GenerateContentConfig(
            system_instruction=full_system,
            response_mime_type="application/json",
            response_schema=FlashcardBatch,
            temperature=0.3,
        ),
    )

    # Parse structured output
    try:
        batch: FlashcardBatch = response.parsed
    except Exception as exc:
        raise ValueError(
            f"Gemini returned unparseable structured output: {exc}\n"
            f"Raw text: {response.text[:500] if response.text else 'empty'}"
        ) from exc

    if batch is None:
        raise ValueError(
            "Gemini returned None for structured output. "
            "Check model compatibility or response content."
        )

    # Sanitize every card
    batch = batch.model_copy(update={
        "cards": [sanitize_card(c) for c in batch.cards]
    })

    return batch
