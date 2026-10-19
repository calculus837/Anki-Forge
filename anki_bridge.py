"""
anki_bridge.py
--------------
Dual export engine:
  Option A — Push cards live to Anki via AnkiConnect (localhost:8765)
  Option B — Build and return a .apkg file bytes via genanki
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import genanki
import requests

from gemini_engine import AnkiCard

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ANKICONNECT_URL = "http://localhost:8765"


def _build_back_html(card: Any) -> str:
    """
    Combine back, intuition, and source_reference into a rich HTML string
    suitable for Anki's back field / extra field.
    """
    back = card.back if hasattr(card, "back") else card.get("back", "")
    intuition = card.intuition if hasattr(card, "intuition") else card.get("intuition", "")
    source_ref = card.source_reference if hasattr(card, "source_reference") else card.get("source_reference", "")
    return (
        f"{back}"
        f"<br><br><hr>"
        f"<b>💡 First-Principles Intuition:</b> {intuition}"
        f"<br><small><i>📖 Source: {source_ref}</i></small>"
    )


def _tags_list(card: Any) -> List[str]:
    """Return the card's tags as a flat list, replacing spaces with underscores."""
    tags = card.tags if hasattr(card, "tags") else card.get("tags", [])
    if isinstance(tags, list):
        return [str(t).replace(" ", "_") for t in tags]
    return []


# ---------------------------------------------------------------------------
# Option A — AnkiConnect
# ---------------------------------------------------------------------------

def _ankiconnect_request(action: str, **params) -> dict:
    """Send a single request to the AnkiConnect HTTP API."""
    payload = {"action": action, "version": 6, "params": params}
    try:
        resp = requests.post(ANKICONNECT_URL, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"AnkiConnect error: {data['error']}")
        return data
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach AnkiConnect at http://localhost:8765. "
            "Make sure Anki is open with the AnkiConnect add-on installed."
        )


def ankiconnect_is_available() -> bool:
    """Return True if AnkiConnect responds to a ping."""
    try:
        _ankiconnect_request("version")
        return True
    except Exception:
        return False


def get_deck_names() -> List[str]:
    """Fetch all existing deck names from Anki."""
    result = _ankiconnect_request("deckNames")
    return sorted(result.get("result", []))


def ensure_deck(deck_name: str) -> None:
    """Create the deck if it doesn't already exist."""
    _ankiconnect_request("createDeck", deck=deck_name)


def store_media_file(filename: str, data: str | bytes) -> dict:
    """
    Store an image or audio file in Anki's collection.media directory via AnkiConnect.

    Args:
        filename: e.g. "ankiforge_12345678.png"
        data: Base64 string or raw bytes or dict
    """
    if isinstance(data, bytes):
        b64_content = base64.b64encode(data).decode("utf-8")
    elif isinstance(data, dict):
        b64_content = data.get("b64") or (base64.b64encode(data["bytes"]).decode("utf-8") if "bytes" in data else "")
    else:
        b64_str = str(data)
        if "," in b64_str and "base64" in b64_str:
            b64_content = b64_str.split(",", 1)[1]
        else:
            b64_content = b64_str
    return _ankiconnect_request("storeMediaFile", filename=filename, data=b64_content)


def _normalize_cloze_syntax(text: str) -> str:
    """Fix common minor syntax variations in Cloze formatting."""
    if not text:
        return text
    # {{c1:word}} -> {{c1::word}} (single colon instead of double)
    text = re.sub(r'\{\{c(\d+):(?!:)(.*?)\}\}', r'{{c\1::\2}}', text)
    # {c1::word} -> {{c1::word}} (single outer brace)
    text = re.sub(r'(?<!\{)\{c(\d+)::(.*?)\}(?!\})', r'{{c\1::\2}}', text)
    # {{c::word}} -> {{c1::word}} (missing number)
    text = re.sub(r'\{\{c::(.*?)\}\}', r'{{c1::\1}}', text)
    # [c1::word] -> {{c1::word}}
    text = re.sub(r'\[c(\d+)::(.*?)\]', r'{{c\1::\2}}', text)
    return text


def prepare_ankiconnect_payload(
    cards: list,
    deck_name: str,
) -> dict:
    """
    Construct the AnkiConnect 'addNotes' request payload.

    Returns:
        dict: {"action": "addNotes", "version": 6, "params": {"notes": [...]}}
    """
    notes = []
    for card in cards:
        c_type = card.card_type if hasattr(card, "card_type") else card.get("card_type", "Basic")
        front = card.front if hasattr(card, "front") else card.get("front", "")
        back = card.back if hasattr(card, "back") else card.get("back", "")
        back_html = _build_back_html(card)
        tags = _tags_list(card)

        # Normalize any slight syntax variations in cloze tags
        front_norm = _normalize_cloze_syntax(front.strip())
        back_norm = _normalize_cloze_syntax(back.strip())

        has_cloze_in_front = bool(re.search(r"\{\{c\d+::", front_norm))
        has_cloze_in_back = bool(re.search(r"\{\{c\d+::", back_norm))

        if c_type == "Cloze":
            if has_cloze_in_front:
                note = {
                    "deckName": deck_name,
                    "modelName": "Cloze",
                    "fields": {
                        "Text": front_norm,
                        "Back Extra": back_html,
                        "Extra": back_html,
                    },
                    "tags": tags,
                    "options": {"allowDuplicate": False},
                }
            elif has_cloze_in_back:
                # Cloze was placed in back field instead of front
                note = {
                    "deckName": deck_name,
                    "modelName": "Cloze",
                    "fields": {
                        "Text": back_norm,
                        "Back Extra": f"<b>Prompt:</b> {front_norm}<br><br>{back_html}",
                        "Extra": f"<b>Prompt:</b> {front_norm}<br><br>{back_html}",
                    },
                    "tags": tags,
                    "options": {"allowDuplicate": False},
                }
            else:
                # No cloze deletions exist! Anki would fail with 'cannot create note for unknown reason'
                # Fall back safely to Basic model so the card is never rejected.
                note = {
                    "deckName": deck_name,
                    "modelName": "Basic",
                    "fields": {
                        "Front": front_norm,
                        "Back": back_html,
                    },
                    "tags": tags,
                    "options": {"allowDuplicate": False},
                }
        else:  # Basic
            note = {
                "deckName": deck_name,
                "modelName": "Basic",
                "fields": {
                    "Front": front_norm,
                    "Back": back_html,
                },
                "tags": tags,
                "options": {"allowDuplicate": False},
            }
        notes.append(note)

    return {
        "action": "addNotes",
        "version": 6,
        "params": {
            "notes": notes,
        },
    }


def push_cards_to_anki(
    cards: list,
    deck_name: str,
    media_map: Optional[dict] = None,
) -> Tuple[int, List[str]]:
    """
    Push a list of AnkiCard objects to Anki via AnkiConnect.
    Optionally uploads any attached media files before creating notes.

    Returns:
        (success_count, list_of_errors)
    """
    ensure_deck(deck_name)

    # Store media files in Anki collection.media first so <img> tags resolve
    if media_map:
        for fname, f_data in media_map.items():
            try:
                store_media_file(fname, f_data)
            except Exception:
                pass

    payload = prepare_ankiconnect_payload(cards, deck_name)
    result = _ankiconnect_request("addNotes", **payload["params"])
    note_ids = result.get("result", [])

    errors = []
    success = 0
    for i, nid in enumerate(note_ids):
        if nid is None:
            front_text = cards[i].front if hasattr(cards[i], "front") else cards[i].get("front", "")
            errors.append(f"Card {i + 1} ('{front_text[:40]}…') was rejected (duplicate or invalid).")
        else:
            success += 1

    return success, errors


def check_duplicates_in_anki(cards: list, deck_name: Optional[str] = None) -> List[bool]:
    """
    Check if cards are duplicates of existing notes in the user's Anki collection
    (or duplicates within the current batch).

    Args:
        cards: List of AnkiCard objects or card dicts.
        deck_name: Target deck name to check against (defaults to "Default" if None or empty).

    Returns:
        List[bool]: True for each card that is a duplicate, False for new cards.
                    Returns [False] * len(cards) gracefully if Anki is offline.
    """
    if not cards:
        return []

    is_dupe = [False] * len(cards)

    try:
        # 1. Intra-batch duplicate check (normalized whitespace & lower-cased)
        seen_fronts = set()
        for i, card in enumerate(cards):
            front = card.front if hasattr(card, "front") else card.get("front", "")
            norm = " ".join(front.strip().lower().split())
            if norm and norm in seen_fronts:
                is_dupe[i] = True
            elif norm:
                seen_fronts.add(norm)

        # 2. Check against Anki collection via AnkiConnect (if Anki is running)
        if ankiconnect_is_available():
            try:
                target_deck = (
                    deck_name.strip()
                    if (deck_name and isinstance(deck_name, str) and deck_name.strip())
                    else "Default"
                )
                ensure_deck(target_deck)

                notes_payload = []
                for card in cards:
                    c_type = card.card_type if hasattr(card, "card_type") else card.get("card_type", "Basic")
                    front = card.front if hasattr(card, "front") else card.get("front", "")
                    front_clean = front.strip()
                    back = card.back if hasattr(card, "back") else card.get("back", "")

                    if c_type == "Basic":
                        notes_payload.append({
                            "deckName": target_deck,
                            "modelName": "Basic",
                            "fields": {"Front": front_clean, "Back": back},
                            "tags": [],
                        })
                    else:
                        notes_payload.append({
                            "deckName": target_deck,
                            "modelName": "Cloze",
                            "fields": {"Text": front_clean, "Back Extra": back},
                            "tags": [],
                        })

                # Check via canAddNotes
                res = _ankiconnect_request("canAddNotes", notes=notes_payload)
                can_add_list = res.get("result", [])
                for i, can_add in enumerate(can_add_list):
                    if can_add is False:
                        is_dupe[i] = True

                # Also check via findNotes for near-identical matches (e.g. slight formatting differences)
                for i, card in enumerate(cards):
                    if not is_dupe[i]:
                        front = card.front if hasattr(card, "front") else card.get("front", "")
                        clean_query = front.replace('"', '').replace('\\', '').strip()
                        unclozed = re.sub(r'\{\{c\d+::(.*?)(?:::.*?)?\}\}', r'\1', clean_query).strip()
                        if len(unclozed) >= 15:
                            first_phrase = unclozed[:50].strip()
                            safe_phrase = re.sub(r'[\\:*?"<>|]', ' ', first_phrase).strip()
                            if safe_phrase:
                                try:
                                    find_res = _ankiconnect_request("findNotes", query=f'"{safe_phrase}"')
                                    if find_res.get("result"):
                                        is_dupe[i] = True
                                except Exception:
                                    pass
            except Exception:
                pass
    except Exception:
        pass

    return is_dupe


# ---------------------------------------------------------------------------
# Option B — genanki .apkg builder
# ---------------------------------------------------------------------------

# Stable model IDs (must be unique per model type)
_BASIC_MODEL_ID = 1607392319
_CLOZE_MODEL_ID = 998877661


def _make_basic_model() -> genanki.Model:
    return genanki.Model(
        _BASIC_MODEL_ID,
        "AnkiForge Basic",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
            }
        ],
        css="""
        .card {
            font-family: "Georgia", serif;
            font-size: 18px;
            text-align: center;
            color: #1a1a2e;
            background-color: #f8f9fa;
            padding: 20px;
            line-height: 1.6;
        }
        hr { border-color: #dee2e6; }
        b { color: #495057; }
        small { color: #6c757d; font-size: 0.8em; }
        """,
    )


def _make_cloze_model() -> genanki.Model:
    return genanki.Model(
        _CLOZE_MODEL_ID,
        "AnkiForge Cloze",
        model_type=genanki.Model.CLOZE,
        fields=[
            {"name": "Text"},
            {"name": "Back Extra"},
        ],
        templates=[
            {
                "name": "Cloze",
                "qfmt": "{{cloze:Text}}",
                "afmt": "{{cloze:Text}}<br>{{Back Extra}}",
            }
        ],
        css="""
        .card {
            font-family: "Georgia", serif;
            font-size: 18px;
            text-align: center;
            color: #1a1a2e;
            background-color: #f8f9fa;
            padding: 20px;
            line-height: 1.6;
        }
        .cloze { font-weight: bold; color: #0d6efd; }
        hr { border-color: #dee2e6; }
        b { color: #495057; }
        small { color: #6c757d; font-size: 0.8em; }
        """,
    )


def build_apkg(
    cards: List[AnkiCard],
    deck_name: str,
    media_map: Optional[dict] = None,
) -> bytes:
    """
    Compile a list of AnkiCard objects into a .apkg bytes buffer using genanki.
    Optionally embeds media files from media_map into the package.

    Args:
        cards: List of AnkiCard objects.
        deck_name: Name of the deck.
        media_map: Optional dict mapping filename (e.g. "ankiforge_xxx.png")
                   to bytes, base64 str, or dict {"bytes": ..., "b64": ...}.

    Returns:
        Raw bytes of the .apkg file ready for download.
    """
    # Deterministic deck ID derived from name so re-imports update, not duplicate
    deck_id = abs(hash(deck_name)) % (10 ** 10)
    deck = genanki.Deck(deck_id, deck_name)

    basic_model = _make_basic_model()
    cloze_model = _make_cloze_model()

    for card in cards:
        front = card.front if hasattr(card, "front") else card.get("front", "")
        back = card.back if hasattr(card, "back") else card.get("back", "")
        c_type = card.card_type if hasattr(card, "card_type") else card.get("card_type", "Basic")
        back_html = _build_back_html(card)
        tags = _tags_list(card)

        front_norm = _normalize_cloze_syntax(front.strip())
        back_norm = _normalize_cloze_syntax(back.strip())

        has_cloze_in_front = bool(re.search(r"\{\{c\d+::", front_norm))
        has_cloze_in_back = bool(re.search(r"\{\{c\d+::", back_norm))

        # Stable note ID: hash of front text to avoid duplicates on re-export
        note_id = abs(hash(front_norm + c_type)) % (10 ** 13)

        if c_type == "Cloze" and (has_cloze_in_front or has_cloze_in_back):
            text_field = front_norm if has_cloze_in_front else back_norm
            extra_field = back_html if has_cloze_in_front else f"<b>Prompt:</b> {front_norm}<br><br>{back_html}"
            note = genanki.Note(
                model=cloze_model,
                fields=[text_field, extra_field],
                tags=tags,
                guid=note_id,
            )
        else:
            note = genanki.Note(
                model=basic_model,
                fields=[front_norm, back_html],
                tags=tags,
                guid=note_id,
            )
        deck.add_note(note)

    package = genanki.Package(deck)

    # If media files are provided, write them to a temp folder and set package.media_files
    temp_dir = None
    try:
        if media_map:
            temp_dir = tempfile.TemporaryDirectory()
            media_paths = []
            for fname, f_data in media_map.items():
                if isinstance(f_data, bytes):
                    raw_bytes = f_data
                elif isinstance(f_data, dict):
                    raw_bytes = f_data.get("bytes")
                    if not raw_bytes and "b64" in f_data:
                        raw_bytes = base64.b64decode(f_data["b64"])
                elif isinstance(f_data, str):
                    clean_str = f_data.split(",", 1)[1] if ("," in f_data and "base64" in f_data) else f_data
                    raw_bytes = base64.b64decode(clean_str)
                else:
                    continue

                if raw_bytes:
                    fpath = os.path.join(temp_dir.name, fname)
                    with open(fpath, "wb") as f:
                        f.write(raw_bytes)
                    media_paths.append(fpath)

            if media_paths:
                package.media_files = media_paths

        buf = io.BytesIO()
        package.write_to_file(buf)
        return buf.getvalue()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
