"""
app.py
------
AnkiForge — AI-powered Anki flashcard generator.

Run with:
    streamlit run app.py
"""

import base64
import hashlib
import io
import json
import os
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google.genai import types
from PIL import Image
from streamlit_paste_button import paste_image_button

from anki_bridge import (
    ankiconnect_is_available,
    build_apkg,
    check_duplicates_in_anki,
    get_deck_names,
    prepare_ankiconnect_payload,
    push_cards_to_anki,
)
from gemini_engine import AnkiCard, FlashcardBatch, generate_flashcards
from pdf_processor import (
    get_page_count,
    image_bytes_to_part,
    pdf_bytes_to_part,
    render_page_thumbnail,
    slice_pdf,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AnkiForge — AI Flashcard Generator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards in the staging area */
    .card-preview {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .card-preview .card-front {
        font-weight: 600;
        font-size: 1.05rem;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .card-preview .card-back {
        color: #a0aec0;
        font-size: 0.92rem;
    }
    .card-preview .card-meta {
        margin-top: 8px;
        font-size: 0.78rem;
        color: #718096;
    }
    .badge-basic {
        background: #3182ce;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-cloze {
        background: #805ad5;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-dupe {
        background: #e53e3e;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Generate button */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* Data editor */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Success / error boxes */
    .success-box {
        background: rgba(72, 187, 120, 0.15);
        border: 1px solid rgba(72, 187, 120, 0.4);
        border-radius: 8px;
        padding: 12px 16px;
        color: #68d391;
    }
    .error-box {
        background: rgba(252, 129, 74, 0.15);
        border: 1px solid rgba(252, 129, 74, 0.4);
        border-radius: 8px;
        padding: 12px 16px;
        color: #fc814a;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load environment / API key
# ---------------------------------------------------------------------------

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "staged_cards" not in st.session_state:
    st.session_state.staged_cards: List[dict] = []
if "deck_suggestion" not in st.session_state:
    st.session_state.deck_suggestion: str = "AnkiForge::GeneratedDeck"
if "generation_count" not in st.session_state:
    st.session_state.generation_count: int = 0
if "pasted_images" not in st.session_state:
    st.session_state.pasted_images: List[bytes] = []
if "paste_counter" not in st.session_state:
    st.session_state.paste_counter: int = 0
if "last_pasted_hash" not in st.session_state:
    st.session_state.last_pasted_hash: Optional[str] = None
if "staged_media" not in st.session_state:
    st.session_state.staged_media: dict = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔥 AnkiForge")
    st.markdown("*AI-powered Anki card generator*")
    st.divider()

    # --- API Key ---
    secret_key = ""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    env_key = os.getenv("GEMINI_API_KEY", "")

    if secret_key and secret_key != "your_api_key_here":
        st.success("✅ API key loaded from secrets")
        api_key = secret_key
        with st.expander("🔑 Override with your own key", expanded=False):
            custom_key = st.text_input(
                "Custom Gemini API Key",
                type="password",
                placeholder="Paste key to override secrets…",
                help="Enter your own key to override the default secret.",
                key="custom_api_key_override",
            )
            if custom_key:
                api_key = custom_key
    elif env_key and env_key != "your_api_key_here":
        st.success("✅ API key loaded from `.env`")
        api_key = env_key
        with st.expander("🔑 Override with your own key", expanded=False):
            custom_key = st.text_input(
                "Custom Gemini API Key",
                type="password",
                placeholder="Paste key to override .env…",
                help="Enter your own key to override the .env key.",
                key="custom_api_key_override",
            )
            if custom_key:
                api_key = custom_key
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your key here…",
            help="Get your key at https://aistudio.google.com/app/apikey",
        )
        if not api_key:
            st.warning("⚠️ Enter your Gemini API key to get started.")

    st.divider()

    # --- Model selection ---
    MODEL_OPTIONS = {
        "gemini-3.8-flash (Default)": "gemini-3.8-flash",
        "gemini-3.5-flash-lite (Fast)": "gemini-3.5-flash-lite",
        "gemini-4-argon (Best)": "gemini-4-argon",
    }
    selected_label = st.selectbox(
        "🤖 Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
    )
    model_id = MODEL_OPTIONS[selected_label]

    st.divider()

    # --- Staging area stats ---
    n_cards = len(st.session_state.staged_cards)
    col_a, col_b = st.columns(2)
    col_a.metric("Staged Cards", n_cards)
    col_b.metric("Batches Run", st.session_state.generation_count)

    if n_cards > 0 and st.button("🗑️ Clear All Cards", use_container_width=True):
        st.session_state.staged_cards = []
        st.session_state.staged_media = {}
        st.session_state.deck_suggestion = "AnkiForge::GeneratedDeck"
        st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#718096'>AnkiForge v1.0 · Powered by Gemini</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">🔥 AnkiForge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload textbook pages or PDFs → Get atomic, MathJax-ready Anki flashcards in seconds.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Preset instruction buttons + custom instructions area
# ---------------------------------------------------------------------------

PRESETS = {
    "📖 Definitions & Theorems (Cloze)": (
        "Focus exclusively on definitions, theorems, and named results. "
        "Use Cloze deletions to hide the definition body or the theorem conditions. "
        "Every Cloze card must test exactly one term or one theorem statement."
    ),
    "∫ Formula Derivations & Conditions": (
        "Generate one Basic card per major formula explaining WHY it holds from first principles. "
        "Also generate Cloze cards for the necessary/sufficient conditions of each formula. "
        "Always include MathJax-formatted equations."
    ),
    "📊 Deconstruct Tables & Comparisons": (
        "If the content contains any comparison tables, property lists, or side-by-side analyses, "
        "break them down into individual atomic Cloze cards — one card per row or one card per property. "
        "Do NOT create one giant card for the whole table."
    ),
    "⚠️ Traps, Edge Cases & Counterexamples": (
        "Focus on common misconceptions, edge cases, degenerate cases, and classic counterexamples. "
        "Frame each Basic card as 'What goes wrong when…' or 'Why does X fail when…'. "
        "Include the specific condition that triggers the edge case."
    ),
}

st.markdown("### 📝 Custom Instructions")
preset_cols = st.columns(len(PRESETS))
selected_preset = ""
for col, (label, text) in zip(preset_cols, PRESETS.items()):
    if col.button(label, use_container_width=True, key=f"preset_{label[:10]}"):
        selected_preset = text

if "custom_instructions" not in st.session_state:
    st.session_state.custom_instructions = ""

if selected_preset:
    st.session_state.custom_instructions = selected_preset

custom_instructions = st.text_area(
    "Additional instructions (or use a preset above)",
    value=st.session_state.custom_instructions,
    height=100,
    placeholder="e.g. 'Only generate Cloze cards for this topology chapter. Ignore all examples.'",
    key="custom_instructions_area",
)
st.session_state.custom_instructions = custom_instructions

st.divider()

# ---------------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------------

tab_img, tab_pdf = st.tabs(["🖼️ Screenshot / Image Upload", "📄 Textbook PDF Slicer"])

content_parts: List[types.Part] = []
current_batch_media: List[str] = []
ready_to_generate = False

# ── Tab 1: Image upload & Clipboard Paste ───────────────────────────────────
with tab_img:
    st.markdown("#### 📸 Capture or Upload Textbook Images")
    st.caption(
        "Use **Win+Shift+S** (Windows) or **Cmd+Shift+4** (Mac) to snip any formula, table, or paragraph, "
        "then click the button below to paste directly from your clipboard without saving files."
    )

    # Prominent clipboard paste section
    col_paste_btn, col_paste_status = st.columns([1, 2])
    with col_paste_btn:
        paste_result = paste_image_button(
            label="📋 Paste Image from Clipboard",
            text_color="#ffffff",
            background_color="#6366f1",
            hover_background_color="#4f46e5",
            key=f"paste_btn_{st.session_state.paste_counter}",
            errors="ignore",
        )
    with col_paste_status:
        n_pasted = len(st.session_state.pasted_images)
        if n_pasted > 0:
            st.markdown(
                f"<div style='margin-top:6px; color:#a5b4fc; font-weight:500;'>"
                f"✅ <b>{n_pasted} image{'s' if n_pasted > 1 else ''}</b> currently queued from clipboard"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("👈 Click to paste clipboard screenshot directly")

    # Handle incoming clipboard paste
    if paste_result.image_data is not None:
        buf = io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        pasted_bytes = buf.getvalue()
        pasted_hash = hashlib.md5(pasted_bytes).hexdigest()
        if pasted_hash != st.session_state.last_pasted_hash:
            st.session_state.last_pasted_hash = pasted_hash
            st.session_state.pasted_images.append(pasted_bytes)
            st.session_state.paste_counter += 1
            st.toast(f"📋 Pasted screenshot #{len(st.session_state.pasted_images)} from clipboard!", icon="✅")
            st.rerun()

    # If any images are queued from clipboard, show thumbnails with management controls
    if st.session_state.pasted_images:
        st.markdown("---")
        head_col, clear_col = st.columns([3, 1])
        with head_col:
            st.markdown(f"##### 📋 Queued Clipboard Images ({len(st.session_state.pasted_images)})")
        with clear_col:
            if st.button("🗑️ Clear Pasted Images", use_container_width=True, key="clear_all_pasted"):
                st.session_state.pasted_images = []
                st.session_state.last_pasted_hash = None
                st.session_state.paste_counter += 1
                st.rerun()

        cols_per_row = 4
        pasted_list = list(enumerate(st.session_state.pasted_images))
        for row_start in range(0, len(pasted_list), cols_per_row):
            thumb_cols = st.columns(cols_per_row)
            for col_idx, (img_idx, img_bytes) in enumerate(pasted_list[row_start : row_start + cols_per_row]):
                with thumb_cols[col_idx]:
                    st.image(img_bytes, caption=f"Pasted #{img_idx + 1}", use_container_width=True)
                    if st.button("❌ Remove", key=f"del_paste_{img_idx}", use_container_width=True):
                        st.session_state.pasted_images.pop(img_idx)
                        st.session_state.last_pasted_hash = None
                        st.session_state.paste_counter += 1
                        st.rerun()

    st.markdown("---")
    st.markdown("##### 📁 Or Upload Files from Disk")
    uploaded_images = st.file_uploader(
        "Drag & drop images here",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )

    if uploaded_images:
        st.markdown(f"##### 🖼️ Uploaded Files ({len(uploaded_images)})")
        up_cols = st.columns(min(len(uploaded_images), 4))
        for i, img_file in enumerate(uploaded_images):
            with up_cols[i % 4]:
                st.image(img_file.getvalue(), caption=img_file.name, use_container_width=True)

    st.markdown("---")
    attach_source_images = st.checkbox(
        "🖼️ Attach source image to the back of generated cards",
        value=True,
        help="Uploads the source image to Anki collection.media via storeMediaFile, embeds <br><img src='ankiforge_<hash>.png'> on the back of each card, and packages it into .apkg exports.",
        key="attach_source_images_cb",
    )

    # Unified generate button for all queued images (pasted + uploaded)
    total_queued = len(st.session_state.pasted_images) + (len(uploaded_images) if uploaded_images else 0)

    if total_queued > 0:
        desc_parts = []
        if st.session_state.pasted_images:
            desc_parts.append(f"{len(st.session_state.pasted_images)} pasted")
        if uploaded_images:
            desc_parts.append(f"{len(uploaded_images)} uploaded")
        detail_text = f" ({' + '.join(desc_parts)})"

        gen_img = st.button(
            f"⚡ Generate Flashcards from {total_queued} Image(s){detail_text}",
            type="primary",
            use_container_width=True,
            key="gen_img_btn",
        )
        if gen_img:
            image_parts: List[types.Part] = []
            batch_media_files: List[str] = []

            # Add pasted images
            for p_bytes in st.session_state.pasted_images:
                image_parts.append(image_bytes_to_part(p_bytes, "image/png"))
                if attach_source_images:
                    img_hash = hashlib.md5(p_bytes).hexdigest()[:12]
                    fname = f"ankiforge_{img_hash}.png"
                    st.session_state.staged_media[fname] = p_bytes
                    if fname not in batch_media_files:
                        batch_media_files.append(fname)

            # Add uploaded images
            if uploaded_images:
                for img_file in uploaded_images:
                    raw = img_file.getvalue()
                    mime = img_file.type or "image/png"
                    image_parts.append(image_bytes_to_part(raw, mime))
                    if attach_source_images:
                        img_hash = hashlib.md5(raw).hexdigest()[:12]
                        fname = f"ankiforge_{img_hash}.png"
                        st.session_state.staged_media[fname] = raw
                        if fname not in batch_media_files:
                            batch_media_files.append(fname)

            content_parts = image_parts
            current_batch_media = batch_media_files
            ready_to_generate = True
    else:
        st.info("💡 Snip a screenshot (`Win+Shift+S`) and click **Paste Image from Clipboard**, or upload image files above.")

# ── Tab 2: PDF Slicer ────────────────────────────────────────────────────────
with tab_pdf:
    st.markdown("#### Upload a PDF and select the page range to extract")
    uploaded_pdf = st.file_uploader(
        "Drag & drop your PDF here",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_pdf:
        pdf_raw = uploaded_pdf.read()
        total_pages = get_page_count(pdf_raw)
        st.info(f"📄 **{uploaded_pdf.name}** — {total_pages} pages detected")

        col_start, col_end = st.columns(2)
        with col_start:
            start_page = st.number_input(
                "Start page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="pdf_start",
            )
        with col_end:
            end_page = st.number_input(
                "End page",
                min_value=1,
                max_value=total_pages,
                value=min(8, total_pages),
                step=1,
                key="pdf_end",
            )

        page_count_selected = int(end_page) - int(start_page) + 1
        if start_page > end_page:
            st.error("⚠️ Start page must be ≤ End page.")
        else:
            st.caption(
                f"Will extract **{page_count_selected} page(s)** "
                f"(pages {int(start_page)}–{int(end_page)}) and send to Gemini."
            )

            # Live thumbnail of first selected page
            with st.expander("🔍 Preview first selected page", expanded=False):
                try:
                    thumb = render_page_thumbnail(pdf_raw, int(start_page))
                    st.image(thumb, caption=f"Page {int(start_page)}", use_container_width=True)
                except Exception as e:
                    st.warning(f"Preview unavailable: {e}")

            gen_pdf = st.button(
                f"⚡ Generate Flashcards from Pages {int(start_page)}–{int(end_page)}",
                type="primary",
                use_container_width=True,
                key="gen_pdf_btn",
                disabled=(start_page > end_page),
            )
            if gen_pdf:
                with st.spinner("✂️ Slicing PDF…"):
                    try:
                        sliced = slice_pdf(pdf_raw, int(start_page), int(end_page))
                        content_parts = [pdf_bytes_to_part(sliced)]
                        ready_to_generate = True
                    except ValueError as e:
                        st.error(str(e))

# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

if ready_to_generate:
    if not api_key:
        st.error("❌ Please enter your Gemini API key in the sidebar.")
    elif not content_parts:
        st.error("❌ No content to send — upload an image or PDF first.")
    else:
        with st.spinner(f"🧠 Asking {selected_label} to forge your flashcards…"):
            try:
                batch: FlashcardBatch = generate_flashcards(
                    api_key=api_key,
                    model_id=model_id,
                    content_parts=content_parts,
                    custom_instructions=custom_instructions,
                )
                # Append new cards to the staging area with duplicate detection
                new_cards = [c.model_dump() for c in batch.cards]

                # Append attached source image(s) to card back if enabled
                if current_batch_media:
                    img_tags = "".join([f'<br><img src="{fname}">' for fname in current_batch_media])
                    for c in new_cards:
                        c["back"] = (c.get("back", "") + img_tags).strip()

                dupe_flags = check_duplicates_in_anki(new_cards, batch.deck_suggestion)
                for c, is_dupe in zip(new_cards, dupe_flags):
                    c["is_duplicate"] = is_dupe
                    c["include_in_export"] = not is_dupe  # Default: True for new, False for duplicates

                st.session_state.staged_cards.extend(new_cards)
                st.session_state.deck_suggestion = batch.deck_suggestion
                st.session_state.generation_count += 1

                dupe_count = sum(dupe_flags)
                if dupe_count > 0:
                    st.warning(
                        f"⚠️ Detected **{dupe_count} duplicate card(s)** already in Anki or repeated in batch! "
                        "They have been unchecked by default in the staging table."
                    )
                st.success(
                    f"✅ Generated **{len(new_cards)} cards** ({len(new_cards) - dupe_count} new) — "
                    f"suggested deck: **{batch.deck_suggestion}**"
                )
            except Exception as exc:
                st.error(f"❌ Generation failed: {exc}")

# ---------------------------------------------------------------------------
# Staging area
# ---------------------------------------------------------------------------

st.divider()
st.markdown("## 🗂️ Flashcard Staging Area")

if not st.session_state.staged_cards:
    st.info("No cards staged yet. Upload content above and click Generate.")
else:
    # Ensure all cards have duplicate and export status keys
    for card_item in st.session_state.staged_cards:
        if "is_duplicate" not in card_item:
            card_item["is_duplicate"] = False
        if "include_in_export" not in card_item:
            card_item["include_in_export"] = not card_item.get("is_duplicate", False)

    n = len(st.session_state.staged_cards)
    dupes_in_stage = sum(1 for c in st.session_state.staged_cards if c.get("is_duplicate"))
    selected_for_export = sum(1 for c in st.session_state.staged_cards if c.get("include_in_export"))

    # Toolbar above the table
    toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([2, 1, 1])
    with toolbar_col1:
        caption_dupe = f" · <span style='color:#fc8181;'><b>{dupes_in_stage} duplicate(s) flagged</b></span>" if dupes_in_stage else ""
        st.markdown(
            f"**{n} card(s)** staged ({selected_for_export} selected for export){caption_dupe}",
            unsafe_allow_html=True,
        )
    with toolbar_col2:
        if st.button("🔍 Re-check Duplicates", use_container_width=True, help="Scan cards against Anki collection"):
            with st.spinner("Checking Anki collection for duplicates…"):
                fresh_dupes = check_duplicates_in_anki(st.session_state.staged_cards, st.session_state.deck_suggestion)
                for c, is_d in zip(st.session_state.staged_cards, fresh_dupes):
                    c["is_duplicate"] = is_d
                    if is_d:
                        c["include_in_export"] = False
                st.rerun()
    with toolbar_col3:
        if dupes_in_stage > 0:
            if st.button("🚫 Skip All Duplicates", use_container_width=True, help="Uncheck all detected duplicates"):
                for c in st.session_state.staged_cards:
                    if c.get("is_duplicate"):
                        c["include_in_export"] = False
                st.rerun()
        else:
            if st.button("✅ Select All Cards", use_container_width=True):
                for c in st.session_state.staged_cards:
                    c["include_in_export"] = True
                st.rerun()

    # ── Editable data table ──────────────────────────────────────────────────
    import pandas as pd

    df = pd.DataFrame(st.session_state.staged_cards)
    for col in ["card_type", "front", "back", "intuition", "source_reference", "tags"]:
        if col not in df.columns:
            df[col] = ""
    if "include_in_export" not in df.columns:
        df["include_in_export"] = True
    if "is_duplicate" not in df.columns:
        df["is_duplicate"] = False

    df["include_in_export"] = df["include_in_export"].astype(bool)
    df["is_duplicate"] = df["is_duplicate"].astype(bool)

    # Convert tags list → comma-separated string for editing
    df["tags"] = df["tags"].apply(
        lambda t: ", ".join(t) if isinstance(t, list) else str(t)
    )

    # Prepare display dataframe with exact column names requested
    display_df = df[[
        "include_in_export",
        "is_duplicate",
        "card_type",
        "front",
        "back",
        "intuition",
        "source_reference",
        "tags",
    ]].rename(columns={
        "include_in_export": "Include in Export",
        "is_duplicate": "⚠️ Duplicate?",
    })

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Include in Export": st.column_config.CheckboxColumn(
                "Include in Export",
                help="Check to include this card when pushing to Anki or downloading .apkg",
                default=True,
                width="small",
            ),
            "⚠️ Duplicate?": st.column_config.CheckboxColumn(
                "⚠️ Duplicate?",
                help="Card already exists in your Anki collection or is repeated in this batch",
                disabled=True,
                width="small",
            ),
            "card_type": st.column_config.SelectboxColumn(
                "Type",
                options=["Basic", "Cloze"],
                width="small",
            ),
            "front": st.column_config.TextColumn("Front", width="large"),
            "back": st.column_config.TextColumn("Back", width="large"),
            "intuition": st.column_config.TextColumn("Intuition", width="medium"),
            "source_reference": st.column_config.TextColumn("Source", width="medium"),
            "tags": st.column_config.TextColumn("Tags (comma-separated)", width="medium"),
        },
        key="staging_editor",
    )

    # Sync edits back to session state
    def df_to_cards(df) -> List[dict]:
        records = []
        for _, row in df.iterrows():
            tags_raw = row.get("tags", "")
            if isinstance(tags_raw, list):
                tags = tags_raw
            else:
                tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()]
            records.append({
                "include_in_export": bool(row.get("Include in Export", True)),
                "is_duplicate": bool(row.get("⚠️ Duplicate?", False)),
                "card_type": row.get("card_type", "Basic"),
                "front": row.get("front", ""),
                "back": row.get("back", ""),
                "intuition": row.get("intuition", ""),
                "source_reference": row.get("source_reference", ""),
                "tags": tags,
            })
        return records

    st.session_state.staged_cards = df_to_cards(edited_df)

    # ── Live card preview ────────────────────────────────────────────────────
    with st.expander("👁️ Live Card Preview", expanded=False):
        for i, card_dict in enumerate(st.session_state.staged_cards[:20]):
            badge_class = "badge-cloze" if card_dict.get("card_type") == "Cloze" else "badge-basic"
            badge_label = card_dict.get("card_type", "Basic")
            dupe_badge = '<span class="badge-dupe">⚠️ DUPLICATE</span>' if card_dict.get("is_duplicate") else ""
            skip_style = "opacity: 0.55;" if not card_dict.get("include_in_export", True) else ""
            tags_str = ", ".join(card_dict.get("tags", [])) if isinstance(card_dict.get("tags"), list) else card_dict.get("tags", "")

            # In live preview, resolve <img src="ankiforge_..."> to inline base64 data URIs for display
            back_display = card_dict.get('back', '')
            if st.session_state.get("staged_media"):
                for fname, mdata in st.session_state.staged_media.items():
                    if f'src="{fname}"' in back_display:
                        if isinstance(mdata, bytes):
                            b64 = base64.b64encode(mdata).decode("utf-8")
                        else:
                            b64 = str(mdata)
                        back_display = back_display.replace(
                            f'src="{fname}"',
                            f'src="data:image/png;base64,{b64}" style="max-height:220px; border-radius:6px; margin-top:8px; box-shadow:0 2px 8px rgba(0,0,0,0.3);"'
                        )

            st.markdown(
                f"""<div class="card-preview" style="{skip_style}">
                    <div class="card-front">
                        <span class="{badge_class}">{badge_label}</span>
                        {dupe_badge}
                        {card_dict.get('front', '')}
                    </div>
                    <div class="card-back">{back_display}</div>
                    <div class="card-meta">
                        💡 {card_dict.get('intuition', '')}
                        <br>📖 {card_dict.get('source_reference', '')}
                        {"&nbsp;·&nbsp;🏷 " + tags_str if tags_str else ""}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        if len(st.session_state.staged_cards) > 20:
            st.caption(f"… and {len(st.session_state.staged_cards) - 20} more card(s) not shown.")

    # ---------------------------------------------------------------------------
    # Export controls
    # ---------------------------------------------------------------------------

    def _render_browser_bridge_html(
        notes: list,
        media: list,
        deck_name: str,
        disabled: bool = False,
    ) -> str:
        """Generate client-side HTML/JS component to push cards directly to 127.0.0.1:8765."""
        notes_json = json.dumps(notes)
        media_json = json.dumps(media)
        deck_json = json.dumps(deck_name)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            * {{
              box-sizing: border-box;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }}
            body {{
              margin: 0;
              padding: 2px 0;
              background: transparent;
            }}
            .bridge-btn {{
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 100%;
              padding: 0.65rem 1rem;
              font-size: 0.95rem;
              font-weight: 600;
              color: #ffffff;
              background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
              border: 1px solid rgba(255, 255, 255, 0.15);
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.2s ease;
              box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
            }}
            .bridge-btn:hover:not(:disabled) {{
              background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
              box-shadow: 0 6px 18px rgba(79, 70, 229, 0.5);
              transform: translateY(-1px);
            }}
            .bridge-btn:active:not(:disabled) {{
              transform: translateY(0);
            }}
            .bridge-btn:disabled {{
              opacity: 0.5;
              cursor: not-allowed;
              box-shadow: none;
            }}
            .status-box {{
              margin-top: 10px;
              padding: 10px 14px;
              border-radius: 8px;
              font-size: 0.88rem;
              line-height: 1.45;
              display: none;
            }}
            .status-info {{
              display: block;
              background: rgba(99, 102, 241, 0.15);
              border: 1px solid rgba(99, 102, 241, 0.4);
              color: #c7d2fe;
            }}
            .status-success {{
              display: block;
              background: rgba(16, 185, 129, 0.15);
              border: 1px solid rgba(16, 185, 129, 0.5);
              color: #6ee7b7;
            }}
            .status-error {{
              display: block;
              background: rgba(239, 68, 68, 0.15);
              border: 1px solid rgba(239, 68, 68, 0.5);
              color: #fca5a5;
            }}
            code {{
              background: rgba(0, 0, 0, 0.35);
              padding: 2px 5px;
              border-radius: 4px;
              font-family: monospace;
              color: #f1f5f9;
            }}
          </style>
        </head>
        <body>
          <button id="bridge-btn" class="bridge-btn" {"disabled" if disabled else ""} onclick="pushToLocalAnki()">
            🌐 Cloud-to-Local Push to Anki (Browser Bridge)
          </button>
          <div id="bridge-status" class="status-box"></div>

          <script>
            const NOTES = {notes_json};
            const MEDIA = {media_json};
            const DECK_NAME = {deck_json};

            async function ankiFetch(action, params = {{}}) {{
              const res = await fetch("http://127.0.0.1:8765", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ action: action, version: 6, params: params }})
              }});
              if (!res.ok) {{
                throw new Error(`HTTP error: ${{res.status}} ${{res.statusText}}`);
              }}
              const data = await res.json();
              if (data.error) {{
                throw new Error(data.error);
              }}
              return data.result;
            }}

            async function pushToLocalAnki() {{
              const btn = document.getElementById("bridge-btn");
              const status = document.getElementById("bridge-status");

              if (!NOTES || NOTES.length === 0) {{
                status.className = "status-box status-info";
                status.innerHTML = "ℹ️ No cards selected for export.";
                return;
              }}

              btn.disabled = true;
              status.className = "status-box status-info";
              status.innerHTML = "⏳ Connecting to local AnkiConnect at <code>http://127.0.0.1:8765</code>…";

              try {{
                // 1. Ensure Deck exists
                await ankiFetch("createDeck", {{ deck: DECK_NAME }});

                // 2. Upload any staged media
                if (MEDIA && MEDIA.length > 0) {{
                  for (let i = 0; i < MEDIA.length; i++) {{
                    status.innerHTML = `⏳ Uploading media files (${{i + 1}}/${{MEDIA.length}})…`;
                    await ankiFetch("storeMediaFile", {{
                      filename: MEDIA[i].filename,
                      data: MEDIA[i].data
                    }});
                  }}
                }}

                // 3. Add Notes
                status.innerHTML = `⏳ Adding ${{NOTES.length}} card(s) to deck <b>${{DECK_NAME}}</b>…`;
                const noteIds = await ankiFetch("addNotes", {{ notes: NOTES }});

                let success = 0;
                let skipped = 0;
                for (const nid of (noteIds || [])) {{
                  if (nid !== null && nid !== undefined) {{
                    success++;
                  }} else {{
                    skipped++;
                  }}
                }}

                status.className = "status-box status-success";
                let msg = `✅ <b>${{success}} card(s)</b> successfully added to Anki deck <b>${{DECK_NAME}}</b>!`;
                if (skipped > 0) {{
                  msg += `<br><small style="opacity: 0.9">ℹ️ ${{skipped}} card(s) were rejected by Anki (duplicates or invalid).</small>`;
                }}
                status.innerHTML = msg;
              }} catch (err) {{
                console.error("AnkiConnect Browser Bridge error:", err);
                status.className = "status-box status-error";
                let errStr = (err && err.message) ? err.message : String(err);
                let extra = "";
                if (errStr.includes("Failed to fetch") || errStr.includes("NetworkError")) {{
                  extra = "<br><small style='display:inline-block; margin-top:6px;'><b>Could not reach local Anki:</b><br>" +
                          "1. Make sure Anki desktop is open on your computer.<br>" +
                          "2. Verify the AnkiConnect add-on (code <code>2055492159</code>) is installed.<br>" +
                          "3. Check your AnkiConnect CORS settings (see instructions below).</small>";
                }}
                status.innerHTML = `❌ <b>Browser Bridge error:</b> ${{errStr}}${{extra}}`;
              }} finally {{
                btn.disabled = false;
              }}
            }}
          </script>
        </body>
        </html>
        """

    st.divider()
    st.markdown("## 📤 Export")

    # Filter cards eligible for export: non-empty front AND include_in_export is True
    cards_to_export = [
        AnkiCard(
            card_type=c["card_type"],
            front=c["front"],
            back=c["back"],
            intuition=c["intuition"],
            source_reference=c["source_reference"],
            tags=c["tags"],
        )
        for c in st.session_state.staged_cards
        if c.get("front") and c.get("include_in_export", True)
    ]
    total_valid = len([c for c in st.session_state.staged_cards if c.get("front")])
    skipped_count = total_valid - len(cards_to_export)

    deck_name = st.text_input(
        "Deck name",
        value=st.session_state.deck_suggestion,
        help="Use `::` to create nested decks, e.g. `Math::Topology::Chains`",
    )

    col_apkg, col_connect = st.columns([1, 1], gap="large")

    # ── Option A: Universal Export (.apkg) ───────────────────────────────────
    with col_apkg:
        st.markdown("### 📦 Universal Export: Download `.apkg`")
        st.caption("Works offline & everywhere — Desktop (Windows/Mac/Linux) & Mobile (AnkiDroid / AnkiMobile iOS).")

        if len(cards_to_export) > 0:
            with st.spinner("Building .apkg…"):
                try:
                    apkg_bytes = build_apkg(
                        cards_to_export,
                        deck_name,
                        media_map=st.session_state.get("staged_media", {}),
                    )
                    safe_name = deck_name.replace("::", "_").replace(" ", "_")
                    st.download_button(
                        label=f"⬇️ Download {safe_name}.apkg ({len(cards_to_export)} cards)",
                        data=apkg_bytes,
                        file_name=f"{safe_name}.apkg",
                        mime="application/octet-stream",
                        type="primary",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to build .apkg: {e}")
            if skipped_count > 0:
                st.caption(f"ℹ️ {skipped_count} card(s) excluded from .apkg.")
        else:
            st.info("No cards selected for export.")

    # ── Option B: Live Push to Anki ───────────────────────────────────────────
    with col_connect:
        st.markdown("### ⚡ Live Sync: Push to Desktop Anki")
        st.caption("Directly sync flashcards and media into your running Anki desktop app.")

        # Prepare payloads for both Browser Bridge and local push
        notes_payload, media_payload = prepare_ankiconnect_payload(
            cards_to_export,
            deck_name,
            media_map=st.session_state.get("staged_media", {}),
        )

        # 1. Cloud-to-Local Push (Browser Bridge) via st.components.v1.html
        bridge_disabled = (len(cards_to_export) == 0)
        bridge_html = _render_browser_bridge_html(
            notes=notes_payload,
            media=media_payload,
            deck_name=deck_name,
            disabled=bridge_disabled,
        )
        components.html(bridge_html, height=170)

        with st.expander("ℹ️ AnkiConnect CORS Setup (Required for Browser Bridge on Cloud)", expanded=False):
            st.markdown(
                """
                When running AnkiForge on **Streamlit Cloud**, your browser connects directly to Anki on your computer (`http://127.0.0.1:8765`).
                Because of browser security (CORS), AnkiConnect must allow web requests from your browser:

                1. In Anki desktop, go to **Tools** ➔ **Add-ons**.
                2. Select **AnkiConnect** (code: `2055492159`) and click **Config** (bottom-right).
                3. In the config JSON, add `"*"` (or your Streamlit app domain) to `"webCorsOriginList"`:
                ```json
                {
                    "apiKey": null,
                    "apiLogPath": null,
                    "ignoreOriginList": [],
                    "webBindAddress": "127.0.0.1",
                    "webBindPort": 8765,
                    "webCorsOriginList": [
                        "http://localhost",
                        "https://*.streamlit.app",
                        "*"
                    ]
                }
                ```
                4. Click **OK**, then restart Anki.
                """
            )

        # 2. Local Python direct sync (available when running locally on localhost)
        is_live = ankiconnect_is_available()
        if is_live:
            with st.expander("🖥️ Local Host Push (Server-side)", expanded=False):
                st.markdown('<span style="color:#68d391">● Anki detected on host system</span>', unsafe_allow_html=True)
                try:
                    existing_decks = get_deck_names()
                    st.caption(f"Existing decks: {len(existing_decks)} found")
                except Exception:
                    existing_decks = []

                if st.button(
                    f"🚀 Push {len(cards_to_export)} cards via Local Python",
                    type="secondary",
                    use_container_width=True,
                    key="push_anki_local",
                    disabled=(len(cards_to_export) == 0),
                ):
                    with st.spinner("Pushing to Anki…"):
                        try:
                            success, errors = push_cards_to_anki(
                                cards_to_export,
                                deck_name,
                                media_map=st.session_state.get("staged_media", {}),
                            )
                            skip_msg = f" ({skipped_count} skipped)" if skipped_count > 0 else ""
                            st.markdown(
                                f'<div class="success-box">✅ {success} card(s) added to <b>{deck_name}</b>{skip_msg}</div>',
                                unsafe_allow_html=True,
                            )
                            if errors:
                                for err in errors:
                                    st.warning(err)
                        except ConnectionError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"AnkiConnect error: {e}")

