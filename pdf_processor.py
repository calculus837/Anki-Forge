"""
pdf_processor.py
----------------
Handles PDF operations:
  - Page count inspection via PyMuPDF (fitz)
  - Slicing a page range into an in-memory PDF bytes buffer
  - Building a Gemini types.Part for the sliced PDF
"""

import io
from typing import Tuple

import fitz  # PyMuPDF
from google.genai import types


def get_page_count(pdf_bytes: bytes) -> int:
    """
    Return the total number of pages in a PDF given its raw bytes.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Integer page count.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = doc.page_count
    doc.close()
    return count


def slice_pdf(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    """
    Extract a range of pages from a PDF and return them as bytes.

    Pages are 1-indexed (inclusive on both ends) for user-friendliness;
    PyMuPDF uses 0-indexed internally.

    Args:
        pdf_bytes:  Raw bytes of the source PDF.
        start_page: First page to include (1-indexed).
        end_page:   Last page to include (1-indexed, inclusive).

    Returns:
        Raw bytes of the sliced PDF.

    Raises:
        ValueError: If page numbers are out of range or start > end.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = doc.page_count

    if start_page < 1 or end_page > total or start_page > end_page:
        doc.close()
        raise ValueError(
            f"Invalid page range [{start_page}, {end_page}] "
            f"for a {total}-page document."
        )

    # Build a new document with only the selected pages (0-indexed)
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)

    buf = io.BytesIO()
    new_doc.save(buf)
    new_doc.close()
    doc.close()

    return buf.getvalue()


def pdf_bytes_to_part(pdf_bytes: bytes) -> types.Part:
    """
    Wrap raw PDF bytes in a Gemini types.Part ready for multimodal generation.

    Args:
        pdf_bytes: Raw bytes of the PDF (typically already sliced).

    Returns:
        A types.Part with mime_type="application/pdf".
    """
    return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def image_bytes_to_part(image_bytes: bytes, mime_type: str = "image/png") -> types.Part:
    """
    Wrap raw image bytes in a Gemini types.Part.

    Args:
        image_bytes: Raw bytes of the image.
        mime_type:   MIME type string, e.g. "image/png" or "image/jpeg".

    Returns:
        A types.Part with the appropriate mime_type.
    """
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def render_page_thumbnail(pdf_bytes: bytes, page_number: int, zoom: float = 1.5) -> bytes:
    """
    Render a single PDF page as a PNG thumbnail for preview.

    Args:
        pdf_bytes:   Raw bytes of the PDF.
        page_number: 1-indexed page number to render.
        zoom:        Scale factor for rendering resolution.

    Returns:
        PNG image bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
