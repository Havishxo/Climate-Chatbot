"""
multimodal.py — Unified multimodal input router.

Handles:
  - PDF files   → extract text, chunk, ingest to RAG
  - Images      → describe (vision model) → extract entities
  - CSV/Excel   → convert to text → ingest to RAG
  - Plain text  → ingest directly
"""

import logging
import os
import tempfile
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def process_upload(file_path: str, original_name: str = "") -> Dict[str, Any]:
    """
    Detect file type and process accordingly.
    Returns:
      {
        "type": str,           # "pdf" | "image" | "csv" | "text" | "unknown"
        "text_extracted": str, # extracted/described text
        "chunks_added": int,   # chunks added to vector store
        "description": str,    # human-friendly summary of what was processed
        "entities": dict,      # extracted entities (if available)
        "error": str | None,
      }
    """
    name = original_name.lower() or os.path.basename(file_path).lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext == "pdf":
        return _process_pdf(file_path)
    elif ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
        return _process_image(file_path)
    elif ext in ("csv", "tsv"):
        return _process_csv(file_path, sep="\t" if ext == "tsv" else ",")
    elif ext in ("txt", "md", "rst"):
        return _process_text(file_path)
    else:
        return {
            "type": "unknown",
            "text_extracted": "",
            "chunks_added": 0,
            "description": f"Unsupported file type: .{ext}",
            "entities": {},
            "error": f"Unsupported extension: .{ext}",
        }


# ── PDF processing ─────────────────────────────────────────────────────────────
def _process_pdf(file_path: str) -> Dict[str, Any]:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages_text.append(text)
        doc.close()

        full_text = "\n\n".join(pages_text)

        # Ingest into RAG
        from app.rag import ingest_pdf
        chunks = ingest_pdf(file_path)

        # Extract entities from first 2000 chars
        from app.entity_extraction import extract_entities
        entities = extract_entities(full_text[:2000])

        return {
            "type": "pdf",
            "text_extracted": full_text,
            "chunks_added": chunks,
            "pages": len(pages_text),
            "description": (
                f"✅ PDF processed: {len(pages_text)} pages, "
                f"{len(full_text.split())} words, "
                f"{chunks} chunks added to knowledge base."
            ),
            "entities": entities,
            "error": None,
        }
    except Exception as exc:
        logger.error("PDF processing error: %s", exc)
        return {
            "type": "pdf",
            "text_extracted": "",
            "chunks_added": 0,
            "description": f"❌ PDF processing failed: {exc}",
            "entities": {},
            "error": str(exc),
        }


# ── Image processing ───────────────────────────────────────────────────────────
def _process_image(file_path: str) -> Dict[str, Any]:
    try:
        from app.entity_extraction import describe_and_extract_from_image
        result = describe_and_extract_from_image(file_path)
        description = result.get("description", "")

        # Ingest description into RAG
        from app.rag import ingest_text
        chunks = 0
        if description:
            chunks = ingest_text(
                description,
                source=os.path.basename(file_path),
                topic="image-description",
            )

        return {
            "type": "image",
            "text_extracted": description,
            "chunks_added": chunks,
            "description": (
                f"✅ Image analysed. Description: {description[:200]}..."
                if len(description) > 200 else f"✅ Image analysed: {description}"
            ),
            "entities": result.get("entities", {}),
            "error": None,
        }
    except Exception as exc:
        logger.error("Image processing error: %s", exc)
        return {
            "type": "image",
            "text_extracted": "",
            "chunks_added": 0,
            "description": f"❌ Image processing failed: {exc}",
            "entities": {},
            "error": str(exc),
        }


# ── CSV processing ─────────────────────────────────────────────────────────────
def _process_csv(file_path: str, sep: str = ",") -> Dict[str, Any]:
    try:
        import pandas as pd

        df = pd.read_csv(file_path, sep=sep, nrows=500)
        rows, cols = df.shape

        # Generate a text representation
        text = f"Dataset: {os.path.basename(file_path)}\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n"
        text += f"Shape: {rows} rows × {cols} columns\n\n"
        text += "First 20 rows:\n"
        text += df.head(20).to_string(index=False)

        # Ingest into RAG
        from app.rag import ingest_text
        chunks = ingest_text(
            text,
            source=os.path.basename(file_path),
            topic="dataset",
        )

        # Basic stats for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        stats_text = ""
        if numeric_cols:
            stats = df[numeric_cols].describe().round(3)
            stats_text = f"\nColumn statistics:\n{stats.to_string()}"
            ingest_text(stats_text, source=f"{os.path.basename(file_path)}_stats", topic="dataset")

        return {
            "type": "csv",
            "text_extracted": text + stats_text,
            "chunks_added": chunks,
            "rows": rows,
            "columns": cols,
            "column_names": df.columns.tolist(),
            "description": (
                f"✅ CSV loaded: {rows} rows × {cols} columns. "
                f"Columns: {', '.join(df.columns.tolist()[:6])}{'...' if cols > 6 else ''}. "
                f"{chunks} chunks added to knowledge base."
            ),
            "entities": {},
            "error": None,
        }
    except Exception as exc:
        logger.error("CSV processing error: %s", exc)
        return {
            "type": "csv",
            "text_extracted": "",
            "chunks_added": 0,
            "description": f"❌ CSV processing failed: {exc}",
            "entities": {},
            "error": str(exc),
        }


# ── Plain text processing ──────────────────────────────────────────────────────
def _process_text(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        from app.rag import ingest_text
        chunks = ingest_text(
            text,
            source=os.path.basename(file_path),
            topic="user-document",
        )

        from app.entity_extraction import extract_entities
        entities = extract_entities(text[:2000])

        return {
            "type": "text",
            "text_extracted": text,
            "chunks_added": chunks,
            "description": (
                f"✅ Text file processed: {len(text.split())} words, "
                f"{chunks} chunks added to knowledge base."
            ),
            "entities": entities,
            "error": None,
        }
    except Exception as exc:
        return {
            "type": "text",
            "text_extracted": "",
            "chunks_added": 0,
            "description": f"❌ Text file processing failed: {exc}",
            "entities": {},
            "error": str(exc),
        }


# ── Image validation ───────────────────────────────────────────────────────────
def validate_image(file_path: str) -> Tuple[bool, str]:
    """Check if an image file is valid and within size limits."""
    try:
        from PIL import Image
        img = Image.open(file_path)
        img.verify()
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 20:
            return False, f"Image too large ({size_mb:.1f} MB). Max 20 MB."
        return True, f"{img.format} image, {img.size[0]}×{img.size[1]}px, {size_mb:.1f} MB"
    except Exception as exc:
        return False, f"Invalid image: {exc}"
