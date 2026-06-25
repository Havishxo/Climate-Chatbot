"""
entity_extraction.py — Named entity recognition and summarization for climate text.

Provides:
  - extract_entities()   : structured JSON extraction via Granite / LLM
  - summarise_document() : map-reduce summarization for long documents
  - extract_from_image() : describe an image and extract entities
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ── Extraction prompt ──────────────────────────────────────────────────────────
ENTITY_EXTRACTION_PROMPT = """You are a climate science information extractor.
Extract the following entities from the text below and return ONLY valid JSON.
Do not include any explanation, preamble, or markdown fences.

Text:
{text}

Extract these fields (use empty list [] if none found):
{{
  "greenhouse_gases": ["list of gases mentioned, e.g. CO2, CH4, N2O"],
  "locations": ["geographic places, regions, countries, oceans"],
  "temperatures": ["temperature values with units, e.g. 1.5°C, 2°F"],
  "co2_values": ["CO2 concentration values, e.g. 421 ppm"],
  "years_periods": ["years or date ranges mentioned"],
  "organizations": ["bodies like IPCC, NOAA, EPA, UNFCCC, NASA"],
  "policy_frameworks": ["agreements, protocols, e.g. Paris Agreement, Kyoto Protocol"],
  "key_statistics": ["any notable numeric facts with units"],
  "sentiment": "positive/negative/neutral (toward climate outlook)"
}}"""


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract structured climate entities from text using the configured LLM.
    Falls back to regex-based extraction if the LLM returns invalid JSON.
    """
    from app.llm_provider import generate, MockLLM, get_llm

    llm = get_llm()
    prompt = ENTITY_EXTRACTION_PROMPT.replace("{text}", text[:2000])

    raw = generate(prompt, llm=llm)

    # Try to parse JSON from the response
    entities = _parse_json_from_response(raw)
    if entities is None:
        logger.info("LLM JSON parse failed — using regex extraction.")
        entities = _regex_fallback(text)

    entities["_source_text_length"] = len(text)
    return entities


def _parse_json_from_response(raw: str) -> Optional[Dict]:
    """Attempt to extract valid JSON from a raw LLM response."""
    raw = raw.strip()
    # Remove markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object in the string
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _regex_fallback(text: str) -> Dict[str, Any]:
    """Basic regex-based entity extraction when LLM JSON fails."""
    gases = re.findall(
        r"\b(CO[₂2]|CH[₄4]|N[₂2]O|HFCs?|PFCs?|SF[₆6]|nitrous oxide|methane|carbon dioxide)\b",
        text, re.IGNORECASE
    )
    temps = re.findall(r"[-+]?\d+\.?\d*\s*°[CF]", text)
    ppm_vals = re.findall(r"\d+\.?\d*\s*ppm", text, re.IGNORECASE)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    orgs = re.findall(
        r"\b(IPCC|NOAA|NASA|EPA|UNFCCC|IEA|NSIDC|WMO|WHO)\b", text
    )
    locs = re.findall(
        r"\b(Arctic|Antarctic|Greenland|Amazon|Pacific|Atlantic|Indian Ocean|"
        r"Himalaya|Sahel|Mediterranean|tropics?)\b", text, re.IGNORECASE
    )
    agreements = re.findall(
        r"\b(Paris Agreement|Kyoto Protocol|COP\d+|NDC|Nationally Determined)\b",
        text, re.IGNORECASE
    )

    return {
        "greenhouse_gases": list(dict.fromkeys(g.upper() for g in gases)),
        "locations": list(dict.fromkeys(locs)),
        "temperatures": list(dict.fromkeys(temps)),
        "co2_values": list(dict.fromkeys(ppm_vals)),
        "years_periods": list(dict.fromkeys(years))[:10],
        "organizations": list(dict.fromkeys(orgs)),
        "policy_frameworks": list(dict.fromkeys(agreements)),
        "key_statistics": [],
        "sentiment": "neutral",
        "_method": "regex_fallback",
    }


# ── Summarization ──────────────────────────────────────────────────────────────
def summarise_document(
    text: str,
    max_summary_words: int = 200,
    style: str = "factual",   # factual | accessible | bullet
) -> Dict[str, Any]:
    """
    Summarise a climate document using a map-reduce strategy for long texts.
    Short texts (<1000 words) are summarised directly.
    """
    words = text.split()
    if len(words) <= 1000:
        return _direct_summarise(text, max_summary_words, style)
    return _map_reduce_summarise(text, max_summary_words, style)


def _direct_summarise(text: str, max_words: int, style: str) -> Dict[str, Any]:
    from app.llm_provider import generate

    style_instruction = {
        "factual": "Focus on key facts, statistics, and scientific conclusions.",
        "accessible": "Use simple language accessible to a general audience. Avoid jargon.",
        "bullet": "Return 5-7 bullet points covering the key findings.",
    }.get(style, "Focus on key facts and conclusions.")

    prompt = (
        f"Summarise the following climate-related text in approximately {max_words} words. "
        f"{style_instruction}\n\nText:\n{text[:3000]}"
    )
    summary = generate(prompt)
    return {
        "summary": summary,
        "strategy": "direct",
        "original_words": len(text.split()),
        "style": style,
    }


def _map_reduce_summarise(text: str, max_words: int, style: str) -> Dict[str, Any]:
    """Summarise long text by chunking, summarising each chunk, then combining."""
    from app.llm_provider import generate

    # Split into ~800-word chunks
    words = text.split()
    chunk_size = 800
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    # Map: summarise each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:8]):  # cap at 8 chunks
        prompt = (
            f"Summarise this passage from a climate document in 2-3 sentences, "
            f"focusing on key facts and numbers:\n\n{chunk}"
        )
        summary = generate(prompt)
        chunk_summaries.append(summary)
        logger.debug("Map step %d/%d done.", i + 1, len(chunks))

    # Reduce: combine chunk summaries
    combined = "\n\n".join(chunk_summaries)
    style_instruction = {
        "factual": "key facts, statistics, and scientific conclusions",
        "accessible": "plain language accessible to a general audience",
        "bullet": "5-7 bullet points covering the key findings",
    }.get(style, "key facts")

    reduce_prompt = (
        f"You have these partial summaries of a climate document. "
        f"Write a coherent final summary of approximately {max_words} words covering {style_instruction}:\n\n"
        f"{combined[:3000]}"
    )
    final_summary = generate(reduce_prompt)

    return {
        "summary": final_summary,
        "strategy": "map_reduce",
        "chunks_processed": len(chunk_summaries),
        "original_words": len(words),
        "style": style,
    }


# ── Image description + entity extraction ─────────────────────────────────────
def describe_and_extract_from_image(image_path: str) -> Dict[str, Any]:
    """
    Generate a text description of a climate image (chart, satellite, etc.)
    then extract entities from that description.
    Falls back to a filename-based description if vision is unavailable.
    """
    description = _describe_image(image_path)
    entities = extract_entities(description)
    return {
        "description": description,
        "entities": entities,
        "image_path": image_path,
    }


def _describe_image(image_path: str) -> str:
    """
    Attempt to describe an image using available vision capabilities.
    For Anthropic API, uses base64 image input.
    Falls back to generic description based on filename keywords.
    """
    import os
    filename = os.path.basename(image_path).lower()

    # Try Anthropic vision if key available
    from app.config import ANTHROPIC_API_KEY, USE_ANTHROPIC
    if USE_ANTHROPIC:
        try:
            return _describe_with_anthropic(image_path)
        except Exception as exc:
            logger.warning("Anthropic vision failed: %s", exc)

    # Filename-keyword fallback
    if any(k in filename for k in ["temp", "temperature", "anomaly"]):
        return "A climate chart showing global temperature anomaly trends over time."
    if any(k in filename for k in ["co2", "carbon", "emissions"]):
        return "A chart displaying CO2 concentration or carbon emissions data."
    if any(k in filename for k in ["sea", "ice", "arctic"]):
        return "A satellite or data image related to sea ice or Arctic conditions."
    if any(k in filename for k in ["solar", "wind", "renewable"]):
        return "An image related to renewable energy sources."
    return (
        f"An uploaded climate-related image (file: {os.path.basename(image_path)}). "
        "Please describe what you see in this image in your question."
    )


def _describe_with_anthropic(image_path: str) -> str:
    """Use Anthropic's Claude vision to describe an image."""
    import base64
    import anthropic
    from app.config import ANTHROPIC_API_KEY

    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    media_type = media_type_map.get(ext, "image/png")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": media_type, "data": img_data}},
                    {"type": "text", "text": (
                        "Describe this climate-related image in detail. "
                        "Include any visible data, trends, locations, temperatures, "
                        "dates, or scientific measurements."
                    )},
                ],
            }
        ],
    )
    return message.content[0].text
