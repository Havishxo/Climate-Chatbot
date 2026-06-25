"""
agent.py — Agentic AI orchestrator for the Climate Awareness Chatbot.

Implements a lightweight ReAct-style agent that:
  1. Analyses the user's intent
  2. Decides which tools to call
  3. Integrates tool results
  4. Returns a coherent, cited answer
"""

import json
import logging
import re
from typing import Dict, Any, List, Tuple

from app.tools import call_tool, describe_tools, TOOL_REGISTRY
from app.rag import rag_query
from app.llm_provider import generate
from app.entity_extraction import extract_entities, summarise_document

logger = logging.getLogger(__name__)

# ── Intent classification ─────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "live_co2": re.compile(
        r"(current|latest|today'?s?|real.?time|now).*(co2|carbon dioxide|ppm)|"
        r"co2.*(right now|current|latest|today)|what is.*co2",
        re.IGNORECASE,
    ),
    "temperature": re.compile(
        r"(current|latest|today'?s?).*temp(erature)?|"
        r"temp.*anomaly|how (hot|warm) is|warming.*now",
        re.IGNORECASE,
    ),
    "carbon_footprint": re.compile(
        r"(carbon footprint|my.*emission|calculate.*carbon|how much.*co2.*i|"
        r"personal.*carbon|footprint.*calcul)",
        re.IGNORECASE,
    ),
    "summarise": re.compile(
        r"(summar|summarize|summarise|tldr|brief|overview|key points|main points)",
        re.IGNORECASE,
    ),
    "extract_entities": re.compile(
        r"(extract|identify|find|list|what entities|named entity|what gases|"
        r"which countries|what organizations)",
        re.IGNORECASE,
    ),
}


def classify_intent(query: str) -> List[str]:
    """Return a list of detected intents for the query."""
    intents = []
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(query):
            intents.append(intent)
    if not intents:
        intents.append("general_rag")
    return intents


# ── Conversation memory ────────────────────────────────────────────────────────
class ConversationMemory:
    """Rolling conversation memory with automatic summarisation."""

    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns
        self.summary: str = ""

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self._summarise_old_turns()

    def _summarise_old_turns(self):
        """Collapse the oldest half of the history into a summary."""
        half = len(self.history) // 2
        old_turns = self.history[:half]
        self.history = self.history[half:]
        old_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content'][:200]}" for t in old_turns
        )
        new_summary = generate(
            f"Summarise this conversation history in 3 sentences:\n\n{old_text}"
        )
        self.summary = new_summary if not self.summary else (
            f"{self.summary}\n[Later:] {new_summary}"
        )

    def get_context_string(self, last_n: int = 4) -> str:
        """Return recent history + summary as a formatted string."""
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary]: {self.summary}")
        for turn in self.history[-last_n * 2 :]:
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content'][:300]}")
        return "\n".join(parts)

    def clear(self):
        self.history = []
        self.summary = ""


# ── Main agent function ────────────────────────────────────────────────────────
def agent_respond(
    query: str,
    memory: ConversationMemory,
    uploaded_text: str = "",
) -> Dict[str, Any]:
    """
    Main entry point. Returns:
      {
        "answer": str,
        "sources": list,
        "tool_calls": list,
        "entities": dict,
        "context_used": bool,
      }
    """
    result = {
        "answer": "",
        "sources": [],
        "tool_calls": [],
        "entities": {},
        "context_used": False,
    }

    intents = classify_intent(query)
    logger.info("Query intents: %s", intents)

    # ── Uploaded document context ──────────────────────────────────────────────
    uploaded_context = ""
    if uploaded_text:
        uploaded_context = f"\n[Uploaded document excerpt]:\n{uploaded_text[:1500]}\n"
        # Auto-ingest into RAG
        from app.rag import ingest_text
        ingest_text(uploaded_text, source="user-uploaded-session", topic="user-document")

    # ── Tool calls based on intent ─────────────────────────────────────────────
    tool_results_text = ""

    if "live_co2" in intents:
        co2 = call_tool("get_live_co2")
        result["tool_calls"].append({"tool": "get_live_co2", "result": co2})
        tool_results_text += (
            f"\n[Live CO2 Data — {co2.get('source', 'N/A')}]: "
            f"Current atmospheric CO2 = {co2.get('ppm', 'N/A')} ppm "
            f"(trend: {co2.get('trend_ppm', 'N/A')} ppm, "
            f"{co2.get('year', '')}/{co2.get('month', '')})\n"
        )

    if "temperature" in intents:
        temp = call_tool("get_temperature_anomaly")
        result["tool_calls"].append({"tool": "get_temperature_anomaly", "result": temp})
        tool_results_text += (
            f"\n[Live Temperature Data — {temp.get('source', 'N/A')}]: "
            f"Global temperature anomaly = +{temp.get('anomaly_c', 'N/A')} °C "
            f"above {temp.get('baseline', '1951-1980')} baseline "
            f"(year: {temp.get('year', 'N/A')})\n"
        )

    if "carbon_footprint" in intents:
        # Parse numeric values from query with basic regex
        fp_params = _parse_footprint_params(query)
        fp = call_tool("calculate_carbon_footprint", **fp_params)
        result["tool_calls"].append({"tool": "calculate_carbon_footprint", "result": fp})
        tool_results_text += (
            f"\n[Carbon Footprint Calculation]: "
            f"Estimated footprint = {fp.get('total_tonnes_co2e', 'N/A')} tonnes CO2e/year. "
            f"Breakdown: {fp.get('breakdown', {})}. "
            f"Tips: {' | '.join(fp.get('tips', []))}\n"
        )

    if "extract_entities" in intents and uploaded_text:
        entities = extract_entities(uploaded_text or query)
        result["entities"] = entities
        tool_results_text += (
            f"\n[Extracted Entities]: {json.dumps(entities, indent=2)[:500]}\n"
        )

    if "summarise" in intents and (uploaded_text or len(query) > 300):
        text_to_summarise = uploaded_text if uploaded_text else query
        summary_result = summarise_document(text_to_summarise, max_summary_words=150)
        tool_results_text += (
            f"\n[Document Summary]: {summary_result.get('summary', '')}\n"
        )

    # ── RAG retrieval for all queries ──────────────────────────────────────────
    rag_result = rag_query(query)
    result["sources"] = rag_result.get("sources", [])
    result["context_used"] = rag_result.get("context_used", False)

    # Format retrieved context
    rag_context = ""
    for src in result["sources"][:3]:
        rag_context += f"\n[{src['source']}]: {src['excerpt'][:300]}\n"

    # ── Build final response prompt ────────────────────────────────────────────
    conv_context = memory.get_context_string(last_n=3)

    final_prompt = f"""You are ClimateBot, an AI climate science expert.
Use the information below to answer the user's question accurately and concisely.
Always cite sources from the CONTEXT when available.

CONVERSATION HISTORY:
{conv_context if conv_context else "(new conversation)"}

RETRIEVED CLIMATE KNOWLEDGE:
{rag_context if rag_context else "(no context retrieved)"}

TOOL RESULTS:
{tool_results_text if tool_results_text else "(no tools invoked)"}

UPLOADED DOCUMENT:
{uploaded_context if uploaded_context else "(none)"}

USER QUESTION: {query}

Instructions: Provide a clear, fact-based answer. Cite sources. If you used live data, \
mention it. If data is unavailable, say so. Be concise but thorough."""

    answer = generate(final_prompt)
    result["answer"] = answer

    # Update memory
    memory.add("user", query)
    memory.add("assistant", answer[:500])

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_footprint_params(query: str) -> Dict[str, Any]:
    """Extract carbon footprint parameters from a natural language query."""
    params: Dict[str, Any] = {}

    km_match = re.search(r"(\d+[\d,]*)\s*(km|kilometers|miles)\s*(a year|per year|annually)?", query, re.IGNORECASE)
    if km_match:
        val = float(km_match.group(1).replace(",", ""))
        if "mile" in km_match.group(2).lower():
            val *= 1.609
        params["car_km_year"] = val

    flights_short = re.search(r"(\d+)\s*short.?haul flights?", query, re.IGNORECASE)
    if flights_short:
        params["flights_short"] = int(flights_short.group(1))

    flights_long = re.search(r"(\d+)\s*long.?haul flights?", query, re.IGNORECASE)
    if flights_long:
        params["flights_long"] = int(flights_long.group(1))

    elec_match = re.search(r"(\d+)\s*kwh\s*(per|a)\s*month", query, re.IGNORECASE)
    if elec_match:
        params["electricity_kwh_month"] = float(elec_match.group(1))

    if re.search(r"\bvegan\b", query, re.IGNORECASE):
        params["diet_type"] = "vegan"
    elif re.search(r"\bvegetar", query, re.IGNORECASE):
        params["diet_type"] = "vegetarian"
    elif re.search(r"\bmeat.?heavy\b", query, re.IGNORECASE):
        params["diet_type"] = "meat_heavy"

    return params
