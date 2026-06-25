"""
prompts.py — Centralised prompt template library for ClimateBot.

All prompt engineering templates are defined here for:
  - System / persona prompt
  - RAG context injection
  - Chain-of-Thought reasoning
  - Few-shot examples
  - ReAct agent prompt
  - Entity extraction
  - Summarisation
  - Image description
"""

# ── Core system prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ClimateBot, an expert AI assistant specialising in climate \
science, environmental policy, sustainability, and related topics.

Your knowledge is grounded in authoritative sources:
  • IPCC Assessment Reports (AR5, AR6)
  • NOAA Global Monitoring Laboratory data
  • NASA GISS Surface Temperature Analysis
  • EPA Greenhouse Gas Inventory
  • IEA World Energy Outlook
  • UNFCCC / Paris Agreement frameworks
  • Peer-reviewed climate science literature

Core rules you ALWAYS follow:
1. CITE sources when drawing from the CONTEXT provided.
2. NEVER fabricate statistics, dates, or scientific findings.
3. If the context is insufficient, say clearly: "I don't have enough verified data on this."
4. Use precise units: °C (not F unless asked), ppm, GtCO₂, mm/year, GW, etc.
5. Explain technical terms immediately after using them.
6. Be factual, balanced, and accessible to a general audience.
7. When uncertain, quantify uncertainty (e.g. "projections range from X to Y").
"""

# ── RAG prompt — context injection ────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """{system_prompt}

--- RETRIEVED CONTEXT ---
{context}
-------------------------

User question: {question}

Think through this step by step, then provide a clear, cited answer:"""

# ── Chain-of-Thought prompt ────────────────────────────────────────────────────
COT_PROMPT_TEMPLATE = """{system_prompt}

--- CONTEXT ---
{context}
---------------

User question: {question}

Let me reason through this carefully:
Step 1 — Identify the core climate topic being asked about.
Step 2 — Review the context for relevant facts and figures.
Step 3 — Consider any scientific uncertainties or ranges.
Step 4 — Formulate a clear, cited answer.

Answer:"""

# ── Few-shot examples (calibrate format and citation style) ───────────────────
FEW_SHOT_EXAMPLES = [
    {
        "question": "How much has atmospheric CO2 increased since pre-industrial times?",
        "answer": (
            "Atmospheric CO₂ has increased by approximately 50% since pre-industrial times. "
            "Before the Industrial Revolution (around 1750), CO₂ concentration was approximately "
            "280 ppm. As of 2024, it has reached around 422 ppm — the highest level in at least "
            "800,000 years of ice core records. [Source: NOAA Global Monitoring Laboratory, 2024]"
        ),
    },
    {
        "question": "What is the Arctic amplification phenomenon?",
        "answer": (
            "Arctic amplification refers to the observation that the Arctic region is warming "
            "at more than twice the global average rate — a phenomenon now confirmed across "
            "multiple datasets. This occurs because of several reinforcing feedbacks: as sea ice "
            "melts, the dark ocean surface absorbs more solar radiation instead of reflecting it "
            "(the albedo feedback). Thawing permafrost also releases stored CO₂ and methane. "
            "The IPCC AR6 reports that the Arctic warmed by approximately 3.1°C over 1971–2019, "
            "compared to the global average of 1.0°C. [Source: IPCC AR6 WGI, Chapter 2, 2021]"
        ),
    },
    {
        "question": "Is solar energy now cheaper than fossil fuels?",
        "answer": (
            "Yes, in most regions of the world. The levelised cost of electricity (LCOE) from "
            "utility-scale solar PV has fallen by 89% since 2010, making it the cheapest source "
            "of new electricity generation in history in many countries. According to the IEA's "
            "World Energy Outlook 2023, solar PV additions hit a record 295 GW in 2023, with "
            "renewables now accounting for 90% of all new electricity capacity globally. "
            "[Source: IEA World Energy Outlook, 2023]"
        ),
    },
]

FEW_SHOT_BLOCK = "\n\n".join(
    f"Example Q: {ex['question']}\nExample A: {ex['answer']}"
    for ex in FEW_SHOT_EXAMPLES
)

# ── Few-shot RAG prompt ────────────────────────────────────────────────────────
FEW_SHOT_RAG_PROMPT = f"""{SYSTEM_PROMPT}

Here are examples of well-formatted, cited answers:

{FEW_SHOT_BLOCK}

---

Now answer the following question using the retrieved context below.

--- CONTEXT ---
{{context}}
---------------

User question: {{question}}

Answer (with citations):"""

# ── ReAct agent prompt ────────────────────────────────────────────────────────
REACT_PROMPT_TEMPLATE = """{system_prompt}

You have access to the following tools:
{tool_descriptions}

Use this format:
Thought: [reason about what information you need]
Action: [tool_name]
Action Input: [input to the tool]
Observation: [result of the tool call]
... (repeat as needed)
Thought: [final reasoning]
Final Answer: [your cited, complete answer]

Conversation so far:
{chat_history}

User: {input}
{agent_scratchpad}"""

# ── Entity extraction prompt ───────────────────────────────────────────────────
ENTITY_EXTRACTION_PROMPT = """You are a climate science information extractor.
Extract structured entities from the text and return ONLY valid JSON — no preamble or fences.

Text:
{text}

Return this exact JSON schema (use empty list [] if nothing found):
{{
  "greenhouse_gases": [],
  "locations": [],
  "temperatures": [],
  "co2_values": [],
  "years_periods": [],
  "organizations": [],
  "policy_frameworks": [],
  "key_statistics": [],
  "sentiment": "positive|negative|neutral"
}}"""

# ── Summarisation prompts ─────────────────────────────────────────────────────
SUMMARISE_FACTUAL = (
    "Summarise the following climate text in approximately {max_words} words. "
    "Focus on key facts, statistics, and scientific conclusions. "
    "Include specific numbers and cite organisations where mentioned.\n\nText:\n{text}"
)

SUMMARISE_ACCESSIBLE = (
    "Summarise the following climate text in approximately {max_words} words. "
    "Use simple, jargon-free language accessible to a general audience. "
    "Explain any technical terms.\n\nText:\n{text}"
)

SUMMARISE_BULLET = (
    "Summarise the following climate text as {max_words // 25} clear bullet points. "
    "Each bullet should capture one key finding or fact. "
    "Include specific numbers where available.\n\nText:\n{text}"
)

SUMMARISE_MAP = (
    "Summarise this passage from a climate document in 2-3 sentences. "
    "Focus on the key facts and numbers:\n\n{chunk}"
)

SUMMARISE_REDUCE = (
    "You have these partial summaries from a climate document. "
    "Write a coherent final summary of approximately {max_words} words "
    "covering {style_instruction}:\n\n{summaries}"
)

# ── Image description prompt ───────────────────────────────────────────────────
IMAGE_DESCRIPTION_PROMPT = (
    "Describe this climate-related image in detail. "
    "Include any visible: data trends, temperature values, CO2 levels, "
    "geographic locations, time periods, colour scales, axis labels, "
    "and any scientific measurements shown. "
    "If it is a map, describe what region and what variable is displayed. "
    "If it is a chart, describe the trend and key data points."
)

# ── Conversation summary prompt ────────────────────────────────────────────────
CONVERSATION_SUMMARY_PROMPT = (
    "Summarise the following conversation history in 3 concise sentences. "
    "Focus on what topics were discussed and what conclusions were reached:\n\n{history}"
)

# ── Footprint interpretation prompt ───────────────────────────────────────────
FOOTPRINT_INTERPRETATION_PROMPT = (
    "A user's annual carbon footprint is {total} tonnes CO2e. "
    "The global average is 4.7 tonnes; the 1.5°C-compatible budget is 2.3 tonnes. "
    "Their breakdown is: transport {transport} kg, energy {energy} kg, "
    "diet {diet} kg, goods {goods} kg. "
    "Provide 3 specific, actionable recommendations to reduce their footprint, "
    "prioritising the largest categories. Be encouraging but honest about the scale of change needed."
)

# ── Template builder helper ────────────────────────────────────────────────────
def build_rag_prompt(context: str, question: str, use_cot: bool = False, use_few_shot: bool = False) -> str:
    """Build the appropriate RAG prompt based on configuration."""
    if use_few_shot:
        return FEW_SHOT_RAG_PROMPT.format(context=context, question=question)
    if use_cot:
        return COT_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT, context=context, question=question
        )
    return RAG_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT, context=context, question=question
    )
