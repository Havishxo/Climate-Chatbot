"""
ui.py — Gradio 6-compatible multimodal chat interface for ClimateBot.

Fixes applied vs original:
  1. Chatbot history now uses messages format (list of dicts with role/content)
     instead of tuples — required by Gradio 6.x
  2. Removed show_copy_button (not supported in Gradio 6)
  3. Removed gr.themes.Soft + CSS (kept simple for Gradio 6 compat)
  4. chat() and clear_chat() return correct messages-format history
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent import agent_respond, ConversationMemory
from app.multimodal import process_upload
from app.tools import get_live_co2, get_temperature_anomaly, calculate_carbon_footprint
from app.entity_extraction import extract_entities, summarise_document
from app.rag import get_vectorstore_stats, ingest_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global conversation memory ────────────────────────────────────────────────
_memory = ConversationMemory(max_turns=12)
_uploaded_text_cache: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n📚 **Sources retrieved:**"]
    seen = set()
    for s in sources[:4]:
        src = s.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            lines.append(f"• {src}")
    return "\n".join(lines)


def _format_entities(entities: dict) -> str:
    if not entities:
        return "No entities extracted."
    lines = []
    label_map = {
        "greenhouse_gases": "🌫️ Greenhouse Gases",
        "locations": "📍 Locations",
        "temperatures": "🌡️ Temperatures",
        "co2_values": "📊 CO₂ Values",
        "years_periods": "📅 Years / Periods",
        "organizations": "🏛️ Organizations",
        "policy_frameworks": "📜 Policy Frameworks",
        "key_statistics": "📈 Key Statistics",
        "sentiment": "💬 Sentiment",
    }
    for key, label in label_map.items():
        val = entities.get(key)
        if val and val != [] and val != "":
            if isinstance(val, list):
                lines.append(f"**{label}:** {', '.join(str(v) for v in val[:8])}")
            else:
                lines.append(f"**{label}:** {val}")
    return "\n\n".join(lines) if lines else "No entities found."


# ── FIX 1: Chat handler — returns messages-format history (list of dicts) ─────
def chat(
    message: str,
    history: List[Dict[str, str]],   # messages format: [{"role":..,"content":..}]
    file_obj,
) -> tuple:
    global _uploaded_text_cache

    if not message.strip():
        return history, ""

    history = history or []

    # Process uploaded file
    upload_info = ""
    if file_obj is not None:
        try:
            result = process_upload(
                file_obj.name,
                original_name=os.path.basename(file_obj.name),
            )
            _uploaded_text_cache = result.get("text_extracted", "")
            upload_info = f"\n\n📎 **{result['description']}**"
        except Exception as exc:
            upload_info = f"\n\n⚠️ File processing error: {exc}"

    # Run agent
    try:
        response = agent_respond(
            query=message,
            memory=_memory,
            uploaded_text=_uploaded_text_cache[:3000],
        )
        answer = response["answer"]
        sources_text = _format_sources(response.get("sources", []))

        tool_badges = ""
        for tc in response.get("tool_calls", []):
            tool_badges += f"\n🔧 `{tc['tool']}` called"

        full_answer = answer + tool_badges + sources_text + upload_info

    except Exception as exc:
        logger.error("Agent error: %s", exc)
        full_answer = f"⚠️ Error: {exc}"

    # ✅ FIX: append dicts, not tuples
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_answer})
    return history, ""


# ── FIX 2: clear_chat returns [] not ([], "") with messages format ─────────────
def clear_chat():
    global _uploaded_text_cache
    _memory.clear()
    _uploaded_text_cache = ""
    return [], ""   # empty messages list, empty textbox


# ── Live data ─────────────────────────────────────────────────────────────────
def refresh_live_data():
    co2 = get_live_co2()
    temp = get_temperature_anomaly()

    co2_text = (
        f"### 🌍 Atmospheric CO₂\n"
        f"**{co2.get('ppm', 'N/A')} ppm**\n\n"
        f"Trend: {co2.get('trend_ppm', 'N/A')} ppm\n"
        f"Period: {co2.get('year', '')}/{co2.get('month', '')}\n"
        f"*{co2.get('source', '')}*"
    )
    temp_text = (
        f"### 🌡️ Temperature Anomaly\n"
        f"**+{temp.get('anomaly_c', 'N/A')} °C**\n\n"
        f"Above {temp.get('baseline', '1951–1980')} baseline\n"
        f"Year: {temp.get('year', 'N/A')}\n"
        f"*{temp.get('source', '')}*"
    )
    status = (
        f"**CO₂:** {co2.get('status', '?')} | "
        f"**Temp:** {temp.get('status', '?')}"
    )
    return co2_text, temp_text, status


# ── Carbon footprint ──────────────────────────────────────────────────────────
def run_footprint_calc(car_km, short_flights, long_flights,
                       elec_kwh, gas_kwh, diet, spend):
    result = calculate_carbon_footprint(
        car_km_year=float(car_km or 0),
        flights_short=int(short_flights or 0),
        flights_long=int(long_flights or 0),
        electricity_kwh_month=float(elec_kwh or 0),
        natural_gas_kwh_month=float(gas_kwh or 0),
        diet_type=str(diet).lower().replace(" ", "_"),
        goods_spend_monthly_usd=float(spend or 200),
    )
    total = result["total_tonnes_co2e"]
    bd = result["breakdown"]
    comp = result["comparison"]
    tips = result["tips"]

    return (
        f"## 🌱 Your Annual Carbon Footprint\n\n"
        f"### **{total} tonnes CO₂e / year**\n\n"
        f"---\n"
        f"**Breakdown:**\n"
        f"- 🚗 Transport: {bd['transport_kg']:.0f} kg\n"
        f"- ⚡ Energy: {bd['energy_kg']:.0f} kg\n"
        f"- 🥗 Diet: {bd['diet_kg']:.0f} kg\n"
        f"- 🛍️ Goods & Services: {bd['goods_kg']:.0f} kg\n\n"
        f"---\n"
        f"**How you compare:**\n"
        f"- Global average (4.7 t): **{comp['vs_global_avg']}×**\n"
        f"- 1.5°C budget (2.3 t): **{comp['vs_1_5c_budget']}×**\n\n"
        f"---\n"
        f"**💡 Reduction tips:**\n"
        + "\n".join(f"- {t}" for t in tips)
    )


# ── Entity extraction ─────────────────────────────────────────────────────────
def run_entity_extraction(text):
    if not text.strip():
        return "Please enter some climate text to analyse."
    return _format_entities(extract_entities(text))


def run_summarisation(text, style):
    if not text.strip():
        return "Please enter text to summarise."
    result = summarise_document(text, max_summary_words=200, style=style.lower())
    return result["summary"]


def get_kb_stats():
    stats = get_vectorstore_stats()
    return (
        f"📚 **Knowledge Base Status**\n\n"
        f"- Collection: `{stats['collection']}`\n"
        f"- Document chunks: **{stats['document_count']}**\n"
        f"- Sources: IPCC AR6, NOAA, EPA, NASA GISS, IEA, IUCN\n"
        f"- Embedding model: all-MiniLM-L6-v2 / IBM Granite Embedding"
    )


# ── Build UI ──────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ClimateBot — AI Climate Assistant") as app:

        gr.Markdown(
            """
# 🌍 ClimateBot — AI Climate Awareness Assistant
**Powered by IBM Granite · RAG · Agentic AI · Multimodal Inputs**

Ask anything about climate science, upload documents for analysis, or use the tools below.
            """
        )

        with gr.Tabs():

            # ── TAB 1: Chat ───────────────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # ✅ FIX: removed show_copy_button (Gradio 6 removed it)
                        chatbot = gr.Chatbot(
                            label="ClimateBot",
                            height=480,
                            render_markdown=True,
                            elem_classes=["chatbot-container"],
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder=(
                                    "Ask me about climate change, sea level rise, "
                                    "CO₂ levels, renewable energy, the Paris Agreement..."
                                ),
                                label="Your question",
                                scale=5,
                                lines=2,
                            )
                        with gr.Row():
                            file_input = gr.File(
                                label="📎 Upload PDF / Image / CSV (optional)",
                                file_types=[".pdf", ".png", ".jpg", ".jpeg",
                                            ".csv", ".txt", ".webp"],
                                scale=3,
                            )
                            with gr.Column(scale=2):
                                send_btn = gr.Button("🌿 Send", variant="primary", size="lg")
                                clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

                        gr.Markdown("**💡 Suggested questions:**")
                        with gr.Row():
                            q1 = gr.Button("What is the current CO₂ level?", size="sm")
                            q2 = gr.Button("Explain the 1.5°C warming target", size="sm")
                            q3 = gr.Button("How fast are sea levels rising?", size="sm")
                        with gr.Row():
                            q4 = gr.Button("What are the biggest emission sources?", size="sm")
                            q5 = gr.Button("Tell me about the Paris Agreement", size="sm")
                            q6 = gr.Button("What solutions exist for climate change?", size="sm")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📡 Live Climate Data")
                        co2_display = gr.Markdown("*Loading...*")
                        temp_display = gr.Markdown("*Loading...*")
                        data_status = gr.Markdown("")
                        refresh_btn = gr.Button("🔄 Refresh Live Data", size="sm")
                        gr.Markdown("---")
                        kb_stats = gr.Markdown(get_kb_stats())

            # ── TAB 2: Carbon Footprint ───────────────────────────────────────
            with gr.TabItem("🌱 Carbon Footprint"):
                gr.Markdown(
                    "## Personal Carbon Footprint Calculator\n"
                    "Estimate your annual CO₂ emissions and get personalised reduction tips."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🚗 Transport")
                        fp_car   = gr.Number(label="Car distance driven (km/year)", value=10000)
                        fp_short = gr.Number(label="Short-haul flights (<3h) per year", value=2)
                        fp_long  = gr.Number(label="Long-haul flights (>6h) per year", value=1)
                        gr.Markdown("### ⚡ Home Energy")
                        fp_elec  = gr.Number(label="Electricity use (kWh/month)", value=250)
                        fp_gas   = gr.Number(label="Natural gas use (kWh/month)", value=100)
                    with gr.Column():
                        gr.Markdown("### 🥗 Diet & Lifestyle")
                        fp_diet  = gr.Dropdown(
                            label="Diet type",
                            choices=["Meat heavy", "Average", "Vegetarian", "Vegan"],
                            value="Average",
                        )
                        fp_spend = gr.Number(
                            label="Monthly spend on goods & services (USD)", value=300
                        )
                        gr.Markdown("&nbsp;")
                        calc_btn = gr.Button("🧮 Calculate My Footprint", variant="primary")

                fp_output = gr.Markdown(label="Your Results")

            # ── TAB 3: Document Analysis ──────────────────────────────────────
            with gr.TabItem("🔬 Document Analysis"):
                gr.Markdown(
                    "## Entity Extraction & Summarisation\n"
                    "Paste climate text to extract named entities and generate summaries."
                )
                with gr.Row():
                    with gr.Column():
                        analysis_text = gr.Textbox(
                            label="Climate text to analyse",
                            placeholder=(
                                "Paste a paragraph or section from a climate report, "
                                "news article, or scientific abstract..."
                            ),
                            lines=10,
                        )
                        with gr.Row():
                            extract_btn    = gr.Button("🏷️ Extract Entities", variant="primary")
                            summarise_style = gr.Dropdown(
                                label="Summary style",
                                choices=["Factual", "Accessible", "Bullet"],
                                value="Factual",
                            )
                            summarise_btn  = gr.Button("📝 Summarise", variant="secondary")
                    with gr.Column():
                        entity_output  = gr.Markdown(label="Extracted Entities")
                        summary_output = gr.Markdown(label="Summary")

                gr.Markdown("**Example texts to try:**")
                with gr.Row():
                    ex1 = gr.Button("IPCC 1.5°C finding", size="sm")
                    ex2 = gr.Button("Renewable energy stats", size="sm")
                    ex3 = gr.Button("Paris Agreement text", size="sm")

            # ── TAB 4: About ──────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
## ClimateBot Architecture

| Component | Technology | Role |
|---|---|---|
| **LLM** | IBM Granite 3.3-8B-Instruct (Watsonx.ai) | Conversational AI backbone |
| **Fallback LLM** | Anthropic Claude Sonnet | Development & testing |
| **Embeddings** | IBM Granite Embedding / all-MiniLM-L6-v2 | Document vectorisation |
| **Vector Store** | ChromaDB (persistent) | Knowledge retrieval |
| **RAG** | LangChain RetrievalQA | Grounded answer generation |
| **Agent** | Custom ReAct orchestrator | Tool use & intent routing |
| **PDF Parsing** | PyMuPDF (fitz) | Document ingestion |
| **Image Analysis** | Claude Vision / filename heuristics | Multimodal understanding |
| **UI** | Gradio 6 | Web interface |
| **API** | FastAPI + Uvicorn | REST backend |

### Knowledge Sources
- **IPCC AR6** — Sixth Assessment Report (2021–2022)
- **NOAA** — Global Monitoring Laboratory, CO₂ data
- **NASA GISS** — Surface Temperature Analysis
- **EPA** — US Greenhouse Gas Inventory
- **IEA** — World Energy Outlook
- **UNFCCC** — Paris Agreement, NDCs

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add IBM_API_KEY or ANTHROPIC_API_KEY
python main.py         # http://localhost:7860
```
                    """
                )

        # ── Event wiring ───────────────────────────────────────────────────────
        send_btn.click(fn=chat, inputs=[msg_input, chatbot, file_input],
                       outputs=[chatbot, msg_input])
        msg_input.submit(fn=chat, inputs=[msg_input, chatbot, file_input],
                         outputs=[chatbot, msg_input])
        clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])
        refresh_btn.click(fn=refresh_live_data,
                          outputs=[co2_display, temp_display, data_status])
        calc_btn.click(fn=run_footprint_calc,
                       inputs=[fp_car, fp_short, fp_long, fp_elec, fp_gas, fp_diet, fp_spend],
                       outputs=fp_output)
        extract_btn.click(fn=run_entity_extraction, inputs=analysis_text, outputs=entity_output)
        summarise_btn.click(fn=run_summarisation, inputs=[analysis_text, summarise_style],
                            outputs=summary_output)

        # Suggested question quick-fills
        q1.click(fn=lambda: "What is the current atmospheric CO₂ concentration?", outputs=msg_input)
        q2.click(fn=lambda: "Explain the 1.5°C warming target from the Paris Agreement.", outputs=msg_input)
        q3.click(fn=lambda: "How fast are global sea levels rising and what are the projections?", outputs=msg_input)
        q4.click(fn=lambda: "What are the biggest sources of greenhouse gas emissions globally?", outputs=msg_input)
        q5.click(fn=lambda: "What is the Paris Agreement and are countries on track to meet its goals?", outputs=msg_input)
        q6.click(fn=lambda: "What are the most effective solutions available for climate change?", outputs=msg_input)

        # Example texts
        IPCC_EXAMPLE = (
            "The Sixth Assessment Report of the IPCC (2021) states that global surface "
            "temperature has increased by 1.1°C above 1850-1900 levels. Human influence has "
            "warmed the climate at an unprecedented rate. CO2 concentrations have increased by "
            "47% since pre-industrial times. The Arctic is warming at more than twice the global "
            "rate. Sea levels have risen 20 cm since 1900 and the rate is accelerating. "
            "Extreme heat events are now 5 times more likely than in the pre-industrial period."
        )
        RENEWABLE_EXAMPLE = (
            "In 2023, the International Energy Agency (IEA) reported that renewable energy "
            "capacity additions reached a record 295 GW of solar PV globally. The cost of "
            "solar electricity has fallen 89% since 2010, making it the cheapest source of "
            "electricity in history in most countries. Wind power added 116 GW. Together, "
            "renewables accounted for 90% of all new electricity capacity. The United States, "
            "China, and the European Union led in new installations."
        )
        PARIS_EXAMPLE = (
            "The Paris Agreement, adopted at COP21 in December 2015, is a legally binding "
            "international treaty on climate change. It was ratified by 195 parties. The "
            "agreement aims to limit global warming to well below 2°C, preferably 1.5°C, "
            "above pre-industrial levels. Countries submit Nationally Determined Contributions "
            "(NDCs) every five years. Current NDCs are projected to result in approximately "
            "2.5-2.9°C of warming by 2100 according to Climate Action Tracker."
        )
        ex1.click(fn=lambda: IPCC_EXAMPLE, outputs=analysis_text)
        ex2.click(fn=lambda: RENEWABLE_EXAMPLE, outputs=analysis_text)
        ex3.click(fn=lambda: PARIS_EXAMPLE, outputs=analysis_text)

        # Auto-load live data on startup
        app.load(fn=refresh_live_data, outputs=[co2_display, temp_display, data_status])

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
