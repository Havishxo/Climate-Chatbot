"""
tools.py — Agentic AI tools for the Climate Awareness Chatbot.

Each function is a callable tool the agent can invoke:
  - get_live_co2()           : current atmospheric CO2 ppm from global-warming.org
  - get_temperature_anomaly(): latest global temperature anomaly
  - calculate_carbon_footprint(): personal carbon footprint calculator
  - search_knowledge_base()  : wrapper around RAG retrieval
  - summarise_text()         : summarise long content
"""

import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ── Live CO2 data ──────────────────────────────────────────────────────────────
def get_live_co2() -> Dict[str, Any]:
    """
    Fetch current atmospheric CO2 concentration from global-warming.org API.
    Returns: {"ppm": float, "year": str, "month": str, "trend": float}
    """
    try:
        resp = requests.get(
            "https://global-warming.org/api/co2-api",
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        # API returns a list of {"year", "month", "day", "cycle", "trend"}
        if "co2" in data and data["co2"]:
            latest = data["co2"][-1]
            return {
                "ppm": float(latest.get("cycle", latest.get("trend", 0))),
                "trend_ppm": float(latest.get("trend", 0)),
                "year": latest.get("year", "N/A"),
                "month": latest.get("month", "N/A"),
                "source": "NOAA / global-warming.org",
                "status": "live",
            }
    except Exception as exc:
        logger.warning("CO2 API request failed: %s", exc)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("CO2 API parse error: %s", exc)

    # Fallback
    return {
        "ppm": 422.5,
        "trend_ppm": 421.8,
        "year": "2024",
        "month": "06",
        "source": "NOAA (cached fallback)",
        "status": "cached",
    }


# ── Live temperature anomaly ───────────────────────────────────────────────────
def get_temperature_anomaly() -> Dict[str, Any]:
    """
    Fetch global surface temperature anomaly from global-warming.org.
    Returns: {"anomaly_c": float, "year": str, "source": str}
    """
    try:
        resp = requests.get(
            "https://global-warming.org/api/temperature-api",
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if "result" in data and data["result"]:
            result = data["result"]
            # API may return either dict {year:value} or list of dicts
            if isinstance(result, dict):
                latest_year, latest_val = list(result.items())[-1]
            elif isinstance(result, list):
                last = result[-1]
                if isinstance(last, dict):
                    latest_year = str(last.get("time") or last.get("year") or "2023")
                    latest_val = float(last.get("station") or last.get("land") or last.get("value") or 1.17)
                else:
                    latest_year, latest_val = "2023", 1.17
            else:
                latest_year, latest_val = "2023", 1.17
            return {
                "anomaly_c": float(latest_val),
                "year": latest_year,
                "baseline": "1951-1980",
                "source": "NASA GISS / global-warming.org",
                "status": "live",
            }
    except Exception as exc:
        logger.warning("Temperature API error: %s", exc)

    return {
        "anomaly_c": 1.17,
        "year": "2023",
        "baseline": "1951-1980",
        "source": "NASA GISS (cached fallback)",
        "status": "cached",
    }


# ── Carbon footprint calculator ────────────────────────────────────────────────
# Emission factors (kgCO2e per unit) — from IPCC / EPA standards
_EMISSION_FACTORS = {
    # Transport
    "car_petrol_km": 0.192,      # per km, average petrol car
    "car_electric_km": 0.053,    # per km, EV (global grid average)
    "flight_short_km": 0.255,    # per km, short haul (<3h)
    "flight_long_km": 0.195,     # per km, long haul (>6h)
    "bus_km": 0.089,
    "train_km": 0.041,
    # Energy
    "electricity_kwh": 0.233,    # global average grid
    "natural_gas_kwh": 0.202,
    # Diet (per year, kgCO2e)
    "diet_meat_heavy": 3300,
    "diet_average": 2500,
    "diet_vegetarian": 1700,
    "diet_vegan": 1500,
    # Goods & services (monthly spend proxy)
    "goods_monthly_spend_usd": 0.5,  # kgCO2e per $1 spent
}

def calculate_carbon_footprint(
    car_km_year: float = 0,
    flights_short: int = 0,
    flights_long: int = 0,
    electricity_kwh_month: float = 0,
    natural_gas_kwh_month: float = 0,
    diet_type: str = "average",    # meat_heavy | average | vegetarian | vegan
    goods_spend_monthly_usd: float = 200,
) -> Dict[str, Any]:
    """
    Estimate annual personal carbon footprint in tonnes CO2-equivalent.
    """
    f = _EMISSION_FACTORS

    transport = (
        car_km_year * f["car_petrol_km"]
        + flights_short * 2000 * f["flight_short_km"]   # avg 2000 km short haul
        + flights_long * 9000 * f["flight_long_km"]     # avg 9000 km long haul
    )
    energy = (electricity_kwh_month * 12 * f["electricity_kwh"]
              + natural_gas_kwh_month * 12 * f["natural_gas_kwh"])
    diet = f.get(f"diet_{diet_type.replace('-','_')}", f["diet_average"])
    goods = goods_spend_monthly_usd * 12 * f["goods_monthly_spend_usd"]

    total_kg = transport + energy + diet + goods
    total_tonnes = total_kg / 1000

    # Global average: ~4.7 tCO2e; 1.5°C budget target: ~2.3 tCO2e/person/year
    comparison = {
        "vs_global_avg": round(total_tonnes / 4.7, 2),
        "vs_1_5c_budget": round(total_tonnes / 2.3, 2),
    }

    return {
        "total_tonnes_co2e": round(total_tonnes, 2),
        "breakdown": {
            "transport_kg": round(transport, 1),
            "energy_kg": round(energy, 1),
            "diet_kg": round(diet, 1),
            "goods_kg": round(goods, 1),
        },
        "comparison": comparison,
        "tips": _get_reduction_tips(transport, energy, diet, goods),
    }


def _get_reduction_tips(transport, energy, diet, goods) -> list:
    tips = []
    if transport > 2000:
        tips.append("🚗 Consider switching to an EV or reducing driving — transport is your biggest footprint.")
    if energy > 1000:
        tips.append("⚡ Switch to renewable electricity tariff to cut energy emissions by up to 70%.")
    if diet > 2000:
        tips.append("🥗 Reducing red meat consumption is one of the highest-impact individual actions.")
    if goods > 1500:
        tips.append("🛍️ Buying second-hand and repairing items significantly cuts goods emissions.")
    if not tips:
        tips.append("✅ Your footprint is below the global average — well done!")
    return tips


# ── Knowledge base search tool ─────────────────────────────────────────────────
def search_knowledge_base(query: str, k: int = 3) -> Dict[str, Any]:
    """RAG retrieval exposed as an agent tool."""
    from app.rag import retrieve_with_scores
    docs_scores = retrieve_with_scores(query, k=k)
    results = []
    for doc, score in docs_scores:
        results.append({
            "content": doc.page_content[:400],
            "source": doc.metadata.get("source", "Unknown"),
            "topic": doc.metadata.get("topic", "general"),
            "relevance_score": round(float(score), 4),
        })
    return {"query": query, "results": results, "count": len(results)}


# ── Text summarisation tool ────────────────────────────────────────────────────
def summarise_text(text: str, max_words: int = 150) -> Dict[str, Any]:
    """Summarise a block of text using the configured LLM."""
    from app.llm_provider import generate
    prompt = (
        f"Summarise the following climate-related text in approximately {max_words} words. "
        f"Focus on key facts, numbers, and conclusions. Be concise.\n\nText:\n{text[:3000]}"
    )
    summary = generate(prompt)
    return {"summary": summary, "original_length": len(text.split())}


# ── Agent tool registry ────────────────────────────────────────────────────────
TOOL_REGISTRY = {
    "get_live_co2": {
        "fn": get_live_co2,
        "description": "Fetch real-time atmospheric CO2 concentration in parts per million (ppm) from NOAA.",
        "args": [],
    },
    "get_temperature_anomaly": {
        "fn": get_temperature_anomaly,
        "description": "Fetch the latest global surface temperature anomaly (°C above baseline) from NASA GISS.",
        "args": [],
    },
    "calculate_carbon_footprint": {
        "fn": calculate_carbon_footprint,
        "description": "Calculate annual personal carbon footprint in tonnes CO2e from lifestyle inputs.",
        "args": ["car_km_year", "flights_short", "flights_long",
                 "electricity_kwh_month", "natural_gas_kwh_month", "diet_type"],
    },
    "search_knowledge_base": {
        "fn": search_knowledge_base,
        "description": "Search the IPCC/NOAA knowledge base for relevant climate science information.",
        "args": ["query"],
    },
    "summarise_text": {
        "fn": summarise_text,
        "description": "Summarise a long piece of text about climate topics.",
        "args": ["text"],
    },
}


def describe_tools() -> str:
    """Return a text description of all available tools for the agent prompt."""
    lines = ["Available tools:\n"]
    for name, info in TOOL_REGISTRY.items():
        args = ", ".join(info["args"]) if info["args"] else "none"
        lines.append(f"  • {name}(args: {args})\n    {info['description']}")
    return "\n".join(lines)


def call_tool(name: str, **kwargs) -> Any:
    """Invoke a registered tool by name."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name]["fn"](**kwargs)
    except Exception as exc:
        logger.error("Tool '%s' error: %s", name, exc)
        return {"error": str(exc)}
