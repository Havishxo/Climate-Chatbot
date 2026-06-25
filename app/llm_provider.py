"""
llm_provider.py — LLM abstraction layer.

Priority order:
  1. IBM Granite via Watsonx.ai  (if IBM_API_KEY + IBM_PROJECT_ID set)
  2. Anthropic Claude             (if ANTHROPIC_API_KEY set)
  3. Mock LLM                     (demo / offline mode)

Exposes a single get_llm() function and a generate() helper used across the app.
"""

import logging
from typing import Optional
from app.config import (
    IBM_API_KEY, IBM_PROJECT_ID, IBM_URL, GRANITE_CHAT_MODEL,
    ANTHROPIC_API_KEY, MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
    USE_IBM, USE_ANTHROPIC,
)

logger = logging.getLogger(__name__)


# ── IBM Granite via Watsonx.ai ─────────────────────────────────────────────────
def _get_ibm_llm():
    """Return a WatsonxLLM instance backed by IBM Granite."""
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
        from langchain_ibm import WatsonxLLM

        params = {
            Params.MAX_NEW_TOKENS: MAX_NEW_TOKENS,
            Params.TEMPERATURE: TEMPERATURE,
            Params.TOP_P: TOP_P,
            Params.DECODING_METHOD: "greedy",
            Params.STOP_SEQUENCES: ["\n\nHuman:", "\n\nUser:"],
        }
        llm = WatsonxLLM(
            model_id=GRANITE_CHAT_MODEL,
            url=IBM_URL,
            apikey=IBM_API_KEY,
            project_id=IBM_PROJECT_ID,
            params=params,
        )
        logger.info("✅ IBM Granite LLM loaded: %s", GRANITE_CHAT_MODEL)
        return llm
    except ImportError:
        logger.warning("ibm-watsonx-ai or langchain-ibm not installed — skipping IBM.")
        return None
    except Exception as exc:
        logger.warning("IBM Granite init failed: %s", exc)
        return None


# ── Anthropic Claude fallback ──────────────────────────────────────────────────
def _get_anthropic_llm():
    """Return a ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=ANTHROPIC_API_KEY,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        logger.info("✅ Anthropic Claude LLM loaded (fallback).")
        return llm
    except ImportError:
        logger.warning("langchain-anthropic not installed — skipping Anthropic.")
        return None
    except Exception as exc:
        logger.warning("Anthropic init failed: %s", exc)
        return None


# ── Mock LLM (demo / offline) ──────────────────────────────────────────────────
class MockLLM:
    """Minimal mock that returns a canned climate response for offline demos."""

    def invoke(self, prompt: str) -> str:
        return self._reply(str(prompt))

    def __call__(self, prompt: str) -> str:
        return self._reply(str(prompt))

    @staticmethod
    def _reply(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "co2" in prompt_lower or "carbon dioxide" in prompt_lower:
            return (
                "🌍 Current atmospheric CO₂ concentration is approximately 422 ppm "
                "(as of 2024), well above the pre-industrial level of 280 ppm. "
                "This increase is primarily driven by fossil fuel combustion and deforestation. "
                "[Source: NOAA Global Monitoring Laboratory]"
            )
        if "temperature" in prompt_lower or "warming" in prompt_lower:
            return (
                "🌡️ Global average surface temperature has risen approximately 1.1 °C "
                "above pre-industrial levels (1850–1900 baseline). The IPCC AR6 report "
                "projects further warming of 1.5–4 °C by 2100 depending on emission pathways. "
                "[Source: IPCC Sixth Assessment Report, 2021]"
            )
        if "sea level" in prompt_lower:
            return (
                "🌊 Global mean sea level has risen about 20 cm since 1900, with the "
                "rate accelerating to 3.6 mm/year over 2006–2015. By 2100, projections "
                "range from 0.3 m to over 1 m depending on greenhouse gas emissions. "
                "[Source: IPCC AR6 WGI, Chapter 9]"
            )
        return (
            "🌱 I'm ClimateBot — your AI guide to climate science. I use IBM Granite "
            "models and retrieval-augmented generation (RAG) to answer questions about "
            "climate change, emissions, sea level rise, and more. "
            "Ask me anything about climate science! "
            "(Running in demo mode — connect IBM Watsonx or Anthropic API for full answers.)"
        )


_llm_instance = None


def get_llm():
    """Return the best available LLM (cached singleton)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if USE_IBM:
        _llm_instance = _get_ibm_llm()
    if _llm_instance is None and USE_ANTHROPIC:
        _llm_instance = _get_anthropic_llm()
    if _llm_instance is None:
        logger.warning("⚠️  No API keys found — using MockLLM (demo mode).")
        _llm_instance = MockLLM()

    return _llm_instance


def generate(prompt: str, llm=None) -> str:
    """Generate a response from the LLM. Handles both Chat and Completion models."""
    if llm is None:
        llm = get_llm()
    try:
        result = llm.invoke(prompt)
        # ChatAnthropic returns an AIMessage object
        if hasattr(result, "content"):
            return result.content
        return str(result)
    except Exception as exc:
        logger.error("LLM generation error: %s", exc)
        return f"⚠️ Generation error: {exc}"
