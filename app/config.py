"""
config.py — Centralised configuration for the Climate Awareness Chatbot.
All settings are read from environment variables (or .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chroma_db"))

# ── IBM Watsonx ────────────────────────────────────────────────────────────────
IBM_API_KEY = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")
IBM_URL = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
GRANITE_CHAT_MODEL = os.getenv("GRANITE_CHAT_MODEL", "ibm/granite-3-3-8b-instruct")
GRANITE_EMBEDDING_MODEL = os.getenv("GRANITE_EMBEDDING_MODEL", "ibm/granite-embedding-107m-multilingual")

# ── Anthropic (testing / fallback LLM) ────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── External APIs ──────────────────────────────────────────────────────────────
NOAA_API_TOKEN = os.getenv("NOAA_API_TOKEN", "")
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
CO2_API_URL = "https://global-warming.org/api/co2-api"   # free, no key
TEMP_ANOMALY_URL = "https://global-warming.org/api/temperature-api"

# ── RAG Settings ───────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"   # local fallback embedding model
COLLECTION_NAME = "climate_knowledge"

# ── Generation Parameters ──────────────────────────────────────────────────────
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.2
TOP_P = 0.9

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Derived Flags ──────────────────────────────────────────────────────────────
USE_IBM = bool(IBM_API_KEY and IBM_PROJECT_ID)
USE_ANTHROPIC = bool(ANTHROPIC_API_KEY)
