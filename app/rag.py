"""
rag.py — Retrieval-Augmented Generation pipeline.
Compatible with LangChain 1.3.x
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.config import (
    CHROMA_DB_PATH, KNOWLEDGE_BASE_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_RETRIEVED_CHUNKS,
)
from app.embeddings import get_embeddings
from app.llm_provider import get_llm

logger = logging.getLogger(__name__)

# ── Climate-domain system prompt ───────────────────────────────────────────────
CLIMATE_SYSTEM_PROMPT = """You are ClimateBot, an expert AI assistant specialising in climate \
science, environmental policy, and sustainability.

Your responses are grounded in the CONTEXT provided below, drawn from authoritative sources \
such as IPCC Assessment Reports, NOAA datasets, EPA publications, and peer-reviewed research.

Rules you must always follow:
1. If the answer is in the CONTEXT, cite the source (e.g. "[Source: IPCC AR6, 2021]").
2. If the CONTEXT does not contain enough information, say so clearly — never fabricate facts.
3. Explain scientific terms in plain language after using them.
4. Use precise numbers and units when available (°C, ppm, mm/year, GtCO₂).
5. Be concise, factual, and accessible to a general audience.

--- CONTEXT ---
{context}
---------------

Human question: {question}

Think step by step, then provide a clear, well-cited answer:"""

CLIMATE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=CLIMATE_SYSTEM_PROMPT,
)

# ── Seed knowledge ─────────────────────────────────────────────────────────────
SEED_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "text": (
            "The Sixth Assessment Report (AR6) by the IPCC (2021-2022) states that global "
            "surface temperature has increased by approximately 1.1°C above 1850-1900 levels "
            "by 2011-2020. Human influence has warmed the atmosphere, ocean, and land at an "
            "unprecedented rate in at least the last 2000 years."
        ),
        "source": "IPCC AR6 WGI, Summary for Policymakers, 2021",
        "topic": "global warming",
    },
    {
        "text": (
            "Atmospheric CO2 concentration reached 421.08 ppm in April 2024 according to NOAA's "
            "Mauna Loa Observatory. This is 50% higher than pre-industrial levels of ~280 ppm. "
            "Methane (CH4) and nitrous oxide (N2O) have also increased by 260% and 123% "
            "respectively since 1750."
        ),
        "source": "NOAA Global Monitoring Laboratory, 2024",
        "topic": "greenhouse gases",
    },
    {
        "text": (
            "Global mean sea level has risen about 20 cm since 1900. The rate of rise has "
            "accelerated from 1.3 mm/year during 1901-1971 to 3.7 mm/year during 2006-2018. "
            "Under a high-emissions scenario (SSP5-8.5), sea levels could rise 0.63-1.01 m "
            "by 2100 relative to 1995-2014."
        ),
        "source": "IPCC AR6 WGI, Chapter 9, 2021",
        "topic": "sea level rise",
    },
    {
        "text": (
            "Arctic sea ice extent has declined by about 13% per decade since 1979, based on "
            "satellite records. The Arctic is warming at more than twice the global average rate "
            "(Arctic amplification). September 2023 saw the lowest Antarctic sea ice extent on "
            "record at 1.79 million km² below the 1981-2010 average."
        ),
        "source": "NSIDC / NOAA Arctic Report Card, 2023",
        "topic": "sea ice",
    },
    {
        "text": (
            "Global greenhouse gas emissions reached approximately 59 GtCO2-equivalent in 2019. "
            "CO2 from fossil fuels and industry accounts for 64% of total emissions. "
            "To limit warming to 1.5°C, global net CO2 emissions must reach net-zero around 2050, "
            "with deep cuts of 45% by 2030 relative to 2010 levels."
        ),
        "source": "IPCC AR6 WGIII, Summary for Policymakers, 2022",
        "topic": "emissions",
    },
    {
        "text": (
            "Extreme heat events that occurred once in 50 years in the pre-industrial climate will "
            "occur 8.6 times per 50 years at 1.5°C warming and 39.2 times at 4°C warming. "
            "Heavy precipitation events have increased in frequency and intensity. "
            "The probability of a Category 4-5 tropical cyclone has increased with warming."
        ),
        "source": "IPCC AR6 WGI, Chapter 11, 2021",
        "topic": "extreme weather",
    },
    {
        "text": (
            "The Paris Agreement (2015) aims to limit global warming to well below 2°C, "
            "preferably 1.5°C, above pre-industrial levels. As of 2023, 195 parties have "
            "ratified the agreement. Current Nationally Determined Contributions (NDCs) are "
            "projected to result in warming of approximately 2.5-2.9°C by 2100."
        ),
        "source": "UNFCCC / Climate Action Tracker, 2023",
        "topic": "policy",
    },
    {
        "text": (
            "Renewable energy capacity additions set a new record in 2023 with 295 GW of solar "
            "PV installed globally (IEA). Wind added 116 GW. The cost of solar PV electricity "
            "has fallen 89% since 2010. Renewables accounted for 90% of all new electricity "
            "capacity additions in 2023."
        ),
        "source": "IEA World Energy Outlook, 2023",
        "topic": "renewable energy",
    },
    {
        "text": (
            "Nature-based solutions such as restoring forests, wetlands, and soils could provide "
            "up to one-third of the climate mitigation needed by 2030 to keep warming below 2°C. "
            "Deforestation accounts for about 11% of global GHG emissions. "
            "Tropical forests store approximately 250 billion tonnes of carbon."
        ),
        "source": "Nature-based Solutions for Climate — IUCN, 2022",
        "topic": "nature-based solutions",
    },
    {
        "text": (
            "Carbon capture and storage (CCS) technology can capture up to 90% of CO2 emitted "
            "from industrial processes. As of 2023, there are 41 commercial CCS facilities "
            "globally with a capture capacity of about 49 MtCO2/year. Direct air capture (DAC) "
            "currently costs $300-1000 per tonne of CO2 but costs are projected to fall."
        ),
        "source": "Global CCS Institute Status Report, 2023",
        "topic": "carbon capture",
    },
]

# ── Vector store management ────────────────────────────────────────────────────
_vectorstore: Optional[Chroma] = None


def _get_or_create_vectorstore(reset: bool = False) -> Chroma:
    global _vectorstore
    if _vectorstore is not None and not reset:
        return _vectorstore

    embeddings = get_embeddings()
    persist_dir = CHROMA_DB_PATH

    try:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        count = _vectorstore._collection.count()
        logger.info("Vector store loaded. Documents: %d", count)
        if count == 0:
            _seed_knowledge_base()
        return _vectorstore
    except Exception as exc:
        logger.error("Chroma init error: %s — using in-memory store.", exc)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        _seed_knowledge_base()
        return _vectorstore


def _seed_knowledge_base():
    global _vectorstore
    if _vectorstore is None:
        return
    docs = [
        Document(
            page_content=item["text"],
            metadata={"source": item["source"], "topic": item["topic"]},
        )
        for item in SEED_KNOWLEDGE
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    _vectorstore.add_documents(chunks)
    logger.info("Seeded knowledge base with %d chunks.", len(chunks))


# ── Document ingestion ─────────────────────────────────────────────────────────
def ingest_pdf(file_path: str) -> int:
    try:
        import fitz
        doc = fitz.open(file_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": os.path.basename(file_path),
                               "page": page_num + 1, "topic": "user-uploaded"},
                ))
        doc.close()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)
        vs = _get_or_create_vectorstore()
        vs.add_documents(chunks)
        logger.info("Ingested PDF '%s': %d chunks.", file_path, len(chunks))
        return len(chunks)
    except Exception as exc:
        logger.error("PDF ingestion error: %s", exc)
        return 0


def ingest_text(text: str, source: str = "user-input", topic: str = "general") -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = [Document(page_content=text, metadata={"source": source, "topic": topic})]
    chunks = splitter.split_documents(docs)
    vs = _get_or_create_vectorstore()
    vs.add_documents(chunks)
    logger.info("Ingested text '%s': %d chunks.", source, len(chunks))
    return len(chunks)


def ingest_csv(file_path: str) -> int:
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return ingest_text(text, source=os.path.basename(file_path), topic="dataset")
    except Exception as exc:
        logger.error("CSV ingestion error: %s", exc)
        return 0


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = MAX_RETRIEVED_CHUNKS) -> List[Document]:
    vs = _get_or_create_vectorstore()
    try:
        return vs.similarity_search(query, k=k)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        return []


def retrieve_with_scores(query: str, k: int = MAX_RETRIEVED_CHUNKS):
    vs = _get_or_create_vectorstore()
    try:
        return vs.similarity_search_with_score(query, k=k)
    except Exception:
        return [(doc, 0.0) for doc in retrieve(query, k)]


# ── RAG query ──────────────────────────────────────────────────────────────────
def build_rag_chain():
    vs = _get_or_create_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": MAX_RETRIEVED_CHUNKS})
    llm = get_llm()
    from app.llm_provider import MockLLM
    if isinstance(llm, MockLLM):
        return None
    try:
        from langchain.chains import RetrievalQA  # type: ignore
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": CLIMATE_PROMPT},
        )
        return chain
    except Exception as exc:
        logger.warning("Could not build RAG chain: %s", exc)
        return None


def rag_query(query: str) -> Dict[str, Any]:
    docs_with_scores = retrieve_with_scores(query)
    context_parts = []
    sources = []
    for doc, score in docs_with_scores:
        excerpt = doc.page_content[:300].strip()
        src = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[{src}]\n{excerpt}")
        sources.append({"source": src, "excerpt": excerpt, "score": float(score)})

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    prompt_text = CLIMATE_SYSTEM_PROMPT.replace("{context}", context).replace("{question}", query)

    from app.llm_provider import generate
    answer = generate(prompt_text)
    return {
        "answer": answer,
        "sources": sources,
        "context_used": bool(context_parts),
    }


def get_vectorstore_stats() -> Dict[str, Any]:
    try:
        vs = _get_or_create_vectorstore()
        count = vs._collection.count()
        return {"document_count": count, "collection": COLLECTION_NAME}
    except Exception:
        return {"document_count": 0, "collection": COLLECTION_NAME}
