"""
tests/test_all.py — Full test suite for ClimateBot.

Run with:  python -m pytest tests/ -v
Or:        python tests/test_all.py
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════
#  1. Config tests
# ═══════════════════════════════════════════════════════════════
class TestConfig(unittest.TestCase):

    def test_config_imports(self):
        from app.config import (
            CHUNK_SIZE, CHUNK_OVERLAP, MAX_RETRIEVED_CHUNKS,
            EMBEDDING_MODEL_LOCAL, COLLECTION_NAME,
        )
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreater(CHUNK_OVERLAP, 0)
        self.assertGreater(MAX_RETRIEVED_CHUNKS, 0)
        self.assertIsInstance(EMBEDDING_MODEL_LOCAL, str)
        self.assertIsInstance(COLLECTION_NAME, str)

    def test_paths_are_strings(self):
        from app.config import CHROMA_DB_PATH, KNOWLEDGE_BASE_DIR
        self.assertIsInstance(str(CHROMA_DB_PATH), str)


# ═══════════════════════════════════════════════════════════════
#  2. LLM Provider tests
# ═══════════════════════════════════════════════════════════════
class TestLLMProvider(unittest.TestCase):

    def test_mock_llm_co2_query(self):
        from app.llm_provider import MockLLM
        llm = MockLLM()
        resp = llm.invoke("What is the current CO2 level?")
        self.assertIn("CO₂", resp)
        self.assertIn("ppm", resp)

    def test_mock_llm_temperature_query(self):
        from app.llm_provider import MockLLM
        llm = MockLLM()
        resp = llm.invoke("How much has global temperature increased?")
        self.assertIn("°C", resp)

    def test_mock_llm_sea_level_query(self):
        from app.llm_provider import MockLLM
        llm = MockLLM()
        resp = llm.invoke("Tell me about sea level rise")
        self.assertIn("sea", resp.lower())

    def test_mock_llm_generic_query(self):
        from app.llm_provider import MockLLM
        llm = MockLLM()
        resp = llm.invoke("Hello")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 10)

    def test_get_llm_returns_something(self):
        from app.llm_provider import get_llm
        llm = get_llm()
        self.assertIsNotNone(llm)

    def test_generate_function(self):
        from app.llm_provider import generate, MockLLM
        result = generate("Tell me about climate change", llm=MockLLM())
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)


# ═══════════════════════════════════════════════════════════════
#  3. Embeddings tests
# ═══════════════════════════════════════════════════════════════
class TestEmbeddings(unittest.TestCase):

    def test_tfidf_embeddings_basic(self):
        from app.embeddings import TFIDFEmbeddings
        emb = TFIDFEmbeddings(dim=64)
        docs = ["Climate change and global warming", "Sea level rise and Arctic ice"]
        vecs = emb.embed_documents(docs)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 64)

    def test_tfidf_query_embedding(self):
        from app.embeddings import TFIDFEmbeddings
        emb = TFIDFEmbeddings(dim=64)
        emb.embed_documents(["warming temperature climate"])
        vec = emb.embed_query("global warming")
        self.assertEqual(len(vec), 64)

    def test_tfidf_normalized(self):
        from app.embeddings import TFIDFEmbeddings
        import math
        emb = TFIDFEmbeddings(dim=32)
        emb.embed_documents(["carbon dioxide emissions"])
        vec = emb.embed_query("carbon")
        norm = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_get_embeddings_not_none(self):
        from app.embeddings import get_embeddings
        emb = get_embeddings()
        self.assertIsNotNone(emb)


# ═══════════════════════════════════════════════════════════════
#  4. Tools tests
# ═══════════════════════════════════════════════════════════════
class TestTools(unittest.TestCase):

    def test_carbon_footprint_basic(self):
        from app.tools import calculate_carbon_footprint
        result = calculate_carbon_footprint(
            car_km_year=10000,
            flights_short=2,
            flights_long=1,
            electricity_kwh_month=250,
            natural_gas_kwh_month=100,
            diet_type="average",
            goods_spend_monthly_usd=300,
        )
        self.assertIn("total_tonnes_co2e", result)
        self.assertIn("breakdown", result)
        self.assertIn("tips", result)
        self.assertIn("comparison", result)
        self.assertGreater(result["total_tonnes_co2e"], 0)

    def test_carbon_footprint_vegan_lower(self):
        from app.tools import calculate_carbon_footprint
        meat = calculate_carbon_footprint(diet_type="meat_heavy")
        vegan = calculate_carbon_footprint(diet_type="vegan")
        self.assertLess(vegan["total_tonnes_co2e"], meat["total_tonnes_co2e"])

    def test_carbon_footprint_zero_inputs(self):
        from app.tools import calculate_carbon_footprint
        result = calculate_carbon_footprint(
            car_km_year=0, flights_short=0, flights_long=0,
            electricity_kwh_month=0, natural_gas_kwh_month=0,
            goods_spend_monthly_usd=0,
        )
        self.assertGreaterEqual(result["total_tonnes_co2e"], 0)

    @patch("requests.get")
    def test_get_live_co2_api_success(self, mock_get):
        from app.tools import get_live_co2
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "co2": [{"year": "2024", "month": "06", "cycle": "422.5", "trend": "421.8"}]
        }
        mock_get.return_value = mock_resp
        result = get_live_co2()
        self.assertIn("ppm", result)
        self.assertEqual(result["ppm"], 422.5)

    @patch("requests.get", side_effect=Exception("Network error"))
    def test_get_live_co2_fallback(self, mock_get):
        from app.tools import get_live_co2
        result = get_live_co2()
        self.assertIn("ppm", result)
        self.assertEqual(result["status"], "cached")

    @patch("requests.get")
    def test_get_temperature_anomaly_success(self, mock_get):
        from app.tools import get_temperature_anomaly
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"result": {"2023": "1.17", "2024": "1.22"}}
        mock_get.return_value = mock_resp
        result = get_temperature_anomaly()
        self.assertIn("anomaly_c", result)

    def test_describe_tools_returns_string(self):
        from app.tools import describe_tools
        desc = describe_tools()
        self.assertIsInstance(desc, str)
        self.assertIn("get_live_co2", desc)
        self.assertIn("calculate_carbon_footprint", desc)

    def test_call_tool_unknown(self):
        from app.tools import call_tool
        result = call_tool("nonexistent_tool")
        self.assertIn("error", result)

    def test_search_knowledge_base_returns_dict(self):
        from app.tools import search_knowledge_base
        result = search_knowledge_base("CO2 emissions", k=2)
        self.assertIn("results", result)
        self.assertIn("count", result)


# ═══════════════════════════════════════════════════════════════
#  5. Entity Extraction tests
# ═══════════════════════════════════════════════════════════════
class TestEntityExtraction(unittest.TestCase):

    SAMPLE_TEXT = (
        "The IPCC AR6 report (2021) states that global temperatures have risen 1.1°C "
        "above pre-industrial levels. CO2 concentration reached 421 ppm. "
        "The Arctic is warming faster. The Paris Agreement targets 1.5°C. "
        "NOAA and NASA track these changes. Methane (CH4) levels have also risen. "
        "Greenland ice loss accelerates every year since 2000."
    )

    def test_regex_fallback_extracts_gases(self):
        from app.entity_extraction import _regex_fallback
        result = _regex_fallback(self.SAMPLE_TEXT)
        self.assertIn("greenhouse_gases", result)
        gases = [g.upper() for g in result["greenhouse_gases"]]
        self.assertTrue(any("CO" in g or "CH" in g for g in gases))

    def test_regex_fallback_extracts_orgs(self):
        from app.entity_extraction import _regex_fallback
        result = _regex_fallback(self.SAMPLE_TEXT)
        orgs = result.get("organizations", [])
        self.assertTrue(any("IPCC" in o or "NOAA" in o or "NASA" in o for o in orgs))

    def test_regex_fallback_extracts_temperatures(self):
        from app.entity_extraction import _regex_fallback
        result = _regex_fallback(self.SAMPLE_TEXT)
        self.assertGreater(len(result.get("temperatures", [])), 0)

    def test_regex_fallback_extracts_years(self):
        from app.entity_extraction import _regex_fallback
        result = _regex_fallback(self.SAMPLE_TEXT)
        self.assertIn("2021", result.get("years_periods", []))

    def test_parse_json_valid(self):
        from app.entity_extraction import _parse_json_from_response
        valid_json = '{"gases": ["CO2"], "locations": ["Arctic"]}'
        result = _parse_json_from_response(valid_json)
        self.assertIsNotNone(result)
        self.assertEqual(result["gases"], ["CO2"])

    def test_parse_json_with_markdown_fences(self):
        from app.entity_extraction import _parse_json_from_response
        fenced = '```json\n{"gases": ["CH4"]}\n```'
        result = _parse_json_from_response(fenced)
        self.assertIsNotNone(result)

    def test_parse_json_invalid_returns_none(self):
        from app.entity_extraction import _parse_json_from_response
        result = _parse_json_from_response("This is not JSON at all.")
        self.assertIsNone(result)

    def test_extract_entities_returns_dict(self):
        from app.entity_extraction import extract_entities
        result = extract_entities(self.SAMPLE_TEXT)
        self.assertIsInstance(result, dict)
        self.assertIn("_source_text_length", result)

    def test_direct_summarise(self):
        from app.entity_extraction import _direct_summarise
        result = _direct_summarise(self.SAMPLE_TEXT, max_words=100, style="factual")
        self.assertIn("summary", result)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 10)

    def test_summarise_document_short_text(self):
        from app.entity_extraction import summarise_document
        result = summarise_document(self.SAMPLE_TEXT)
        self.assertIn("summary", result)
        self.assertIn("strategy", result)


# ═══════════════════════════════════════════════════════════════
#  6. Agent tests
# ═══════════════════════════════════════════════════════════════
class TestAgent(unittest.TestCase):

    def test_classify_intent_co2(self):
        from app.agent import classify_intent
        intents = classify_intent("What is the current CO2 level today?")
        self.assertIn("live_co2", intents)

    def test_classify_intent_temperature(self):
        from app.agent import classify_intent
        intents = classify_intent("What is the current temperature anomaly?")
        self.assertIn("temperature", intents)

    def test_classify_intent_footprint(self):
        from app.agent import classify_intent
        intents = classify_intent("Calculate my carbon footprint")
        self.assertIn("carbon_footprint", intents)

    def test_classify_intent_summarise(self):
        from app.agent import classify_intent
        intents = classify_intent("Summarise this climate report")
        self.assertIn("summarise", intents)

    def test_classify_intent_default(self):
        from app.agent import classify_intent
        intents = classify_intent("What is the Paris Agreement?")
        self.assertIn("general_rag", intents)

    def test_conversation_memory_add(self):
        from app.agent import ConversationMemory
        mem = ConversationMemory(max_turns=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        self.assertEqual(len(mem.history), 2)

    def test_conversation_memory_clear(self):
        from app.agent import ConversationMemory
        mem = ConversationMemory()
        mem.add("user", "Test")
        mem.clear()
        self.assertEqual(len(mem.history), 0)
        self.assertEqual(mem.summary, "")

    def test_conversation_memory_context_string(self):
        from app.agent import ConversationMemory
        mem = ConversationMemory()
        mem.add("user", "What is climate change?")
        mem.add("assistant", "Climate change refers to...")
        ctx = mem.get_context_string()
        self.assertIsInstance(ctx, str)

    def test_parse_footprint_params_car(self):
        from app.agent import _parse_footprint_params
        params = _parse_footprint_params("I drive 15000 km a year")
        self.assertIn("car_km_year", params)
        self.assertAlmostEqual(params["car_km_year"], 15000)

    def test_parse_footprint_params_diet(self):
        from app.agent import _parse_footprint_params
        params = _parse_footprint_params("I am vegan and drive 5000 km")
        self.assertEqual(params.get("diet_type"), "vegan")

    def test_agent_respond_returns_dict(self):
        from app.agent import agent_respond, ConversationMemory
        mem = ConversationMemory()
        result = agent_respond("What causes climate change?", mem)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 5)


# ═══════════════════════════════════════════════════════════════
#  7. RAG tests
# ═══════════════════════════════════════════════════════════════
class TestRAG(unittest.TestCase):

    def test_rag_query_co2(self):
        from app.rag import rag_query
        result = rag_query("What is the current CO2 concentration?")
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("context_used", result)

    def test_rag_query_returns_string_answer(self):
        from app.rag import rag_query
        result = rag_query("Tell me about sea level rise")
        self.assertIsInstance(result["answer"], str)

    def test_retrieve_returns_list(self):
        from app.rag import retrieve
        docs = retrieve("greenhouse gas emissions", k=3)
        self.assertIsInstance(docs, list)

    def test_ingest_text(self):
        from app.rag import ingest_text, retrieve
        test_text = (
            "Xanthium climate test data: The fictional XYZ gas reaches 999 ppm. "
            "Test ingestion only."
        )
        chunks = ingest_text(test_text, source="test", topic="test")
        self.assertGreaterEqual(chunks, 1)

    def test_vectorstore_stats(self):
        from app.rag import get_vectorstore_stats
        stats = get_vectorstore_stats()
        self.assertIn("document_count", stats)
        self.assertGreaterEqual(stats["document_count"], 0)


# ═══════════════════════════════════════════════════════════════
#  8. Multimodal tests
# ═══════════════════════════════════════════════════════════════
class TestMultimodal(unittest.TestCase):

    def test_process_unknown_extension(self):
        from app.multimodal import process_upload
        result = process_upload("/fake/path/file.xyz", original_name="file.xyz")
        self.assertEqual(result["type"], "unknown")
        self.assertIsNotNone(result["error"])

    def test_process_text_file(self):
        import tempfile
        from app.multimodal import process_upload
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                         delete=False, encoding="utf-8") as f:
            f.write(
                "Global CO2 emissions rose to 36.8 GtCO2 in 2023 according to IEA. "
                "This is the highest level ever recorded. "
                "Fossil fuels remain the dominant source."
            )
            tmp_path = f.name
        result = process_upload(tmp_path, original_name="test.txt")
        self.assertEqual(result["type"], "text")
        self.assertGreater(result["chunks_added"], 0)
        os.unlink(tmp_path)

    def test_process_csv_file(self):
        import tempfile
        from app.multimodal import process_upload
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                          delete=False, encoding="utf-8") as f:
            f.write("year,co2_ppm,temp_anomaly\n")
            f.write("2020,412.5,1.02\n")
            f.write("2021,414.7,1.05\n")
            f.write("2022,418.6,1.10\n")
            f.write("2023,421.1,1.17\n")
            tmp_path = f.name
        result = process_upload(tmp_path, original_name="climate_data.csv")
        self.assertEqual(result["type"], "csv")
        self.assertGreater(result["chunks_added"], 0)
        os.unlink(tmp_path)

    def test_image_validation(self):
        from app.multimodal import validate_image
        # Test with a non-existent file
        valid, msg = validate_image("/nonexistent/path/image.png")
        self.assertFalse(valid)


# ═══════════════════════════════════════════════════════════════
#  9. API tests (with TestClient)
# ═══════════════════════════════════════════════════════════════
class TestAPI(unittest.TestCase):

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from api import app
            self.client = TestClient(app)
            self.api_available = True
        except Exception:
            self.api_available = False

    def test_health_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")

    def test_stats_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("knowledge_base", data)

    def test_live_co2_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.get("/live/co2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("ppm", data)

    def test_live_temperature_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.get("/live/temperature")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("anomaly_c", data)

    def test_chat_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.post(
            "/chat",
            json={"message": "What is climate change?", "session_id": "test-session"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertGreater(len(data["answer"]), 5)

    def test_footprint_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.post(
            "/footprint",
            json={
                "car_km_year": 12000, "flights_short": 3,
                "flights_long": 1, "electricity_kwh_month": 300,
                "natural_gas_kwh_month": 80, "diet_type": "average",
                "goods_spend_monthly_usd": 250,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_tonnes_co2e", data)

    def test_extract_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.post(
            "/extract",
            json={"text": "IPCC reports 1.5°C warming. CO2 at 421 ppm. Arctic warming fast."},
        )
        self.assertEqual(resp.status_code, 200)

    def test_summarise_endpoint(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.post(
            "/summarise",
            json={
                "text": "Global temperatures have risen. CO2 is at record highs. "
                        "Sea levels are rising. The IPCC warns of further changes.",
                "max_words": 100,
                "style": "factual",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("summary", data)

    def test_upload_unsupported_type(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.post(
            "/upload",
            files={"file": ("test.exe", b"binary data", "application/octet-stream")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_clear_session(self):
        if not self.api_available:
            self.skipTest("FastAPI not available")
        resp = self.client.delete("/session/test-session-to-clear")
        self.assertEqual(resp.status_code, 200)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧪 Running ClimateBot Test Suite\n" + "=" * 50)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestTools))
    suite.addTests(loader.loadTestsFromTestCase(TestEntityExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestRAG))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodal))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
