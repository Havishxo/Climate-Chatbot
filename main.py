"""
main.py — Entry point for ClimateBot.

Usage:
  python main.py           # Launch Gradio UI (default, port 7860)
  python main.py --api     # Launch FastAPI server (port 8000)
  python main.py --demo    # Run a quick terminal demo
"""

import sys
import os
import logging
import argparse

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def run_ui():
    """Launch the Gradio web interface."""
    logger.info("Starting ClimateBot Gradio UI on http://0.0.0.0:7860")
    from app.ui import build_ui
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


def run_api():
    """Launch the FastAPI REST server."""
    import uvicorn
    logger.info("Starting ClimateBot API on http://0.0.0.0:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


def run_demo():
    """Run an interactive terminal demo."""
    from app.agent import agent_respond, ConversationMemory
    from app.tools import get_live_co2, get_temperature_anomaly

    print("\n" + "=" * 60)
    print("  🌍  ClimateBot — Terminal Demo")
    print("=" * 60)

    # Show live data
    print("\n📡 Fetching live climate data...")
    co2 = get_live_co2()
    temp = get_temperature_anomaly()
    print(f"  CO₂: {co2['ppm']} ppm [{co2['source']}]")
    print(f"  Temperature anomaly: +{temp['anomaly_c']} °C [{temp['source']}]")

    memory = ConversationMemory()
    print("\n💬 Chat started. Type 'quit' to exit.\n")

    sample_questions = [
        "What is the current CO2 level and why does it matter?",
        "How fast are sea levels rising?",
        "What are the most effective climate solutions?",
    ]

    for q in sample_questions:
        print(f"User: {q}")
        result = agent_respond(q, memory)
        answer = result["answer"]
        # Trim for terminal readability
        print(f"ClimateBot: {answer[:500]}{'...' if len(answer) > 500 else ''}\n")

    print("\n" + "=" * 60)
    print("Demo complete. Run 'python main.py' for the full web UI.")


def main():
    parser = argparse.ArgumentParser(description="ClimateBot — AI Climate Awareness Chatbot")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI server instead of UI")
    parser.add_argument("--demo", action="store_true", help="Run terminal demo")
    parser.add_argument("--port", type=int, default=None, help="Override default port")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.api:
        run_api()
    else:
        run_ui()


if __name__ == "__main__":
    main()
