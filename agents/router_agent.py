import json
from typing import Dict
from core.llm_nvidia import chat_text
import os

PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "prompts", "router_prompt.txt")
)
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    ROUTER_PROMPT  = f.read()

def route_question_to_agents(user_question: str) -> Dict[str, bool]:
    """Ask LLM to select which agents to call. Returns a dict of 3 booleans."""
    # Combine the system and user text into one prompt
    prompt = f"""{ROUTER_PROMPT }

User question: {user_question}
Respond strictly with JSON only (no explanations)."""

    # ✅ call matches llm_nvidia.chat_text signature
    resp = chat_text(prompt, system="You are an intelligent routing controller for weather agents.")

    # Parse the JSON output
    try:
        j = json.loads(resp)
        return {
            "super_analyze": bool(j.get("super_analyze")),
            "archive_analyze": bool(j.get("archive_analyze")),
            "forecast_analyze": bool(j.get("forecast_analyze")),
            "alert_analyze": bool(j.get("alert_analyze")),
        }
    except Exception:
        # Safe default: if parsing fails, assume current only
        return {"super_analyze": True, "archive_analyze": False, "forecast_analyze": False}
