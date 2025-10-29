import json
from typing import Dict
from llm_nvidia import chat_text

SYSTEM_PROMPT = """You are a routing controller for a weather agentic system.

Available agents:
1) super_analyze      → for current/real-time questions, alerts, 'now', 'today'.
2) archive_analyze    → for past/historical questions (yesterday, last week, past dates).
3) forecast_analyze   → for future/predictions (tomorrow, next week, upcoming dates).

Given a user question, respond with JSON ONLY, no text. The JSON keys:
{
  "super_analyze": true/false,
  "archive_analyze": true/false,
  "forecast_analyze": true/false
}"""

def route_question_to_agents(user_question: str) -> Dict[str, bool]:
    """Ask LLM to select which agents to call. Returns a dict of 3 booleans."""
    # Combine the system and user text into one prompt
    prompt = f"""{SYSTEM_PROMPT}

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
        }
    except Exception:
        # Safe default: if parsing fails, assume current only
        return {"super_analyze": True, "archive_analyze": False, "forecast_analyze": False}
