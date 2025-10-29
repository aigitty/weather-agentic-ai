import os, requests, json, logging
from llm_nvidia import chat_text

log = logging.getLogger(__name__)

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "alert_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    ALERT_PROMPT = f.read()

def run_alert_agent(region: str, lat: float, lon: float):
    """Fetch and summarize weather alerts (if any) for a given region."""
    log.info(f"⚠️ Running Alert Agent for {region}")

    url = f"https://api.open-meteo.com/v1/weather?latitude={lat}&longitude={lon}&alerts=true"
    resp = requests.get(url, timeout=20)
    data = resp.json()

    alerts = data.get("alerts", {}).get("alert", [])
    if not alerts:
        return {
            "region": region,
            "alert_summary": f"No active weather alerts detected for {region}. Conditions appear normal."
        }

    text_alerts = "\n".join(
        [f"⚠️ {a.get('event')}: {a.get('description', 'No details provided.')}" for a in alerts]
    )

    full_prompt = f"""{ALERT_PROMPT}

Region: {region}
Raw Alerts:
{text_alerts}

Summarize the situation clearly for users."""
    summary = chat_text(full_prompt, system="You are a trusted weather safety advisor.")

    return {"region": region, "alert_summary": summary.strip()}
