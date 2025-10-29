# langgraph_orchestrator.py
from __future__ import annotations
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated
from operator import or_
# top of file
from langgraph.channels import LastValue
import httpx
import os
import time
import os
import sys
import re
from router_agent import route_question_to_agents
from service import geocode_place



print("🧭 Running orchestrator file from:", os.path.abspath(__file__))
print("Loaded modules containing 'orchestrator':")
for m, mod in sys.modules.items():
    if "orchestrator" in m:
        print("   ", m, "→", getattr(mod, "__file__", mod))

print("🧭 Running orchestrator file from:", os.path.abspath(__file__))


# ---- Config ----
ROOT   = "http://127.0.0.1:8000"
A_ARCH = f"{ROOT}/weather/archive-analyze"
A_SUP  = f"{ROOT}/weather/super-analyze"
A_FC   = f"{ROOT}/weather/forecast-analyze"
A_ALERT = f"{ROOT}/weather/alert-analyze"


print("A_SUP:", A_SUP)
print("A_ARCH:", A_ARCH)
print("A_FC:", A_FC)
print("A_ALERT:", A_ALERT)

# ---- State ----
class OrchestratorState(BaseModel):
    query: str
    region: Optional[str] = None
    intent: Literal["current", "history", "forecast", "alert","mixed"] = "current"

    # ✅ LastValue must be parameterized with the data type it holds
    current_analysis: Annotated[Optional[str], LastValue(str)] = None
    archive_summary:  Annotated[Optional[str], LastValue(str)] = None
    forecast_summary: Annotated[Optional[str], LastValue(str)] = None
    final_message:    Annotated[Optional[str], LastValue(str)] = None
    alert_summary:   Annotated[Optional[str], LastValue(str)] = None 
    
    # router decisions
    want_current: bool = False
    want_history: bool = False
    want_forecast: bool = False
    want_alert: bool = False     
    route_index: int = 0

    meta: Dict[str, Any] = Field(default_factory=dict)

# ---- Very small intent heuristic (fast + local). 
# You can swap this with your local LLM later if you want.
KEYS_HISTORY  = ("past", "yesterday", "last week", "last 7", "last month", "histor")
KEYS_FORECAST = ("tomorrow", "next", "forecast", "coming", "later", "hour", "day")

# ---- HTTP helpers ----
def post_agent(url: str, payload: dict, timeout: float = 50.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---- Nodes ----
def node_router(state: OrchestratorState):
    from router_agent import route_question_to_agents
    q = (state.query or "").lower()

    # 🔒 Hard override for alert-y queries
    ALERT_HINTS = (
        "alert", "alerts", "warning", "warnings", "cyclone", "storm",
        "thunder", "lightning", "squall", "flood", "red alert",
        "orange alert", "yellow alert", "imd"
    )
    if any(k in q for k in ALERT_HINTS):
        # ✅ Deterministic: route to alert (others off by default)
        return {
            "want_current": False,
            "want_history": False,
            "want_forecast": False,
            "want_alert": True,
            "intent": "alert",
            "route_index": 0,
        }

    # Otherwise fall back to the LLM router
    sel = route_question_to_agents(state.query)

    want_current  = bool(sel.get("super_analyze"))
    want_history  = bool(sel.get("archive_analyze"))
    want_forecast = bool(sel.get("forecast_analyze"))
    want_alert    = bool(sel.get("alert_analyze"))

    count = sum([want_current, want_history, want_forecast, want_alert])
    intent = (
        "mixed" if count > 1 else
        "current"  if want_current  else
        "history"  if want_history  else
        "forecast" if want_forecast else
        "alert"    if want_alert    else
        "current"
    )

    # 👇 Add a one-line debug so we can see the router’s decisions
    print(f"[ROUTER] want_current={want_current} want_history={want_history} "
          f"want_forecast={want_forecast} want_alert={want_alert} intent={intent}")

    return {
        "want_current": want_current,
        "want_history": want_history,
        "want_forecast": want_forecast,
        "want_alert": want_alert,
        "intent": intent,
        "route_index": 0,
    }


def node_archive(state: OrchestratorState):
    try:
        res = post_agent(A_ARCH, {"region": state.region})
        return {"archive_summary": res.get("summary")}
    except Exception as e:
        return {"archive_summary": f"⚠️ Archive agent failed: {e}"}


def node_super(state: OrchestratorState):
    try:
        res = post_agent(A_SUP, {"region": state.region})
        return {"current_analysis": res.get("summary")}
    except Exception as e:
        return {"current_analysis": f"⚠️ Super agent failed: {e}"}


def node_forecast(state: OrchestratorState):
    try:
        res = post_agent(A_FC, {"region": state.region})
        return {"forecast_summary": res.get("summary")}
    except Exception as e:
        return {"forecast_summary": f"⚠️ Forecast agent failed: {e}"}


def node_compose(state: OrchestratorState):
    q = (state.query or "").lower()

    # 1) If the router intent is ALERT, return only the alert summary (no softening).
    if state.intent == "alert":
        txt = (state.alert_summary or "").strip()
        return {"final_message": txt or "⚠️ No active alerts reported."}

    # 2) If any alert content exists (mixed or other intents), put it FIRST, then add the rest.
    if state.alert_summary:
        parts = [state.alert_summary.strip()]

        # Prefer the intent-specific section next, then the others.
        if state.intent == "current" and state.current_analysis:
            parts.append(state.current_analysis.strip())
        elif state.intent == "forecast" and state.forecast_summary:
            parts.append(state.forecast_summary.strip())
        elif state.intent == "history" and state.archive_summary:
            parts.append(state.archive_summary.strip())
        else:
            # mixed or unspecified: include all remaining in a sensible order
            if state.current_analysis: parts.append(state.current_analysis.strip())
            if state.forecast_summary: parts.append(state.forecast_summary.strip())
            if state.archive_summary:  parts.append(state.archive_summary.strip())

        text = "\n\n".join(p for p in parts if p)
        return {"final_message": text or "⚠️ No data composed."}

    # 3) No alerts present → keep your existing intent-driven behavior.
    if state.intent == "history":
        text = (state.archive_summary or "").strip()
    elif state.intent == "forecast":
        if "week" in q:
            return {"final_message": (state.forecast_summary or "").strip() or "⚠️ No data composed."}
        text = (state.forecast_summary or "").strip()
    elif state.intent == "current":
        text = (state.current_analysis or "").strip()
    else:
        # mixed: current → history → forecast (your original order, without alerts)
        parts = []
        if state.current_analysis: parts.append(state.current_analysis.strip())
        if state.archive_summary:  parts.append(state.archive_summary.strip())
        if state.forecast_summary: parts.append(state.forecast_summary.strip())
        text = "\n\n".join(p for p in parts if p)

    return {"final_message": text or "⚠️ No data composed."}



def node_reason(state: OrchestratorState):
    """Let the LLM reason about user context (e.g., 'Can I go for a run?')."""
    from llm_nvidia import chat_text

    summary = state.final_message or ""
    query = state.query
    prompt_path = os.path.join(os.path.dirname(__file__), "reason_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

    full_prompt = f"""{SYSTEM_PROMPT}

    User question: {query}

    Weather summary from data agents:
    {summary}"""

    response = chat_text(full_prompt, system="You are a helpful contextual weather reasoning agent.")
    return {"final_message": response.strip()}

def node_alert(state: OrchestratorState):
    """Call backend /weather/alert-analyze for alerts."""
    import requests
    region = state.region
    try:
        lat, lon = geocode_place(region)   # ✅ replaced get_lat_lon
        if not lat or not lon:
            raise ValueError(f"Could not geocode region: {region}")

        r = requests.post(
            A_ALERT,   # ✅ use defined A_ALERT constant
            json={"region": region, "lat": lat, "lon": lon}
        )
        js = r.json()
        return {"alert_summary": js.get("alert_summary", "No alert data.")}
    except Exception as e:
        return {"alert_summary": f"⚠️ Alert agent failed: {e}"}


# ---- Graph wiring ----
def build_graph():
    g = StateGraph(OrchestratorState)

    g.add_node("router", node_router)
    g.add_node("super", node_super)
    g.add_node("archive", node_archive)
    g.add_node("forecast", node_forecast)
    g.add_node("compose", node_compose)
    g.add_node("alert", node_alert)


    # Start → Router
    g.add_edge(START, "router")

    # Router → first chosen agent
    # langgraph_orchestrator.py -> build_graph() -> first_hop
    def first_hop(state: OrchestratorState) -> str:
        if state.want_alert:
            return "alert"      # ← alert first
        if state.want_current:
            return "super"
        if state.want_history:
            return "archive"
        if state.want_forecast:
            return "forecast"
        return "compose"


    g.add_conditional_edges("router", first_hop, {
    "alert": "alert",
    "super": "super",
    "archive": "archive",
    "forecast": "forecast",
    "compose": "compose",
    })

    # After SUPER, go to ALERT if requested; else ARCHIVE if requested; else COMPOSE
    g.add_conditional_edges(
        "super",
        lambda s: "alert" if s.want_alert else ("archive" if s.want_history else "compose"),
        {"alert": "alert", "archive": "archive", "compose": "compose"},
    )

    # After ARCHIVE, go to ALERT if requested; else FORECAST if requested; else COMPOSE
    g.add_conditional_edges(
        "archive",
        lambda s: "alert" if s.want_alert else ("forecast" if s.want_forecast else "compose"),
        {"alert": "alert", "forecast": "forecast", "compose": "compose"},
    )

    # NEW: After FORECAST, still allow ALERT; else COMPOSE
    g.add_conditional_edges(
        "forecast",
        lambda s: "alert" if s.want_alert else "compose",
        {"alert": "alert", "compose": "compose"},
    )

    # Reasoning node (unchanged)
    g.add_node("reason", node_reason)

    # Compose → Reason → END (unchanged), and ensure ALERT flows into COMPOSE
    g.add_edge("alert", "compose")
    g.add_edge("compose", "reason")
    g.add_edge("reason", END)


    return g.compile()


# Convenience entry point
graph = build_graph()

def extract_region(text: str) -> str:
    import re
    t = re.sub(r"\s+", " ", (text or "").strip())

    # 1) Prefer "in | at <place>"
    m = re.search(
        r"\b(?:in|at)\s+([A-Za-z][A-Za-z .'-]*?)\s*(?=(?:\bnow\b|\btoday\b|\btomorrow\b|\byesterday\b|\bnext\b|\bthis\b|\bcoming\b|\bafter\b|\bfor\b|\bon\b|\bby\b|[?.!,]|$))",
        t, flags=re.IGNORECASE,
    )

    # 2) Support "for <place>"
    if not m:
        m = re.search(
            r"\bfor\s+([A-Za-z][A-Za-z .'-]*?)\s*(?=(?:\bnow\b|\btoday\b|\btomorrow\b|\bthis\b|\bweek\b|\bweekend\b|\btonight\b|[?.!,]|$))",
            t, flags=re.IGNORECASE,
        )

    candidate = m.group(1).strip() if m else t

    STOP = {
        "now","today","tomorrow","yesterday","tonight","morning","evening",
        "afternoon","week","weekend","and","this","any","alerts","forecast","weather"
    }

    tokens = [w for w in candidate.split() if w.lower() not in STOP]
    name = " ".join(tokens).strip()
    return re.sub(r"\s{2,}", " ", name)


def run_query(query: str, region: Optional[str] = None) -> OrchestratorState:
    """Main entry to run LangGraph with router + agents + compose."""
    region_clean = extract_region(region or query)
    print(f"🧭 Router deciding for region: {region_clean}")

    # initialize state
    init = OrchestratorState(query=query, region=region_clean)

    # build and run the graph
    graph = build_graph()
    result = graph.invoke(init)

    # Convert result into OrchestratorState object
    if isinstance(result, dict):
        return OrchestratorState(**result)
    return result


def wait_for_service(url="http://127.0.0.1:8000/health", timeout=10):
    import time, requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print("✅ Backend service is up!")
                return True
        except Exception:
            pass
        print("⏳ Waiting for backend service to be ready...")
        time.sleep(1)
    raise RuntimeError("Backend service not reachable on port 8000")


if __name__ == "__main__":
    wait_for_service()
    print("\n🌦  LangGraph Orchestrator Interactive Mode 🌦")
    while True:
        q = input("\nEnter your weather question (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        result = run_query(q)
        print("\n🪄 Final Answer:\n", result.final_message)

