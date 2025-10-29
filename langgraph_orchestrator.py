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

print("A_SUP:", A_SUP)
print("A_ARCH:", A_ARCH)
print("A_FC:", A_FC)

# ---- State ----
class OrchestratorState(BaseModel):
    query: str
    region: Optional[str] = None
    intent: Literal["current", "history", "forecast", "mixed"] = "current"

    # ✅ LastValue must be parameterized with the data type it holds
    current_analysis: Annotated[Optional[str], LastValue(str)] = None
    archive_summary:  Annotated[Optional[str], LastValue(str)] = None
    forecast_summary: Annotated[Optional[str], LastValue(str)] = None
    final_message:    Annotated[Optional[str], LastValue(str)] = None
    
    # router decisions
    want_current: bool = False
    want_history: bool = False
    want_forecast: bool = False
    route_index: int = 0

    meta: Dict[str, Any] = Field(default_factory=dict)

# ---- Very small intent heuristic (fast + local). 
# You can swap this with your local LLM later if you want.
KEYS_HISTORY  = ("past", "yesterday", "last week", "last 7", "last month", "histor")
KEYS_FORECAST = ("tomorrow", "next", "forecast", "coming", "later", "hour", "day")

# ---- HTTP helpers ----
def post_agent(url: str, payload: dict, timeout: float = 25.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---- Nodes ----
def node_router(state: OrchestratorState):
    """LLM-based router using router_agent"""
    from router_agent import route_question_to_agents

    sel = route_question_to_agents(state.query)

    want_current  = bool(sel.get("super_analyze"))
    want_history  = bool(sel.get("archive_analyze"))
    want_forecast = bool(sel.get("forecast_analyze"))

    # Derive a human intent label
    count = sum([want_current, want_history, want_forecast])
    intent = (
        "mixed" if count > 1
        else "current" if want_current
        else "history" if want_history
        else "forecast" if want_forecast
        else "current"
    )

    return {
        "want_current": want_current,
        "want_history": want_history,
        "want_forecast": want_forecast,
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
    if state.intent == "history":
        text = (state.archive_summary or "").strip()
    elif state.intent == "forecast":
        # If the question mentions week/weekend, return forecast-only
        if "week" in (state.query or "").lower():
            return {"final_message": (state.forecast_summary or "").strip() or "⚠️ No data composed."}
        text = (state.forecast_summary or "").strip()
    elif state.intent == "current":
        text = (state.current_analysis or "").strip()
    else:
        text = (
            (state.current_analysis or "")
            + ("\n\n" if state.current_analysis else "")
            + (state.archive_summary or "")
            + ("\n\n" if (state.archive_summary and state.forecast_summary) else "")
            + (state.forecast_summary or "")
        ).strip()
    return {"final_message": text or "⚠️ No data composed."}


# ---- Graph wiring ----
def build_graph():
    g = StateGraph(OrchestratorState)

    g.add_node("router", node_router)
    g.add_node("super", node_super)
    g.add_node("archive", node_archive)
    g.add_node("forecast", node_forecast)
    g.add_node("compose", node_compose)

    # Start → Router
    g.add_edge(START, "router")

    # Router → first chosen agent
    def first_hop(state: OrchestratorState) -> str:
        if state.want_current:
            return "super"
        if state.want_history:
            return "archive"
        if state.want_forecast:
            return "forecast"
        return "compose"

    g.add_conditional_edges("router", first_hop, {
        "super": "super",
        "archive": "archive",
        "forecast": "forecast",
        "compose": "compose",
    })

    # Keep the rest of your chain as before
    g.add_conditional_edges(
        "super",
        lambda s: "archive" if s.want_history else "compose",
        {"archive": "archive", "compose": "compose"},
    )
    g.add_conditional_edges(
        "archive",
        lambda s: "forecast" if s.want_forecast else "compose",
        {"forecast": "forecast", "compose": "compose"},
    )
    g.add_edge("forecast", "compose")
    g.add_edge("compose", END)
    return g.compile()


# Convenience entry point
graph = build_graph()

def extract_region(text: str) -> str:
    import re
    t = re.sub(r"\s+", " ", (text or "").strip())  # <-- collapse newlines/spaces
    m = re.search(
        r"\b(?:in|at)\s+([A-Za-z][A-Za-z .'-]*?)\s*(?=(?:\bnow\b|\btoday\b|\btomorrow\b|\byesterday\b|\bnext\b|\bthis\b|\bcoming\b|\bafter\b|\bfor\b|\bon\b|\bby\b|[?.!,]|$))",
        t, flags=re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
    else:
        parts = re.findall(r"[A-Za-z][A-Za-z .'-]*", t)
        candidate = parts[-1].strip() if parts else t

    STOP = {"now","today","tomorrow","yesterday","tonight","morning","evening","afternoon","week","weekend"}
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

