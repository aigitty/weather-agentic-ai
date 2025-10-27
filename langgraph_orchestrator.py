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

    meta: Dict[str, Any] = Field(default_factory=dict)

# ---- Very small intent heuristic (fast + local). 
# You can swap this with your local LLM later if you want.
KEYS_HISTORY  = ("past", "yesterday", "last week", "last 7", "last month", "histor")
KEYS_FORECAST = ("tomorrow", "next", "forecast", "coming", "later", "hour", "day")
import re

def detect_intent(text: str) -> str:
    t = (text or "").lower()

    # --- HISTORY ---
    if (
        "last" in t or "ago" in t or "past" in t or "before" in t
        or "day before yesterday" in t
        or re.search(r"\byest[a-z]*\b", t)   # yesterday, yestersay, yerstdays...
        or re.search(r"\b(on|for)\s+\d{1,2}\s+\w+\s*(\d{4})?\b", t)  # explicit past dates
    ):
        return "history"

    # --- FORECAST: add strong week signals ---
    if any(x in t for x in [
        "tomorrow","next","coming","future","later","upcoming",
        "this week","next week","over the week","for the week",
        "next 7 days","coming week","weekend","this weekend"
    ]):
        return "forecast"

    # If a relative phrase like "today" appears with "this week", prefer forecast
    if ("this week" in t or "next 7 days" in t) and ("today" in t or "now" in t):
        return "forecast"

    # --- CURRENT ---
    if any(x in t for x in ["current","now","today","as of","right now","present"]):
        return "current"

    # --- Simple date heuristic stays as you had, if needed ---

    return "mixed"


# ---- HTTP helpers ----
def post_agent(url: str, payload: dict, timeout: float = 25.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---- Nodes ----
def node_router(state: OrchestratorState):
    region = state.region.strip() if state.region else state.query.strip()
    intent = state.intent or detect_intent(state.query)
    # return only what changed
    return {"region": region, "intent": intent}


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

    g.add_node("super", node_super)
    g.add_node("archive", node_archive)
    g.add_node("forecast", node_forecast)
    g.add_node("compose", node_compose)

    # Entry routing
    g.add_conditional_edges(
        START,
        lambda s: s.intent,
        {
            "current": "super",
            "history": "archive",
            "forecast": "forecast",
            "mixed": "super",        # run all, starting at super
        },
    )

    # After SUPER: go to ARCHIVE only for mixed, else compose
    g.add_conditional_edges(
        "super",
        lambda s: "archive" if s.intent == "mixed" else "compose",
        {"archive": "archive", "compose": "compose"},
    )

    # After ARCHIVE: go to FORECAST only for mixed, else compose
    g.add_conditional_edges(
        "archive",
        lambda s: "forecast" if s.intent == "mixed" else "compose",
        {"forecast": "forecast", "compose": "compose"},
    )

    # After FORECAST: always compose
    g.add_edge("forecast", "compose")

    # Final
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
    region_clean = extract_region(region or query)
    intent = detect_intent(query)

    # ✅ Add this line here:
    print(f"🧭 Intent: {intent}, Region: {region_clean}")

    init = OrchestratorState(
        query=query,
        region=region_clean,
        intent=intent,
    )
    result = graph.invoke(init)
    return OrchestratorState(**result)


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

