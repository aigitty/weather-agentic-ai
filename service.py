# service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import logging

# local imports — uses your existing helpers
from agent import get_weather      # function returns (current, raw) as defined
from db import insert_weather
import db_tasks                    # create_task, fetch_and_claim_next_task, complete_task, fail_task

load_dotenv()
app = FastAPI(title="Weather Agent (A2A-ready)")

AGENT_NAME = os.getenv("AGENT_NAME", "weather-agent-local")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "http://localhost:8000")  # set if different

log = logging.getLogger("uvicorn.error")

class TaskCreate(BaseModel):
    requester: str
    type: str
    payload: dict
    priority: int = 50
    target_agent: str | None = None
    external_id: str | None = None

class TaskComplete(BaseModel):
    result: dict

class WeatherQuery(BaseModel):
    latitude: float
    longitude: float

@app.get("/health")
def health():
    return {"status":"ok","time": datetime.now(timezone.utc).isoformat()}

# Minimal Agent Card for A2A discovery
@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "id": AGENT_NAME,
        "name": AGENT_NAME,
        "description": "Local Weather Agent — provides forecasts & task endpoints",
        "service_endpoint": AGENT_ENDPOINT,
        "capabilities": {
            "tools": [
                {
                    "id": "open-meteo",
                    "name": "Open-Meteo wrapper",
                    "invoke_endpoint": f"{AGENT_ENDPOINT}/weather",
                    "inputs": {"latitude":"number","longitude":"number"}
                }
            ],
            "tasks": ["fetch_weather","enrich","alert"]
        }
    }

# Tool: fetch weather (other agents can call this)
@app.post("/weather")
def weather_tool(q: WeatherQuery):
    try:
        current, raw = get_weather(q.latitude, q.longitude)
        if current is None:
            raise HTTPException(status_code=502, detail="Open-Meteo failed")
        # build an event
        event = {
            "region": f"{q.latitude},{q.longitude}",
            "source": "open-meteo",
            "ts": current.get("time"),
            "temp_c": current.get("temperature"),
            "wind_kph": current.get("windspeed"),
            "precip_mm": None,
            "conditions": current.get("weathercode"),
            "raw": raw
        }
        new_id = insert_weather(event)
        return {"status":"ok","event_id": new_id, "current": current}
    except Exception as e:
        log.exception("weather tool failed")
        raise HTTPException(status_code=500, detail=str(e))

# A2A: create task
@app.post("/a2a/tasks")
def create_task_endpoint(t: TaskCreate):
    try:
        tid = db_tasks.create_task(
            requester=t.requester,
            task_type=t.type,
            payload=t.payload,
            target_agent=t.target_agent,
            priority=t.priority,
            external_id=t.external_id
        )
        return {"task_id": tid}
    except Exception as e:
        log.exception("create_task failed")
        raise HTTPException(status_code=500, detail=str(e))

# A2A: poll and claim next pending task as worker
@app.get("/a2a/tasks/poll")
def poll_task(worker: str):
    try:
        task = db_tasks.fetch_and_claim_next_task(worker)
        if not task:
            return {"task": None}
        return {"task": task}
    except Exception as e:
        log.exception("poll failed")
        raise HTTPException(status_code=500, detail=str(e))

# A2A: complete
@app.post("/a2a/tasks/{task_id}/complete")
def complete_task_endpoint(task_id: int, body: TaskComplete):
    try:
        ok = db_tasks.complete_task(task_id, body.result)
        if ok:
            return {"status":"completed"}
        raise HTTPException(status_code=404, detail="task not found or not updated")
    except Exception as e:
        log.exception("complete failed")
        raise HTTPException(status_code=500, detail=str(e))

# A2A: fail
@app.post("/a2a/tasks/{task_id}/fail")
def fail_task_endpoint(task_id: int, body: dict):
    err = body.get("error","failed")
    try:
        ok = db_tasks.fail_task(task_id, err)
        if ok:
            return {"status":"failed"}
        raise HTTPException(status_code=404, detail="task not found")
    except Exception as e:
        log.exception("fail failed")
        raise HTTPException(status_code=500, detail=str(e))
