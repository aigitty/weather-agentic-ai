# service.py
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import psycopg2
import psycopg2.extras
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------- Logging: JSON to file ----------
from pythonjsonlogger import jsonlogger
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("service")
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("logs/service.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(fh)

# ---------- Env / DB ----------
load_dotenv()
DB_USER = os.getenv("POSTGRES_USER", "weather_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "weather_pass")
DB_NAME = os.getenv("POSTGRES_DB", "weather_db")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_HOST = "localhost"

OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")

# ---------- Prometheus metrics ----------
REQ_COUNT = Counter("service_requests_total", "HTTP requests", ["method", "path", "status"])
REQ_LAT = Histogram("service_request_duration_seconds", "Request duration (s)", ["path"])
TASKS_CREATED = Counter("a2a_tasks_created_total", "Tasks created", ["type"])
TASKS_POLLED = Counter("a2a_tasks_polled_total", "Tasks polled", ["worker"])
TASKS_COMPLETED = Counter("a2a_tasks_completed_total", "Tasks completed")
WEATHER_CALLS = Counter("weather_calls_total", "Weather tool invocations", ["source"])

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

app = FastAPI(title="Weather Agent Service")

# ---------- Middleware: metrics per request ----------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        resp: Response = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        dur = time.perf_counter() - start
        path = request.url.path
        REQ_COUNT.labels(request.method, path, str(locals().get("status", 500))).inc()
        REQ_LAT.labels(path).observe(dur)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# ---------- Prometheus scrape ----------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------- MCP-style agent card ----------
@app.get("/.well-known/agent.json")
def agent_card():
    card = {
        "id": os.getenv("AGENT_NAME", "weather-agent-local"),
        "name": os.getenv("AGENT_NAME", "weather-agent-local"),
        "description": "Local Weather Agent – provides forecasts & task endpoints",
        "service_endpoint": os.getenv("AGENT_ENDPOINT", "http://localhost:8000"),
        "capabilities": {
            "tools": [
                {
                    "id": "open-meteo",
                    "name": "Open-Meteo wrapper",
                    "invoke_endpoint": f"{os.getenv('AGENT_ENDPOINT','http://localhost:8000')}/weather",
                    "inputs": {"latitude": "number", "longitude": "number"},
                }
            ],
            "tasks": ["fetch_weather", "enrich", "alert"],
        },
    }
    return JSONResponse(card)

# ---------- A2A: create task ----------
@app.post("/a2a/tasks")
def create_task(body: Dict[str, Any]):
    required = {"requester", "type", "payload"}
    if not required.issubset(body):
        raise HTTPException(status_code=400, detail="Missing required fields")
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            INSERT INTO tasks (requester, type, payload, status, created_at)
            VALUES (%s, %s, %s::jsonb, 'pending', NOW())
            RETURNING id;
            """,
            (body["requester"], body["type"], json.dumps(body["payload"])),
        )
        tid = cur.fetchone()[0]
        conn.commit()
    TASKS_CREATED.labels(body["type"]).inc()
    logger.info({"event": "task_created", "task_id": tid, "type": body["type"]})
    return {"task_id": tid}

# ---------- A2A: poll ----------
@app.get("/a2a/tasks/poll")
def poll_task(worker: str):
    with psycopg2.connect(
        host="localhost",
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
        port=os.getenv("POSTGRES_PORT", 5432)
    ) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, requester, type, payload
                FROM tasks
                WHERE state = 'pending'
                ORDER BY created_at
                LIMIT 1
            """)
            task = cur.fetchone()

            if task:
                cur.execute(
                    "UPDATE tasks SET state='in_progress', worker=%s WHERE id=%s",
                    (worker, task["id"])
                )
                conn.commit()
                logging.info(f"🛠️ Worker {worker} claimed task {task['id']}")
                return {"task": task}
            else:
                return {"task": None}


# ---------- A2A: complete ----------
@app.post("/a2a/tasks/{task_id}/complete")
def complete_task(task_id: int, body: dict):
    with psycopg2.connect(
        host="localhost",
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
        port=os.getenv("POSTGRES_PORT", 5432)
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET state='completed', result=%s, completed_at=NOW() WHERE id=%s",
                (json.dumps(body), task_id)
            )
            conn.commit()
            logging.info(f"✅ Task {task_id} marked as completed")
            return {"status": "completed"}


# ---------- Weather tool ----------
@app.post("/weather")
def weather(body: Dict[str, Any]):
    lat = body.get("latitude")
    lon = body.get("longitude")
    if lat is None or lon is None:
        raise HTTPException(400, "latitude & longitude required")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,precipitation_probability,windspeed_10m,cloudcover",
        "current_weather": "true",
        "timezone": "Asia/Calcutta",
        "forecast_days": 1,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(502, f"Upstream error {r.status_code}")
    WEATHER_CALLS.labels("open-meteo").inc()
    data = r.json()
    current = data.get("current_weather", {})
    region = f"{lat},{lon}"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_events (region, source, ts, temp_c, wind_kph, conditions, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,NOW())
            RETURNING id;
            """,
            (
                region,
                "open-meteo",
                current.get("time"),
                current.get("temperature"),
                current.get("windspeed"),
                str(current.get("weathercode")),
            ),
        )
        eid = cur.fetchone()[0]
        conn.commit()
    logger.info({"event": "weather_event_inserted", "event_id": eid, "region": region})
    return {"status": "ok", "event_id": eid, "current": current}
