# service.py
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
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

    priority = body.get("priority", 50)

    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            INSERT INTO tasks (requester, type, payload, state, priority, created_at)
            VALUES (%s, %s, %s::jsonb, 'pending', %s, NOW())
            RETURNING id;
            """,
            (body["requester"], body["type"], json.dumps(body["payload"]), priority),
        )
        task_id = cur.fetchone()[0]
        conn.commit()

    TASKS_CREATED.labels(body["type"]).inc()
    logger.info({"event": "task_created", "task_id": task_id, "type": body["type"], "priority": priority})
    return {"id": task_id, "state": "pending", "priority": priority}

# ---------- A2A: claim next ----------
@app.post("/a2a/tasks/next")
def claim_next_task(body: Dict[str, Any] = None):
    worker = (body or {}).get("worker", "default-worker")

    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            SELECT id, type, payload, priority
            FROM tasks
            WHERE state = 'pending'
            ORDER BY priority DESC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        if not row:
            return {"task": None}

        task_info = {
            "id": row["id"],
            "type": row["type"],
            "payload": row["payload"],
            "priority": row["priority"],
        }

        cur.execute(
            """
            UPDATE tasks
            SET state = 'in_progress', started_at = NOW()
            WHERE id = %s;
            """,
            (row["id"],),
        )
        conn.commit()

    TASKS_POLLED.labels(worker).inc()
    logger.info(f"worker {worker} claimed task {task_info['id']}")
    return {"task": task_info}

# ---------- A2A: poll (GET alias for workers) ----------
@app.get("/a2a/tasks/poll")
def poll_task(worker: str = "default-worker"):
    """Alias endpoint for worker polling; same logic as claim_next_task."""
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """
            SELECT id, type, payload, priority
            FROM tasks
            WHERE state = 'pending'
            ORDER BY priority DESC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        if not row:
            return {"task": None}

        cur.execute(
            "UPDATE tasks SET state='in_progress', started_at=NOW() WHERE id=%s;", (row["id"],)
        )
        conn.commit()

    TASKS_POLLED.labels(worker).inc()
    logger.info(f"worker {worker} polled and claimed task {row['id']}")
    return {"task": dict(row)}


# ---------- A2A: complete ----------
@app.post("/a2a/tasks/{task_id}/complete")
def complete_task(task_id: int, body: dict):
    result = body.get("result", {})
    error = body.get("error")

    new_state = "completed" if error is None else "failed"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = %s,
                result = %s::jsonb,
                error  = %s,
                finished_at = NOW()
            WHERE id = %s;
            """,
            (new_state, json.dumps(result), error, task_id),
        )
        conn.commit()

    if new_state == "completed":
        TASKS_COMPLETED.inc()
    logger.info(f"task {task_id} marked as {new_state}")
    return {"status": new_state, "id": task_id}


# ---------- Weather tool ----------

@app.post("/weather")
def weather(request: Request, body: Dict[str, Any]):
    # 1) Read inputs + CID
    lat = body.get("latitude")
    lon = body.get("longitude")
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="latitude & longitude required")

    cid = request.headers.get("X-Correlation-ID", "none")
    region = f"{lat},{lon}"

    # 2) Build Open-Meteo request
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,precipitation_probability,windspeed_10m,cloudcover",
        "current_weather": "true",
        "timezone": "Asia/Kolkata",  # canonical IANA
        "forecast_days": 1,
    }

    # 3) Log that we’re calling the upstream
    logger.info({"event": "openmeteo_call", "cid": cid, "params": params})

    # 4) Call Open-Meteo
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream error {r.status_code}")

    data = r.json()

    # 5) Normalize a "current" payload
    #    Prefer "current_weather" (Open-Meteo’s field), fallback to "current" if your code already constructs it.
    current = data.get("current_weather") or data.get("current") or {}
    ts = current.get("time")
    temp = current.get("temperature") or current.get("temperature_2m")
    wind = current.get("windspeed") or current.get("windspeed_10m")
    cond = str(current.get("weathercode") or "")

    if not ts:
        # Defensive: derive ts from hourly if needed
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if times:
            ts = times[-1]

    # 6) Idempotent upsert of the weather event; always RETURNING id
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_events (region, source, ts, temp_c, wind_kph, conditions, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (region, ts) DO UPDATE
              SET temp_c = EXCLUDED.temp_c,
                  wind_kph = EXCLUDED.wind_kph,
                  conditions = EXCLUDED.conditions
            RETURNING id;
            """,
            (region, "open-meteo", ts, temp, wind, cond),
        )
        event_id = cur.fetchone()[0]
        conn.commit()

    logger.info({"event": "weather_event_inserted", "cid": cid, "event_id": event_id, "region": region, "ts": ts})
    return {"event_id": event_id, "region": region, "current": current}
