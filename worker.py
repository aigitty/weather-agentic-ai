# worker.py
import os
import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import uuid
import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from prometheus_client import Counter, Gauge, start_http_server
from pythonjsonlogger import jsonlogger

# ---------- Logging ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("logs/worker.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(fh)

# ---------- Env ----------
load_dotenv()
DB_USER = os.getenv("POSTGRES_USER", "weather_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "weather_pass")
DB_NAME = os.getenv("POSTGRES_DB", "weather_db")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_HOST = "localhost"

OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "http://127.0.0.1:8000")
SERVICE_URL = os.getenv("SERVICE_URL", AGENT_ENDPOINT)

# ---------- Metrics (Prometheus) ----------
WEATHER_FETCH_TOTAL = Counter("worker_weather_fetch_total", "Weather fetch attempts", ["region", "status"])
WEATHER_EMBED_TOTAL = Counter("worker_weather_embed_total", "Embeddings created", ["model"])
WORKER_LOOP_ERRORS_TOTAL = Counter("worker_loop_errors_total", "Auto-fetch loop errors", ["stage"])
A2A_POLLS_TOTAL = Counter("worker_a2a_polls_total", "A2A polls")
A2A_TASKS_CLAIMED = Counter("worker_a2a_tasks_claimed_total", "A2A tasks claimed", ["type"])
A2A_TASKS_DONE = Counter("worker_a2a_tasks_done_total", "A2A tasks completed", ["type"])

# ---------- DB ----------
def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def insert_weather_event(region, source, ts, temp_c, wind_kph, conditions):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_events (region, source, ts, temp_c, wind_kph, conditions, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (region, ts) DO NOTHING
            RETURNING id;
            """,
            (region, source, ts, temp_c, wind_kph, conditions),
        )
        row = cur.fetchone()
        if row:
            eid = row[0]
            logger.info({"event": "weather_event_inserted", "event_id": eid, "region": region})
            conn.commit()
            return eid
        else:
            logger.info({"event": "weather_event_skipped", "reason": "duplicate", "region": region, "ts": ts})
            return None


def insert_weather_embedding(event_id: int, model: str, vec, text: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_event_embeddings (weather_event_id, model, vector, metadata)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (weather_event_id, model) DO NOTHING
            RETURNING id;
            """,
            (
                event_id,
                model,
                "[" + ",".join(str(float(x)) for x in vec) + "]",
                json.dumps({"text": text}),
            ),
        )
        row = cur.fetchone()
        if row:
            emb_id = row[0]
            logger.info({"event": "embedding_inserted", "embedding_id": emb_id, "event_id": event_id})
            conn.commit()
            return emb_id
        else:
            logger.info({"event": "embedding_skipped", "reason": "duplicate", "event_id": event_id})
            return None


# ---------- Embedding model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🔧 Using embedding model on {device}")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# ---------- Weather fetch ----------
def fetch_current(lat: float, lon: float, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,precipitation_probability,windspeed_10m,cloudcover",
        "current_weather": "true",
        "timezone": "Asia/Calcutta",
        "forecast_days": 1,
    }
    r = requests.get(OPEN_METEO_URL, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    return r.json().get("current_weather")


# ---------- A2A ----------
async def poll_once():
    A2A_POLLS_TOTAL.inc()
    try:
        r = requests.get(f"{AGENT_ENDPOINT}/a2a/tasks/poll", params={"worker": "worker-1"}, timeout=10)
        if r.status_code != 200:
            return
        data = r.json()
        task = data.get("task")
        if not task:
            return
        A2A_TASKS_CLAIMED.labels(task["type"]).inc()
        # Simple handler: respond OK for now
        requests.post(f"{AGENT_ENDPOINT}/a2a/tasks/{task['id']}/complete", json={"result": {"ok": True}}, timeout=10)
        A2A_TASKS_DONE.labels(task["type"]).inc()
        logger.info({"event": "a2a_task_done", "task_id": task["id"], "type": task["type"]})
    except Exception as e:
        WORKER_LOOP_ERRORS_TOTAL.labels("a2a").inc()
        logger.exception({"event": "a2a_poll_error", "error": str(e)})

# ---------- Auto-fetch + embed ----------
async def auto_fetch_loop():
    region = "12.9716,77.5946"
    lat, lon = 12.9716, 77.5946

    while True:
        try:
            # 🎯 Generate correlation ID
            correlation_id = str(uuid.uuid4())[:8]

            # 🔹 Log before calling /weather
            logger.info({
                "event": "weather_api_triggered",
                "region": region,
                "cid": correlation_id,
                "action": "calling service /weather endpoint"
            })

            # 🔹 Call FastAPI service (not Open-Meteo directly)
            headers = {"X-Correlation-ID": correlation_id, "Content-Type": "application/json"}
            r = requests.post(
                f"{SERVICE_URL}/weather",
                headers=headers,
                json={"latitude": lat, "longitude": lon},
                timeout=20,
            )

            if r.status_code != 200:
                WEATHER_FETCH_TOTAL.labels(region, "fail").inc()
                logger.error({
                    "event": "fetch_fail",
                    "region": region,
                    "cid": correlation_id,
                    "status": r.status_code,
                    "error": r.text,
                })
                eid = None
            else:
                WEATHER_FETCH_TOTAL.labels(region, "ok").inc()
                resp = r.json()
                current = resp.get("current", {})
                eid = resp.get("event_id")
                logger.info({
                    "event": "weather_event_received",
                    "region": region,
                    "cid": correlation_id,
                    "event_id": eid,
                    "current": current,
                })

            # 🔹 Only embed if event_id exists
            if eid:
                ts = current.get("time")
                temp = current.get("temperature")
                wind = current.get("windspeed")
                cond = str(current.get("weathercode"))
                text = f"Weather in region {region} at {ts}: temperature {temp}°C, wind {wind} kph, conditions {cond}"

                vec = embedding_model.encode([text], normalize_embeddings=True)[0]
                emb_id = insert_weather_embedding(eid, "sentence-transformers/all-mpnet-base-v2", vec, text)
                WEATHER_EMBED_TOTAL.labels("sentence-transformers/all-mpnet-base-v2").inc()

                if emb_id:
                    logger.info({
                        "event": "embedding_inserted",
                        "cid": correlation_id,
                        "embedding_id": emb_id,
                        "event_id": eid,
                    })
                else:
                    logger.info({
                        "event": "embedding_skipped",
                        "cid": correlation_id,
                        "region": region,
                        "reason": "duplicate",
                        "event_id": eid
                    })

        except Exception as e:
            WORKER_LOOP_ERRORS_TOTAL.labels("auto_fetch").inc()
            logger.exception({
                "event": "auto_fetch_error",
                "error": str(e),
                "cid": correlation_id,
            })

        # wait a bit (adjust as needed)
        await asyncio.sleep(60)


async def main():
    # Expose worker metrics on :9000
    start_http_server(9000, addr="127.0.0.1")
    logger.info("🚀 Worker started: A2A polling + Auto-fetch loop enabled")

    # run both loops
    await asyncio.gather(
        auto_fetch_loop(),
        *(poll_once() for _ in range(0)),  # no immediate polls at start
        poll_forever()
    )

async def poll_forever():
    while True:
        await poll_once()
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
