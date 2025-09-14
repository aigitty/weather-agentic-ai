import os
import asyncio
import logging
import json
import time
import requests
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load env variables
load_dotenv()

AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "http://localhost:8000")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Bangalore")
AUTO_FETCH_INTERVAL = int(os.getenv("AUTO_FETCH_INTERVAL", "3600"))

# DB config
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
logging.info(f"🔧 Using embedding model on {device}")

# Database connection
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host="localhost",
    port=DB_PORT
)
cur = conn.cursor()

# =============== Helper Functions ===============

def insert_weather_event(region, source, ts, temp, wind, cond):
    cur.execute(
        """
        INSERT INTO weather_events (region, source, ts, temp_c, wind_kph, conditions, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (region, source, ts, temp, wind, cond, datetime.now(timezone.utc)),
    )
    event_id = cur.fetchone()[0]
    conn.commit()
    logging.info(f"🌦️ Stored weather_event id={event_id}")
    return event_id

def embed_weather_event(event_id, text):
    emb = embedding_model.encode([text])[0]
    cur.execute(
        """
        INSERT INTO weather_event_embeddings (weather_event_id, model, vector, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """,
        (event_id, "sentence-transformers/all-mpnet-base-v2", emb.tolist(), Json({"text": text})),
    )
    emb_id = cur.fetchone()[0]
    conn.commit()
    logging.info(f"🧩 Created embedding id={emb_id} for weather_event={event_id}")

def fetch_weather_from_api(location: str):
    lat, lon = 12.9716, 77.5946 if location.lower() == "bangalore" else (0, 0)
    url = f"{os.getenv('OPEN_METEO_URL')}?latitude={lat}&longitude={lon}&current_weather=true"
    res = requests.get(url).json()
    current = res.get("current_weather", {})
    return {
        "region": f"{lat},{lon}",
        "source": "open-meteo",
        "ts": current.get("time"),
        "temp_c": current.get("temperature"),
        "wind_kph": current.get("windspeed"),
        "conditions": str(current.get("weathercode"))
    }

# =============== Async Worker Loops ===============

async def task_poll_loop():
    """Polls for A2A tasks and processes them."""
    while True:
        try:
            resp = requests.get(f"{AGENT_ENDPOINT}/a2a/tasks/poll?worker=worker-1")
            task = resp.json().get("task")
            if not task:
                await asyncio.sleep(5)
                continue

            logging.info(f"📥 Claimed task: {task['id']} {task['type']}")
            if task["type"] == "fetch_weather":
                weather = fetch_weather_from_api(task["payload"]["input"]["location"])
                event_id = insert_weather_event(**weather)
                text = f"Weather in region {weather['region']} at {weather['ts']}: temperature {weather['temp_c']}°C, wind {weather['wind_kph']} kph, conditions {weather['conditions']}"
                embed_weather_event(event_id, text)

                # mark complete
                requests.post(
                    f"{AGENT_ENDPOINT}/a2a/tasks/{task['id']}/complete",
                    json={"status": "completed", "result": {"event_id": event_id}},
                )
                logging.info(f"✅ Task {task['id']} completed -> event {event_id}")

        except Exception as e:
            logging.error(f"❌ Error in task loop: {e}")
            await asyncio.sleep(5)

async def auto_fetch_loop():
    """Periodically fetches + embeds weather without external trigger."""
    while True:
        try:
            weather = fetch_weather_from_api(DEFAULT_LOCATION)
            event_id = insert_weather_event(**weather)
            text = f"Weather in region {weather['region']} at {weather['ts']}: temperature {weather['temp_c']}°C, wind {weather['wind_kph']} kph, conditions {weather['conditions']}"
            embed_weather_event(event_id, text)
            logging.info(f"🔁 Auto-fetch + embed done for {DEFAULT_LOCATION}")
        except Exception as e:
            logging.error(f"❌ Error in auto-fetch loop: {e}")

        await asyncio.sleep(AUTO_FETCH_INTERVAL)

# =============== Entry Point ===============

async def main():
    logging.info("🚀 Worker started: A2A polling + Auto-fetch loop enabled")
    await asyncio.gather(task_poll_loop(), auto_fetch_loop())

if __name__ == "__main__":
    asyncio.run(main())
