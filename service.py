# service.py
import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Body
import psycopg2
import psycopg2.extras
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Gauge
from llm_nvidia import chat_text
from pydantic import BaseModel
from forecast_agent import forecast_summary
from alert_agent import run_alert_agent


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
    
class ChatReq(BaseModel):
    query: str
    region: Optional[str] = None

    
def log_event(level, event, **kwargs):
    entry = {"event": event, "service": "weather-service", **kwargs}
    logger.log(level, json.dumps(entry))
    
def geocode_place(region: str):
    """Geocode a place name using Nominatim (primary) and Open-Meteo fallback."""
    import requests

    # 1. Try Nominatim
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": region, "format": "json", "limit": 1},
            headers={"User-Agent": "WeatherAgenticAI"},
            timeout=10
        )
        if r.ok and r.json():
            result = r.json()[0]
            return float(result["lat"]), float(result["lon"])
    except Exception:
        pass

    # 2. Fallback to Open-Meteo
    try:
        g = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": region, "count": 1, "language": "en", "format": "json"},
            timeout=10
        )
        geo = g.json()
        if geo.get("results"):
            return geo["results"][0]["latitude"], geo["results"][0]["longitude"]
    except Exception:
        pass

    return None, None

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
WEATHER_CALLS = Counter("weather_calls_total", "Weather tool invocations", ["source"])
# Track DB health
DB_HEALTH = Gauge("db_connection_alive", "Database connectivity (1=up,0=down)")
REGIONS_REGISTERED_TOTAL = Counter("regions_registered_total", "Regions registered", ["source"])

# Track active connections
ACTIVE_DB_CONNECTIONS = Gauge("service_db_connections", "Active PostgreSQL connections")

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

app = FastAPI(title="Weather Agent Service")

@app.get("/")
def root():
    return {"status": "ok", "message": "Weather Agent API is running"}

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
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
        DB_HEALTH.set(1)
        ACTIVE_DB_CONNECTIONS.set(1)
    except Exception:
        DB_HEALTH.set(0)
        ACTIVE_DB_CONNECTIONS.set(0)
        raise HTTPException(status_code=500, detail="Database unreachable")

    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ---------- Prometheus scrape ----------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class RegionRequest(BaseModel):
    region: str


def geocode_region_name(name: str):
    """Uses Open-Meteo geocoding API to get latitude & longitude."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        r.raise_for_status()
        js = r.json()
        if not js.get("results"):
            raise HTTPException(status_code=404, detail=f"Could not geocode region: {name}")
        res = js["results"][0]
        return res["latitude"], res["longitude"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {e}")

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

@app.get("/weather/archive/.well-known/agent.json")
def archive_agent_card():
    """
    Archive Agent metadata for A2A discovery.
    """
    return {
        "id": "archive-agent",
        "name": "Archive Weather Agent",
        "description": "Analyzes past 30-day weather patterns and trends from Open-Meteo archive API.",
        "service_endpoint": "http://localhost:8000/weather/archive-analyze",
        "capabilities": {
            "tasks": ["analyze_past_weather"],
            "tools": ["fetch_archive_data", "summarize_trends"]
        }
    }

@app.get("/weather/super/.well-known/agent.json")
def super_agent_card():
    """
    Super Analyzer Agent metadata for A2A discovery.
    """
    return {
        "id": "super-analyzer-agent",
        "name": "Super Analyzer Agent",
        "description": (
            "Combines current, past-hourly, and next-hourly weather data for "
            "anomaly detection, trend analysis, and friendly summarization."
        ),
        "service_endpoint": "http://localhost:8000/weather/super-analyze",
        "capabilities": {
            "tasks": ["detect_weather_anomalies", "generate_summary"],
            "tools": ["fetch_hourly_data", "reason_over_conditions"]
        }
    }

@app.get("/weather/forecast/.well-known/agent.json")
def forecast_agent_card():
    """
    Forecast Agent metadata for A2A discovery.
    """
    return {
        "id": "forecast-agent",
        "name": "Forecast Weather Agent",
        "description": "Generates 7-day weather forecasts and summarizes predicted patterns.",
        "service_endpoint": "http://localhost:8000/weather/forecast-analyze",
        "capabilities": {
            "tasks": ["generate_forecast"],
            "tools": ["fetch_forecast_data", "summarize_predictions"]
        }
    }


@app.post("/regions/register")
def register_region(body: Dict[str, Any]):
    name = body.get("name")
    lat = body.get("latitude")
    lon = body.get("longitude")

    if name:
        name = name.strip().title()

    # If name provided but no coordinates -> geocode it
    if name and (lat is None or lon is None):
        lat, lon = geocode_place(name)
        if not lat or not lon:
            raise HTTPException(status_code=400, detail=f"Could not find coordinates for {name}")
        REGIONS_REGISTERED_TOTAL.labels("name").inc()
    else:
        REGIONS_REGISTERED_TOTAL.labels("coords").inc()

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="latitude and longitude required")

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid lat/lon bounds.")

    region_name = (name or f"Region_{round(lat,2)}_{round(lon,2)}").strip().title()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO regions (name, latitude, longitude, active)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (latitude, longitude)
            DO UPDATE SET active = TRUE, name = EXCLUDED.name
            RETURNING id;
        """, (region_name, lat, lon))
        region_id = cur.fetchone()[0]
        conn.commit()

    logger.info({
        "event": "region_registered",
        "name": region_name,
        "lat": lat,
        "lon": lon,
        "region_id": region_id
    })
    return {
        "region_id": region_id,
        "name": region_name,
        "latitude": lat,
        "longitude": lon,
        "status": "active"
    }
    
@app.post("/weather/live")
def get_live_weather(body: Dict[str, Any]):
    lat, lon = body["latitude"], body["longitude"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "precipitation,pressure_msl,cloud_cover,"
            "wind_speed_10m,apparent_temperature,uv_index"
        ),
        "current_weather": True,
        "timezone": "auto",
    }

    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream error {r.status_code}")

    data = r.json()
    # merge current snapshot + latest hourly sample
    result = {
        "current_weather": data.get("current_weather", {}),
        "hourly": {k: v[-1] if isinstance(v, list) else v for k, v in data.get("hourly", {}).items()},
        "meta": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "elevation": data.get("elevation"),
            "timezone": data.get("timezone"),
        },
    }
    return result

@app.post("/weather/history")
def get_weather_history(body: Dict[str, Any]):
    lat, lon = body["latitude"], body["longitude"]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,wind_speed_10m_max,"
            "humidity_2m_max,humidity_2m_min,"
            "uv_index_max,pressure_msl,"
            "snowfall_sum,cloud_cover_mean,"
            "apparent_temperature_max,apparent_temperature_min,"
            "sea_level_pressure_mean"
        ),
        "timezone": "auto"
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Archive API error {r.status_code}")
    return r.json()  # ✅ Return full archive JSON


@app.post("/weather/analyze")
def analyze_weather(body: Dict[str, Any]):
    region = body.get("region")
    if not region:
        raise HTTPException(status_code=400, detail="region required")

    # 1️⃣ Get coordinates
    lat, lon = geocode_place(region)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail=f"Could not find coordinates for {region}")

    # 2️⃣ Fetch current + past data
    current = get_live_weather({"latitude": lat, "longitude": lon})
    history = get_weather_history({"latitude": lat, "longitude": lon})

    # 3️⃣ Load external prompt from file and fill placeholders
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt file missing or unreadable: {e}")

    prompt = prompt_template.format(
        region=region,
        history=json.dumps(history, indent=2),
        current=json.dumps(current, indent=2)
    )
    
    logger.info({"event": "prompt_loaded", "path": prompt_path, "region": region})

    # 4️⃣ Invoke LLM (use your ChatNVIDIA wrapper)
        # 4️⃣ Invoke LLM
    response = chat_text(prompt)

    # 5️⃣ Store both raw data and LLM summary in DB
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO weather_logs (region, summary, created_at) VALUES (%s, %s, NOW());",
                (region, response)
            )
            conn.commit()
        logger.info({"event": "weather_summary_saved", "region": region})

    except Exception as e:
        logger.warning({"event": "weather_log_failed", "region": region, "error": str(e)})

    return {"region": region, "summary": response}


@app.post("/weather/forecast-analyze")
def forecast_analyze(req: RegionRequest):
    region = req.region.strip()
    lat, lon = geocode_place(region)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail=f"Could not geocode region: {region}")

    summary = forecast_summary(region, lat, lon)
    return {"region": region, "summary": summary}


# ---------- Archive Agent ----------
@app.post("/weather/archive-analyze")
def archive_analyze(body: Dict[str, Any]):
    region = body.get("region")
    if not region:
        raise HTTPException(status_code=400, detail="region is required")

    # --- Step 1: Geocode region ---
        # --- Step 1: Geocode region (using unified geocoder) ---
    lat, lon = geocode_place(region)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail=f"Could not geocode region: {region}")

    # --- Step 2: Import and run Archive Agent ---
    from archive_agent import analyze_archive
    result = analyze_archive(region, lat, lon)

    # --- Step 3: Optional DB log ---
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO weather_logs (region, summary, created_at) VALUES (%s, %s, NOW());",
                (region, result),
            )
            conn.commit()
        logger.info({"event": "archive_analyze_saved", "region": region})
    except Exception as e:
        logger.warning({"event": "archive_analyze_log_failed", "error": str(e)})

    return {"region": region, "summary": result}


@app.post("/weather/super-analyze")
def super_analyze(body: Dict[str, Any]):
    region = body.get("region")
    if not region:
        raise HTTPException(status_code=400, detail="region is required")

    # --- Step 1: Geocode region ---
        # --- Step 1: Geocode region (using unified geocoder) ---
    lat, lon = geocode_place(region)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail=f"Could not geocode region: {region}")

    # --- Step 2: Run Super Analyzer ---
    from super_analyzer_agent import analyze_weather
    result = analyze_weather(region, lat, lon)

    # --- Step 3: Optional DB log (if you want to store summary) ---
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO weather_logs (region, summary, created_at) VALUES (%s, %s, NOW());",
                (region, result),
            )
            conn.commit()
        logger.info({"event": "super_analyze_saved", "region": region})
    except Exception as e:
        logger.warning({"event": "super_analyze_log_failed", "error": str(e)})

    return {"region": region, "summary": result}

@app.post("/chat")
def chat_route(payload: ChatReq):
    # import only when endpoint is hit
    from langgraph_orchestrator import run_query
    st = run_query(payload.query, payload.region)
    return {
        "region": st.region,
        "intent": st.intent,
        "final": st.final_message,
        "debug": {
            "archive_summary": st.archive_summary,
            "current_analysis": st.current_analysis,
            "forecast_summary": st.forecast_summary,
        },
    }

@app.post("/weather/alert-analyze")
def alert_analyze(payload: dict):
    region = payload.get("region")
    lat = payload.get("lat")
    lon = payload.get("lon")
    if not region or lat is None or lon is None:
        return {"error": "Missing region, lat, or lon"}
    return run_alert_agent(region, lat, lon)