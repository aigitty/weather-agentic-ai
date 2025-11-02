import requests
import logging
from datetime import datetime, timedelta, timezone
import os
from core.llm_nvidia import chat_text
import time
logging.basicConfig(level=logging.INFO)

def safe_fetch(url, params, name):
    """Helper to fetch data safely and log warnings instead of crashing."""
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 400:
            logging.warning(f"{name} returned 400 — skipping.")
            return {}
        r.raise_for_status()
        logging.info(f"✅ {name} fetched successfully.")
        return r.json()
    except Exception as e:
        logging.warning(f"⚠️ {name} fetch failed: {e}")
        return {}

def fetch_archive(lat, lon):
    end = datetime.now(timezone.utc).date() - timedelta(days=2)
    start = end - timedelta(days=30)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max,pressure_msl,cloud_cover_mean"
        ),
        "timezone": "auto",
    }
    return safe_fetch("https://archive-api.open-meteo.com/v1/archive", params, "Archive API")

def fetch_current(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "timezone": "auto",
    }
    return safe_fetch("https://api.open-meteo.com/v1/forecast", params, "Current Weather API")

def fetch_forecast(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max,relative_humidity_2m_max,relative_humidity_2m_min,"  # <- fix names
            "uv_index_max,cloud_cover_mean,apparent_temperature_max,apparent_temperature_min"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }
    return safe_fetch("https://api.open-meteo.com/v1/forecast", params, "Forecast API")

def fetch_hourly_past(lat, lon):
    """Fetch past 24 hours of hourly data."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 1,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "windspeed_10m,pressure_msl,cloudcover,apparent_temperature"
        ),
        "timezone": "auto",
    }
    time.sleep(1)
    return safe_fetch("https://api.open-meteo.com/v1/forecast", params, "Past Hourly API")

def fetch_hourly_next(lat, lon):
    """Fetch next 24 hours of hourly forecast."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 2,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "windspeed_10m,pressure_msl,cloudcover,apparent_temperature"
        ),
        "timezone": "auto",
    }
    time.sleep(1)
    return safe_fetch("https://api.open-meteo.com/v1/forecast", params, "Next Hourly API")


def analyze_weather(region, lat, lon):
    logging.info(f"🧩 Running Super Analyzer Agent for {region}")

    # Step 1: Fetch datasets independently
    past_hourly = fetch_hourly_past(lat, lon)
    current = fetch_current(lat, lon)
    next_hourly = fetch_hourly_next(lat, lon)
    
    # Graceful fallback if all data sources fail
    if not any([past_hourly, current, next_hourly]):
        return f"⚠️ Unable to retrieve sufficient data for {region}. Please try again later."

    # Step 2: Merge them into one unified structure
    # Step 2: Merge and tag datasets with time context
    merged = {
        "region": region,
        "latitude": lat,
        "longitude": lon,
        "past_hourly": {
            "type": "historical",
            "data": past_hourly.get("hourly", {})
        },
        "current": {
            "type": "current",
            "data": current.get("current_weather", {})
        },
        "next_hourly": {
            "type": "forecast",
            "data": next_hourly.get("hourly", {})
        },
        "sources_used": [
            name for name, data in {
                "past_hourly": past_hourly,
                "current": current,
                "next_hourly": next_hourly,
            }.items() if data
        ],
    }

    # Step 3: Load prompt template
    PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "prompts", "super_prompt.txt")
    )
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        base_prompt  = f.read()

    prompt = base_prompt.format(
        region=region,
        history=merged["past_hourly"],
        current=merged["current"],
        forecast=merged["next_hourly"],
    )

    # Step 4: Ask the LLM to reason
    response = chat_text(prompt)
    os.makedirs("logs", exist_ok=True)
    with open("logs/super.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Region: {region}\n")
        f.write(f"{response}\n{'-'*80}\n")
    logging.info(f"✅ LLM Response Ready for {region}")
    return response
