import requests
from datetime import date, timedelta
from llm_nvidia import chat_text
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_archive_data(lat: float, lon: float):
    """Fetch past 30 days of weather data from Open-Meteo Archive API."""
    today = date.today()
    start_date = today - timedelta(days=30)

    params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date.isoformat(),
    "end_date": today.isoformat(),
    "daily": (
        "temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,rain_sum,snowfall_sum,"
        "wind_speed_10m_max,wind_gusts_10m_max,"
        "wind_direction_10m_dominant,weathercode,"
        "shortwave_radiation_sum,sunshine_duration"
    ),
    "timezone": "auto",
}


    url = "https://archive-api.open-meteo.com/v1/archive"
    logger.info(f"🌤 Fetching 30-day archive data for lat={lat}, lon={lon}")
    r = requests.get(url, params=params)
    if r.status_code != 200:
        logger.error(f"Archive API returned {r.status_code}: {r.text[:200]}")
        r.raise_for_status()

    return r.json()


def analyze_archive(region: str, lat: float, lon: float):
    """Run Archive Agent to analyze 30-day climate trend."""
    try:
        data = fetch_archive_data(lat, lon)
    except Exception as e:
        logger.exception(f"Archive fetch failed for {region}: {e}")
        return f"⚠️ Unable to retrieve archive data for {region}. Please try again later."

    prompt = open("archive_prompt.txt", encoding="utf-8").read()
    filled_prompt = prompt.format(region=region, history=data)

    logger.info(f"🧩 Running Archive Agent for {region}")
    try:
        response = chat_text(filled_prompt)
        os.makedirs("logs", exist_ok=True)
        with open("logs/archive.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] Region: {region}\n")
            f.write(f"{response}\n{'-'*80}\n")
        logger.info(f"✅ Archive summary ready for {region}")
        return response
    except Exception as e:
        logger.exception(f"LLM failure in Archive Agent for {region}: {e}")
        return f"⚠️ Analysis failed for {region} due to LLM issue."
