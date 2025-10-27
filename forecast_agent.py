import sys
import requests
import json
import logging
from llm_nvidia import chat_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Step 1: Fetch 7-day forecast from Open-Meteo ---
def fetch_forecast(lat: float, lon: float) -> dict:
    params = {
    "latitude": lat,
    "longitude": lon,
    "daily": (
        "temperature_2m_max,temperature_2m_min,"
        "apparent_temperature_max,apparent_temperature_min,"
        "precipitation_sum,wind_speed_10m_max,"
        "wind_gusts_10m_max,cloud_cover_mean,"
        "snowfall_sum,precipitation_hours,"
        "sunrise,sunset,shortwave_radiation_sum"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }


    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# --- Step 2: Load forecasting prompt template ---
def load_prompt(region: str, forecast_json: dict) -> str:
    try:
        with open("forecast_prompt.txt", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        raise RuntimeError("⚠️ forecast_prompt.txt missing — please create it in project root")

    return template.format(region=region, forecast=json.dumps(forecast_json, indent=2))


# --- Step 3: Generate forecast summary using LLM ---
def forecast_summary(region: str, lat: float, lon: float) -> str:
    logging.info(f"🌤️ Generating 7-day forecast summary for {region} ({lat}, {lon})")
    forecast = fetch_forecast(lat, lon)
    prompt = load_prompt(region, forecast)
    response = chat_text(prompt)
    return response


# --- Step 4: CLI entry point ---
if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else input("Enter region: ")

    # quick geocoding using Open-Meteo’s helper API
    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={region}&count=1&language=en&format=json").json()
    if not geo.get("results"):
        print("❌ Could not find coordinates for that region.")
        sys.exit(1)

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    summary = forecast_summary(region, lat, lon)

    print("\n=== 7-Day Forecast Summary ===")
    print(summary)
