import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("OPEN_METEO_URL")

def get_weather(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,precipitation_probability,windspeed_10m,cloudcover",
        "current_weather": "true",
        "timezone": "Asia/Calcutta",
        "forecast_days": 1
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        return data["current_weather"], data
    else:
        return None, {"error": f"Failed to fetch weather. Status code: {response.status_code}"}


if __name__ == "__main__":
    # Example: Bangalore
    lat, lon = 12.9716, 77.5946
    current, raw = get_weather(lat, lon)
    if current:
        print("Current weather:", current)
        # build event structure
        event = {
            "region": "Bengaluru",
            "source": "open-meteo",
            "ts": current.get("time") or datetime.now(timezone.utc).isoformat(),
            "temp_c": current.get("temperature"),
            "wind_kph": current.get("windspeed"),
            "precip_mm": None,   # Open-Meteo current_weather may not include precip in same object
            "conditions": current.get("weathercode"),
            "raw": raw
        }

        # Insert into Postgres
        try:
            from db import insert_weather
            new_id = insert_weather(event)
            print(f"Inserted weather event id={new_id}")
        except Exception as e:
            print("DB insert failed:", e)
    else:
        print("Failed to fetch weather:", raw)
