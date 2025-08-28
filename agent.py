import requests
import os
from dotenv import load_dotenv

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
        return data["current_weather"]
    else:
        return {"error": f"Failed to fetch weather. Status code: {response.status_code}"}


if __name__ == "__main__":
    # Example: Bangalore
    weather = get_weather(12.9716, 77.5946)
    print("Current weather:", weather)
