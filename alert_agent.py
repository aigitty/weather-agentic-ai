import requests, os, logging, requests
from datetime import datetime
from dotenv import load_dotenv
from llm_nvidia import chat_text

load_dotenv()
log = logging.getLogger(__name__)
BASE_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")

def classify_alert_multi(precip, wind, humidity, temp, pressure, region):
    """
    India-style multi-hazard classification.
    Returns (overall_alert_level, detected_hazards[list], explanation[str])
    """
    hazards = []
    level = "NORMAL"

    # --- Thunderstorm / Lightning ---
    if humidity > 80 and wind > 30 and precip > 5:
        hazards.append("Thunderstorm / Lightning")

    # --- Heavy Rain / Flood ---
    if precip > 65:
        hazards.append("Heavy Rain / Flood Risk")
        level = "ORANGE" if precip <= 115 else "RED"

    # --- Cyclone / Gale / Storm Surge ---
    if wind > 60:
        hazards.append("Strong Wind / Storm / Gale")
        if wind > 90:
            level = "RED"
        elif level != "RED":
            level = "ORANGE"
        if pressure < 1000:
            hazards.append("Possible Cyclone or Storm Surge")

    # --- Heatwave ---
    if temp >= 40 and humidity < 40:
        hazards.append("Heatwave")
        level = max(level, "ORANGE", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Cold Wave ---
    if temp <= 4:
        hazards.append("Cold Wave / Frost")
        level = max(level, "YELLOW", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Dust Storm ---
    if wind > 50 and humidity < 30:
        hazards.append("Dust Storm")
        level = max(level, "ORANGE", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Dense Fog ---
    if humidity > 95 and wind < 5:
        hazards.append("Dense Fog")
        level = max(level, "YELLOW", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Landslide (for hilly regions) ---
    hilly_keywords = ["himalaya", "sikkim", "manipur", "mizoram", "uttarakhand", "himachal", "darjeeling", "meghalaya"]
    if any(k in region.lower() for k in hilly_keywords) and precip > 25:
        hazards.append("Landslide Risk")
        level = max(level, "ORANGE", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Avalanche ---
    if any(k in region.lower() for k in ["ladakh", "kargil", "leh", "gulmarg", "manali"]) and temp > 0 and precip > 10:
        hazards.append("Avalanche Risk")
        level = max(level, "RED", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    # --- Drought / Dry Spell ---
    if precip < 1 and humidity < 40 and temp > 35:
        hazards.append("Dry Spell / Drought Risk")
        level = max(level, "YELLOW", key=["NORMAL","YELLOW","ORANGE","RED"].index)

    if not hazards:
        hazards.append("No significant hazard detected")
        explanation = "Weather appears stable with no immediate risks."
    else:
        explanation = f"Potential hazards: {', '.join(hazards)}"

    return level, hazards, explanation

def run_alert_agent(region: str, lat: float, lon: float):
    log.info(f"⚠️ Running Enhanced Alert Agent for {region}")

    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation,windspeed_10m,relativehumidity_2m,pressure_msl"
        f"&current_weather=true&timezone=Asia/Calcutta"
    )
    data = requests.get(url, timeout=20).json()

    current = data.get("current_weather", {})
    hourly = data.get("hourly", {})

    precip = max(hourly.get("precipitation", [0])[-3:])
    wind = current.get("windspeed") or max(hourly.get("windspeed_10m", [0])[-3:])
    humidity = hourly.get("relativehumidity_2m", [50])[-1]
    temp = current.get("temperature") or hourly.get("temperature_2m", [0])[-1]
    pressure = hourly.get("pressure_msl", [1013])[-1]

    alert_level, hazards, explanation = classify_alert_multi(precip, wind, humidity, temp, pressure, region)

    raw_summary = (
        f"IMD-style derived alert for {region}: {alert_level}\n"
        f"{explanation}\n"
        f"Metrics → Rain: {precip:.1f} mm/hr, Wind: {wind:.1f} km/h, "
        f"Humidity: {humidity:.1f} %, Temp: {temp:.1f} °C, Pressure: {pressure:.1f} hPa"
    )

    prompt_path = os.path.join(os.path.dirname(__file__), "alert_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm_prompt = f"{system_prompt}\n\nRaw data summary:\n{raw_summary}\n\nWrite a concise public alert (IMD style) with safety tips."
    llm_summary = chat_text(llm_prompt, system="You are an official IMD-style alert summarizer.")

    return {
        "region": region,
        "alert_summary": llm_summary.strip(),
        "alert_level": alert_level,
        "hazards": hazards,
        "raw_metrics": {
            "precip": precip, "wind": wind,
            "humidity": humidity, "temp": temp, "pressure": pressure
        },
    }
