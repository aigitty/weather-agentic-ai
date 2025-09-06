# worker.py (updated - robust handling of get_weather return types)
import time
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

# local imports (your existing helpers)
import db_tasks
from agent import get_weather     # flexible: may return dict or (current, raw) tuple
from db import insert_weather     # insert_weather(event_dict) -> id

SLEEP_SECONDS = float(os.getenv("WORKER_POLL_INTERVAL", 5))

def extract_current_and_raw(got):
    """
    Accepts multiple shapes from get_weather:
      - None -> (None, None)
      - dict with 'current_weather' -> (current_weather dict, full dict)
      - dict that is already current_weather -> (dict, None)
      - tuple/list (current, raw) -> unpack
    """
    if got is None:
        return None, None
    # tuple/list
    if isinstance(got, (tuple, list)) and len(got) >= 1:
        # unpack first two elements if present
        current = got[0]
        raw = got[1] if len(got) > 1 else None
        return current, raw
    # dict
    if isinstance(got, dict):
        if "error" in got:
            return got, None
        if "current_weather" in got:
            return got["current_weather"], got
        # might already be the current_weather shape
        return got, None
    # unknown
    return None, None

def handle_task(task):
    tid = task["id"]
    ttype = task["type"]
    payload = task.get("payload") or {}
    try:
        if ttype in ("fetch_weather", "weather"):
            lat = None
            lon = None

            # direct lat/lon keys
            if isinstance(payload, dict) and "lat" in payload and "lon" in payload:
                lat = float(payload["lat"]); lon = float(payload["lon"])
            else:
                # nested payload forms
                inp = None
                if isinstance(payload, dict):
                    # payload may be {'input': {...}}
                    inp = payload.get("input") or payload.get("payload") or None
                if inp and isinstance(inp, dict):
                    if "lat" in inp and "lon" in inp:
                        lat = float(inp["lat"]); lon = float(inp["lon"])
                    elif "location" in inp:
                        loc = str(inp["location"]).lower()
                        if "bangalore" in loc or "bengaluru" in loc:
                            lat, lon = 12.9716, 77.5946

            if lat is None or lon is None:
                db_tasks.fail_task(tid, "missing lat/lon in payload")
                print(f"[task {tid}] failed: missing lat/lon")
                return

            # call fetcher
            got = get_weather(lat, lon)
            current, raw = extract_current_and_raw(got)

            # detect error-shaped response
            if isinstance(current, dict) and "error" in current:
                db_tasks.fail_task(tid, f"weather fetch error: {current.get('error')}")
                print(f"[task {tid}] weather fetch error: {current.get('error')}")
                return
            if current is None:
                db_tasks.fail_task(tid, "weather fetch returned no data")
                print(f"[task {tid}] weather fetch returned no data")
                return

            # Build event dict
            event = {
                "region": f"{lat},{lon}",
                "source": "open-meteo",
                "ts": current.get("time") if isinstance(current, dict) else None,
                "temp_c": current.get("temperature") if isinstance(current, dict) else None,
                "wind_kph": current.get("windspeed") if isinstance(current, dict) else None,
                "precip_mm": None,
                "conditions": current.get("weathercode") if isinstance(current, dict) else None,
                "raw": raw or {}
            }

            event_id = insert_weather(event)

            # Insert a dummy embedding for now (safe placeholder)
            try:
                dummy_vector = [0.0] * 8
                db_tasks.insert_embedding(object_type="weather_event", object_id=str(event_id),
                                          model="dummy", vector=dummy_vector, metadata={"lat": lat, "lon": lon})
            except Exception:
                print("[warn] embedding insert failed (non-fatal)")

            result = {"status": "ok", "event_id": event_id}
            db_tasks.complete_task(tid, result)
            print(f"[task {tid}] completed -> event {event_id}")

        else:
            db_tasks.fail_task(tid, f"unknown task type: {ttype}")
            print(f"[task {tid}] unknown task type: {ttype}")

    except Exception as e:
        traceback.print_exc()
        db_tasks.fail_task(tid, f"exception: {e}")
        print(f"[task {tid}] exception: {e}")

def run_worker():
    print("Worker started. Polling for tasks...")
    while True:
        try:
            task = db_tasks.fetch_and_claim_next_task(worker_name="worker-1")
            if task:
                print("Claimed task:", task["id"], task["type"])
                handle_task(task)
            else:
                time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("Worker shutting down (KeyboardInterrupt)")
            break
        except Exception as e:
            print("Worker loop error:", e)
            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    run_worker()
