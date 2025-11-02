import sys
import requests
import logging

if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else input("Enter region: ")
    logging.info(f"🌍 Querying Smart Weather Analyst for {region}...")
    try:
        r = requests.post(
            "http://127.0.0.1:8000/weather/analyze",
            json={"region": region},
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        print("\n=== Final Answer ===")
        print(data.get("summary", "No summary returned."))
    except Exception as e:
        logging.error(f"Request failed: {e}")
