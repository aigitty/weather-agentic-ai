# db.py
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "weather_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "weather_pass")
DB_NAME = os.getenv("POSTGRES_DB", "weather_db")
DB_HOST = os.getenv("DB_HOST", "localhost")   # use localhost for host -> container port mapping
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

def insert_weather(event: dict):
    """
    event: dict with keys: region, source, ts (ISO string), temp_c, wind_kph, precip_mm, conditions, raw (dict)
    """
    sql = """
    INSERT INTO weather_events (region, source, ts, temp_c, wind_kph, precip_mm, conditions, raw)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    conn = None
    try:
        conn = get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    event.get("region"),
                    event.get("source"),
                    event.get("ts"),
                    event.get("temp_c"),
                    event.get("wind_kph"),
                    event.get("precip_mm"),
                    event.get("conditions"),
                    Json(event.get("raw", {}))
                ))
                new_id = cur.fetchone()[0]
                return new_id
    finally:
        if conn:
            conn.close()
