import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
import numpy as np

# --- DB connection ---
conn = psycopg2.connect(
    dbname="weather_db",
    user="weather_user",
    password="weather_pass",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# --- Load embedding model ---
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Ensure embeddings table exists ---
cur.execute("""
CREATE TABLE IF NOT EXISTS weather_event_embeddings (
    id SERIAL PRIMARY KEY,
    weather_event_id INT REFERENCES weather_events(id),
    model TEXT,
    vector vector(768),
    metadata JSONB
);
""")
conn.commit()

# --- Fetch events without embeddings ---
cur.execute("""
SELECT id, region, ts, temp_c, wind_kph, conditions
FROM weather_events
WHERE id NOT IN (SELECT weather_event_id FROM weather_event_embeddings);
""")
rows = cur.fetchall()

print(f"Found {len(rows)} new weather events to embed...")

for row in rows:
    event_id, region, ts, temp_c, wind_kph, conditions = row

    # Convert row into descriptive text
    text = f"Weather in region {region} at {ts}: temperature {temp_c}°C, wind {wind_kph} kph, conditions {conditions}"

    # Create embedding
    embedding = model.encode(text).astype(np.float32)

    # Insert into embeddings table
    cur.execute("""
        INSERT INTO weather_event_embeddings (weather_event_id, model, vector, metadata)
        VALUES (%s, %s, %s, %s)
    """, (
        event_id,
        "sentence-transformers/all-mpnet-base-v2",
        embedding.tolist(),
        Json({"text": text})
    ))
    conn.commit()

print("✅ Embeddings created and stored!")

cur.close()
conn.close()
