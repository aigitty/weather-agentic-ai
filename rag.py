import os
import logging
import psycopg2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -------------------
# Logging setup
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------
# Load environment
# -------------------
load_dotenv()

DB_HOST = "localhost"
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL")

if not NVIDIA_API_KEY:
    raise ValueError("❌ NVIDIA_API_KEY is missing. Please set it in your .env file.")

logging.info("✅ NVIDIA API key and settings loaded successfully.")

# -------------------
# DB connection
# -------------------
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    logging.info("✅ Connected to Postgres successfully.")
except Exception as e:
    logging.error(f"❌ Failed to connect to Postgres: {e}")
    raise

# -------------------
# Embedding model (local)
# -------------------
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
logging.info("✅ Loaded local embedding model.")

# -------------------
# NVIDIA client
# -------------------
client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL
)

# -------------------
# RAG function
# -------------------
def rag_query(user_query: str):
    # 1. Embed the query
    query_vec = embedder.encode(user_query, normalize_embeddings=True).tolist()
    logging.info(f"🔎 Created embedding for query: '{user_query[:50]}...'")

    # 2. Search in pgvector
    with conn.cursor() as cur:
        cur.execute("""
            SELECT we.id, we.region, we.ts, we.temp_c, we.wind_kph, we.conditions,
                   (embedding.vector <-> %s::vector) AS distance
            FROM weather_event_embeddings embedding
            JOIN weather_events we ON embedding.weather_event_id = we.id
            ORDER BY distance ASC
            LIMIT 3;
        """, (query_vec,))
        rows = cur.fetchall()

    if not rows:
        logging.warning("⚠️ No relevant results found in DB.")
        return "No relevant weather data found."

    logging.info(f"✅ Retrieved {len(rows)} relevant rows from DB.")

    # 3. Format retrieved context
    context = "\n".join(
        f"At {ts} in {region}: temp={temp_c}°C, wind={wind_kph} kph, conditions={conditions}"
        for (id, region, ts, temp_c, wind_kph, conditions, distance) in rows
    )

    # 4. Call NVIDIA LLM
    logging.info("🤖 Sending context + question to NVIDIA NIM...")
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant with access to local weather history."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]
    )

    answer = response.choices[0].message.content
    logging.info("✅ Got response from NVIDIA LLM.")
    return answer

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag.py \"your question here\"")
        exit(1)

    query = sys.argv[1]
    logging.info(f"🚀 Running RAG pipeline for query: {query}")
    answer = rag_query(query)
    print("\n=== Final Answer ===")
    print(answer)
