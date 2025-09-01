# db_tasks.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "weather_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "weather_pass")
DB_NAME = os.getenv("POSTGRES_DB", "weather_db")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )

# Create a new task; returns id
def create_task(requester: str, task_type: str, payload: dict, target_agent: str = None, priority: int = 50, external_id: str = None):
    sql = """
    INSERT INTO tasks (external_id, requester, target_agent, type, state, priority, payload, created_at)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    external_id,
                    requester,
                    target_agent,
                    task_type,
                    "pending",
                    priority,
                    Json(payload),
                    datetime.utcnow()
                ))
                tid = cur.fetchone()[0]
                return tid
    finally:
        conn.close()

# Fetch next pending task (simple FIFO by priority then created_at)
def fetch_and_claim_next_task(worker_name: str):
    # This finds one pending task and sets its state to in_progress and sets started_at and target_agent
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                WITH c AS (
                  SELECT id FROM tasks
                  WHERE state = 'pending'
                  ORDER BY priority ASC, created_at ASC
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                )
                UPDATE tasks
                SET state = 'in_progress', started_at = now(), target_agent = %s
                WHERE id IN (SELECT id FROM c)
                RETURNING id, requester, type, payload;
                """, (worker_name,))
                row = cur.fetchone()
                if not row:
                    return None
                tid, requester, ttype, payload = row
                return {"id": tid, "requester": requester, "type": ttype, "payload": payload}
    finally:
        conn.close()

def complete_task(task_id: int, result: dict):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE tasks
                SET state = 'completed', result = %s, finished_at = now()
                WHERE id = %s
                RETURNING id;
                """, (Json(result), task_id))
                return cur.rowcount == 1
    finally:
        conn.close()

def fail_task(task_id: int, error_msg: str):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE tasks
                SET state = 'failed', error = %s, finished_at = now()
                WHERE id = %s
                RETURNING id;
                """, (error_msg, task_id))
                return cur.rowcount == 1
    finally:
        conn.close()

def insert_embedding(object_type: str, object_id: str, model: str, vector: list, metadata: dict = None):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO embeddings (object_type, object_id, model, vector, metadata)
                VALUES (%s,%s,%s,%s,%s) RETURNING id;
                """, (object_type, object_id, model, vector, Json(metadata or {})))
                return cur.fetchone()[0]
    finally:
        conn.close()
