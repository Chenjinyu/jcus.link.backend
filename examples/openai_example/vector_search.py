# vector_search.py
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def query_supabase(query_vector):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    sql = """
    SELECT original_text, model_name, 
           1 - (embedding <=> %s) AS similarity
    FROM embeddings
    ORDER BY embedding <=> %s
    LIMIT 5;
    """

    cur.execute(sql, (query_vector, query_vector))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows
