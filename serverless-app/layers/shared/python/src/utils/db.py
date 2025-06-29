import os
import psycopg2

def get_connection():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        database=os.environ.get("PGDATABASE", "users"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "example"),
        port=os.environ.get("PGPORT", 5432)
    )