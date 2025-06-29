import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from src.utils.db import get_connection 


CSV_PATH = "assets/university_mental_health_iot_dataset.csv"

def load_csv_to_incoming_table(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df["processed"] = False
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows = [
    (
        row.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        row.location_id,
        row.temperature_celsius,
        row.humidity_percent,
        row.air_quality_index,
        row.noise_level_db,
        row.lighting_lux,
        row.crowd_density,
        row.stress_level,
        row.sleep_hours,
        row.mood_score,
        row.mental_health_status,
        row.processed
    )
    for row in df.itertuples(index=False)
    ]

    insert_sql = """
        INSERT INTO incoming_data (
            timestamp, location_id, temperature_celsius, humidity_percent,
            air_quality_index, noise_level_db, lighting_lux, crowd_density,
            stress_level, sleep_hours, mood_score, mental_health_status, processed
        ) VALUES %s
    """



    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, rows)
            print(f"Inserted {len(rows)} rows into incoming_data.")

if __name__ == "__main__":
    load_csv_to_incoming_table()