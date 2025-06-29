import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from src.utils.db import get_connection


def lambda_handler(event, context):
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT *
                    FROM historical_insights
                    ORDER BY created_at DESC
                    LIMIT 1;
                """)
                row = cur.fetchone()

        if not row:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "No insights found."})
            }

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message":  "Latest historic insights (computed using all data)",
                "created_at": row["created_at"].isoformat(),
                "time_range": row["time_range"],
                "days_analyzed": row["days_analyzed"],
                "top_stress_features_shap": row["top_stress_features_shap"],
                "correlations_pearson": row["correlations_pearson"]
            }),
            "headers": {
                "Content-Type": "application/json"
            }
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }