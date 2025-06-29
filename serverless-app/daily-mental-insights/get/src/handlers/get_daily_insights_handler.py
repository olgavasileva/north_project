import os
import json
import pandas as pd
from datetime import datetime
from psycopg2.extras import RealDictCursor, Json
from src.utils.db import get_connection
from src.model.preprocess import preprocess
from src.model.model_runner import load_model
from src.utils.stats import get_correlation_matrix, get_shap_values

"""
Tries to fetch daily insights from the DB
If not found (meaning that day might be not over or is in future or never recorded), 
falls back to unprocessed rows in incoming_data table. If those exist there-computes insights on the fly, returns them, and tells the user it's partial.
If not found - returns 500 error.
"""

def lambda_handler(event, context):
    try:
        # Extract date from query param
        date_str = event.get("queryStringParameters", {}).get("date")
        if not date_str:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'date' parameter"})}
        target_date = pd.to_datetime(date_str).date()

        with get_connection() as conn:
            # First try daily_insights table
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM daily_insights
                    WHERE insight_date = %s
                    LIMIT 1
                """, (target_date,))
                row = cur.fetchone()

            if row:
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "source": "database",
                        "message":  f"Queried insights for {target_date}",
                        "insight_date": row["insight_date"].isoformat(),
                        "top_stress_features_shap": row["top_stress_features_shap"],
                        "correlations_pearson": row["top_stress_features_shap"]
                    })
                }

            # Fallback to unprocessed incoming data
            query = """
                SELECT * FROM incoming_data
                WHERE DATE(timestamp) = %s AND processed = FALSE
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=[target_date])
            if df.empty:
                return {
                    "statusCode": 404,
                    "body": json.dumps({"error": f"No data found for {date_str}"})
                }

            # Compute insights on the fly
            model     = load_model("/opt/python/assets/xgb_model.json")
            feat_cols = model.get_booster().feature_names
            proc_df   = preprocess(df, model_features=feat_cols)

            X = proc_df.drop(columns=["mental_health_status"])
            shap_features = get_shap_values(X, model, n_feat=5)
            corr_map = get_correlation_matrix(proc_df, n_feat=5)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "source": "computed (fallback)",
                    "message":  "These daily insights coming from unprocessed source (day isn't over?)",
                    "insight_date": date_str,
                    "top_stress_features_shap": shap_features,
                    "correlations_pearson": corr_map
                })
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }