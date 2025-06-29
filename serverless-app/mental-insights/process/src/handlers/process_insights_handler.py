import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, Json

from src.model.preprocess import preprocess
from src.model.model_runner import load_model
from src.utils.stats import get_correlation_matrix, get_shap_values
from src.utils.db import get_connection


def lambda_handler(event, context):
    try:
        with get_connection() as conn:
            # Fetch data from incoming_data
            query = """
                SELECT
                    id, timestamp, location_id, temperature_celsius, humidity_percent,
                    air_quality_index, noise_level_db, lighting_lux, crowd_density,
                    stress_level, sleep_hours, mood_score, mental_health_status
                FROM incoming_data
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn)
            if df.empty:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "No data found in incoming_data table."})
                }


            model     = load_model("/opt/python/assets/xgb_model.json")
            feat_cols = model.get_booster().feature_names
            proc_df   = preprocess(df, model_features=feat_cols)
            # Calculate metadata
            X = proc_df.drop(columns=["mental_health_status"])
            date_range = proc_df.index.normalize().unique()

            time_range = f"{date_range.min().date()} - {date_range.max().date()}"
            days_analyzed = len(date_range)

            
            shap_top_features = get_shap_values(X, model, n_feat=5)
            corr_map = get_correlation_matrix(proc_df, n_feat=5)

            # Insert into historical_insights
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO historical_insights (time_range, top_stress_features_shap, correlations_pearson, days_analyzed)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (time_range, Json(shap_top_features), Json(corr_map), days_analyzed))
                conn.commit()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Historical insights processed and saved to DB!",
                "time_range": time_range,
                "days_analyzed": days_analyzed,
                "top_stress_features_shap": shap_top_features,
                "correlations_pearson": corr_map,
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }