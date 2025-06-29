import os
import json
import pandas as pd
from datetime import datetime
from psycopg2.extras import Json
from src.utils.db import get_connection
from src.model.preprocess import preprocess
from src.model.model_runner import load_model
from src.utils.stats import get_correlation_matrix, get_shap_values
from datetime import datetime, timedelta, timezone


def lambda_handler(event, context):
    try:
        query_params = event.get("queryStringParameters", {})
        scheduler = query_params.get("scheduler") if query_params else False
        date_str = query_params.get("date") if query_params else None

        if date_str:
            target_date = pd.to_datetime(date_str).date()
        else:
            if scheduler:
                target_date = datetime.now(timezone.utc).date() - timedelta(days=1)
            else:
                return {"statusCode": 400, "body": json.dumps({"error": "Missing 'date' parameter, provide date: YYYY-MM-DD"})}
    
        model     = load_model("/opt/python/assets/xgb_model.json")
        feat_cols = model.get_booster().feature_names

        with get_connection() as conn:
            # === DAILY INSIGHTS ===
            check_df = pd.read_sql(
                """
                SELECT 1 FROM daily_insights
                WHERE insight_date = %s
                LIMIT 1
                """,
                conn,
                params=[target_date]
            )
            
            if not check_df.empty:
                raise ValueError(f"Insight for {target_date} already exists in daily_insights.")
            
            df_daily = pd.read_sql(
                """
                SELECT * FROM incoming_data
                WHERE DATE(timestamp) = %s AND processed = FALSE
                ORDER BY timestamp
                """,
                conn, params=[target_date]
            )

            if df_daily.empty:
                return {
                    "statusCode": 404,
                    "body": json.dumps({"error": f"No unprocessed data found for {target_date}"})
                }
            expected_intervals = 24 * 4
            if len(df_daily) < 96:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Insufficient data for a full day insight. Expected 96 entries, found {len(df_daily)} for {target_date}."})
                }
        
            df_proc = preprocess(df_daily, model_features=feat_cols)
            X_daily = df_proc.drop(columns=["mental_health_status"])
            top_features = get_shap_values(X_daily, model, n_feat=5)
            correlation_map = get_correlation_matrix(df_proc, n_feat=5)


            # Save daily insights
            with conn.cursor() as cur:

                cur.execute("""
                    INSERT INTO daily_insights (insight_date, top_stress_features_shap, correlations_pearson)
                    VALUES (%s, %s, %s);
                """, (target_date, Json(top_features), Json(correlation_map)))

                ids = df_daily["id"].tolist()
                cur.execute(
                    "UPDATE incoming_data SET processed = TRUE WHERE id = ANY(%s);",
                    (ids,)
                )
                conn.commit()

            # === HISTORICAL INSIGHTS (AFTER DAILY ARE PROCESSED) ===
            df_all = pd.read_sql(
                """
                SELECT * FROM incoming_data
                WHERE timestamp <= %s
                """,
                conn,
                params=[target_date]
            )
            if df_all.empty:
                return {
                    "statusCode": 404,
                    "body": json.dumps({"error": "No data is available for historical insights."})
                }

            proc_df_all = preprocess(df_all, model_features=feat_cols)
            X_all = proc_df_all.drop(columns=["mental_health_status"])
            top_features_all = get_shap_values(X_all, model, n_feat=5)
            correlation_map_all = get_correlation_matrix(proc_df_all, n_feat=5)

            date_range = proc_df_all.index.normalize().unique()
            time_range = f"{date_range.min().date()} to {date_range.max().date()}"
            days_analyzed = len(date_range)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO historical_insights (time_range, top_stress_features_shap, correlations_pearson, days_analyzed)
                    VALUES (%s, %s, %s, %s);
                """, (time_range, Json(top_features_all), Json(correlation_map_all), days_analyzed))
                conn.commit()

                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": f"Historical insights were updated to include {date_str} data.",
                        "top_stress_features_shap": top_features,
                        "correlations_pearson": correlation_map
                    })
                }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }