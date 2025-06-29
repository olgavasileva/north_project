import os
import sys
import json
import pandas as pd
from datetime import datetime
from psycopg2.extras import Json
from src.utils.db import get_connection
from src.model.preprocess import preprocess
from src.model.model_runner import load_model
from src.utils.stats import get_correlation_matrix, get_shap_values


def main():
    try:
        with get_connection() as conn:
            # Get earliest unprocessed date
            date_query = """
                SELECT MIN(DATE(timestamp)) AS earliest_date
                FROM incoming_data
                WHERE processed = FALSE
            """
            target_date = pd.read_sql(date_query, conn).iloc[0]["earliest_date"]

            if target_date is pd.NaT or pd.isnull(target_date):
                print("!===No unprocessed data remaining.\n")
                return
            print(f"===Processing insights for earliest unprocessed date: {target_date}\n")
            # Fetch unprocessed rows
            query_daily = """
                SELECT * FROM incoming_data
                WHERE DATE(timestamp) = %s AND processed = FALSE
                ORDER BY timestamp
            """
            df = pd.read_sql(query_daily, conn, params=[target_date])
            if df.empty:
                print(f"!===No unprocessed data found for {target_date}\n")
                return

            proc_df   = df.copy()

            model     = load_model("assets/xgb_model.json")
            feat_cols = model.get_booster().feature_names
            proc_df   = preprocess(proc_df, model_features=feat_cols)
            X         = proc_df.drop(columns=["mental_health_status"])

            corr_map = get_correlation_matrix(proc_df, n_feat=5)
            top_features = get_shap_values(X, model, n_feat=5)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO daily_insights (insight_date, top_stress_features_shap, correlations_pearson)
                    VALUES (%s, %s, %s);
                """, (target_date, Json(top_features), Json(corr_map)))

                ids = df["id"].tolist()
                cur.execute("UPDATE incoming_data SET processed = TRUE WHERE id = ANY(%s);", (ids,))
                conn.commit()

            print(f"===Daily insights saved for {target_date}\n")

            # Historical insights
            df_all = pd.read_sql("SELECT * FROM incoming_data WHERE processed = TRUE", conn)
            if df_all.empty:
                print("No processed data available for historical insights.")
                return

            df_all = preprocess(df_all, model_features=feat_cols)
            X_all  = df_all.drop(columns=["mental_health_status"])
            top_features_all = get_shap_values(X_all, model, n_feat=5)
            corr_map_all     = get_correlation_matrix(df_all, n_feat=5)

            date_range = df_all.index.normalize().unique()
            time_range = f"{date_range.min().date()} - {target_date}"
            days_analyzed = len(date_range)

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO historical_insights (time_range, top_stress_features_shap, correlations_pearson, days_analyzed)
                    VALUES (%s, %s, %s, %s);
                """, (time_range, Json(top_features_all), Json(corr_map_all), days_analyzed))
                conn.commit()

            print(f"===Historical insights updated for range: {time_range}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()