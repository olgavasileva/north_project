## do i need to include the dependencies I used for the ipynb
# do i need diff dependencies per endpoint?"
Challenge 1: Data Scientist 

===Goal: Analyze mental health data from the IoT dataset and deploy insights via the serverless app.
===Task:
-Load & Explore Dataset:
-Perform EDA (Exploratory Data Analysis) on university_mental_health_iot_dataset.csv
-Identify key indicators of mental stress (e.g., sensor patterns, time-based trends)
-Create a Notebook or Script:
-Generate summary stats and visualizations
-Identify at least 3 significant features correlated with stress level
-Export Key Insights to an API:
-Add a new Lambda (GET /mental-insights) to return:
 {
  "top_stress_features": ["heart_rate", "sleep_hours", "temperature"],
  "correlations": {
    "heart_rate": 0.79,
    "sleep_hours": -0.52
  }
}
===Bonus: Push daily insights into DB and create an endpoint to query them.

===Skills Tested:
-Data exploration
-Feature engineering
-Visualization
-API design
-Integration with AWS Lambda and DB



1.
All data analysis and feature discovery were conducted in the following Jupyter Notebook: 
/serverless-app/data_exploration.ipynb

2.
The template .yaml file:
/Users/olgavasileva/Desktop/north_project/serverless-app/template.yaml

3.
Endpoints logic:
/Users/olgavasileva/Desktop/north_project/serverless-app/mental-insights
/Users/olgavasileva/Desktop/north_project/serverless-app/daily-mental-insights

4.
Scripts:
/Users/olgavasileva/Desktop/north_project/serverless-app/scripts

5.
ETL/Model:
/Users/olgavasileva/Desktop/north_project/serverless-app/layers/shared/python/src

6.
Compiled model:
/Users/olgavasileva/Desktop/north_project/serverless-app/layers/shared/python/assets/xgb_model.json

7.
DB Schema:
/Users/olgavasileva/Desktop/north_project/serverless-app/db/init.sql


SETUP/DB:
The raw CSV is converted into an incoming_data table to make the data queryable. This table mirrors the CSV fields but also includes a processed boolean flag.
Once daily insights are computed for a given date, the corresponding row is marked as processed = true. This setup helps track progress and is scalable for large datasets or continuous ingestion.

Run this command to initiate tables:
PGPASSWORD=example psql -h localhost -U postgres -d users -f db/init.sql

To populate incoming_data table, run this file:
PYTHONPATH=layers/shared/python python3.11 scripts/load_csv_to_db.py

There’s an older script that was used to pre-populate some tables before the API endpoints were ready. It's mostly obsolete now but still included for reference.
To run it (not recommended anymore):
PYTHONPATH=layers/shared/python python3.11 scripts/precompute_insights.py

Tables can be confirmed with:
psql -h localhost -U postgres -d users
password: example

Since the task involves daily aggregations, not real-time processing or high-throughput ingestion, PostgreSQL is a good fit.
It’s easy to query with SQL, works well with Python and pandas for analytics, and keeps the setup simple and maintainable.

BUILD SAM:
sam build && sam local start-api

API OVERVIEW:
The project exposes four API endpoints, grouped into two categories:

1. Daily Mental Insights (/daily-mental-insights)
These endpoints compute insights from individual full-day data slices. Once a full day of sensor data is ingested, a daily insight can be generated.
This means if we have data for seven days, we get seven distinct daily insights, that are independent of each other, - one per day (e.g., Monday, Tuesday, etc.). These capture stress patterns specific to the 24-hour period.

2. Cumulative Mental Insights (/mental-insights)
These endpoints operate on the full dataset and are designed to reveal long-term or lagged patterns: like recurring weekday stress spikes, weekend sleep trends, or behavioral rhythms that span beyond a single day, and can represent weekly or even monthly trends. Please, refer to data_exploration.ipynb for more info on lagged data.

This separation allows for both granular (daily) and higher-level (cumulative) understanding of mental health signals in the IoT time series data.

3. Two key endpoints handle inference and database updates:
/process-daily-mental-insights
This endpoint first checks if a daily insight for the given date already exists.
If it does, it returns a message that the data has already been processed.
If not, it checks the incoming_data table for a full 24-hour cycle.
If the cycle is incomplete, it returns a message saying that daily insights will be drawn on the fly. Which means that the stats aren't read from the 
daily_insights table (because it is supposed to store only complete cycle records), but rather computes stats directly hitting the model.
I thought this functionality would be good if a user wants to see their stats during the day.
If the 24-hour cycle is complete, the endpoint runs inference using the model and stores the result in the daily_insights table. Storing past data in tables
helps to reduce the resources consumption.
On success, it also triggers an update to the cumulative mental_nsights table (runs inference using the model on the entire dataset).
This logic ensures daily insights are always based on complete, clean daily slices.

/process-mental-insights
This endpoint recomputes the cumulative insights using all available data in incoming_data.
It does not accept user input; it always derives metrics directly from the database.
Typically called automatically after daily insights are updated, but can be triggered manually if needed.

4. Typical response includes a message, and top 5 features based on their absolute SHAP value as well as top 5 features correlated with the mental_health_status the most.
If error happens during the process, an error message is also returned

AUTOMATION
The process-daily-mental-insights endpoint is called via a scheduled Lambda at 2:00 AM every night. It is defined in template.yaml using AWS EventBridge (cron).
At 2AM it insures that  the system isn't overloaded with user requests.If successful, it automatically calls process-mental-insights to keep both tables in sync.


TESTS (don't forget to pre-populate incoming_data table!):
Here we will overview the functionality of the endpoints:
--
1. curl "http://127.0.0.1:3000/mental-insights" | jq
{
  "error": "No insights found."
}

Good, we have none! Let's compute some daily insights so they also trigger mental insights (cumulative) to update.
--

2. curl "http://127.0.0.1:3000/daily-mental-insights?date=2024-05-01"
{
  "source": "computed (fallback)",
  "message": "These daily insights coming from unprocessed source (day isn't over?)",
  "insight_date": "2024-05-01",
  "top_stress_features_shap": {
    "stress_level": 5.4016,
    "air_quality_index": 0.3588,
    "crowd_density_lag_13": 0.1688,
    "sleep_hours": 0.1445,
    "mood_score": 0.1411
  },
  "correlations_pearson": {
    "stress_level": 0.7604,
    "humidity_percent": -0.5564,
    "lighting_lux": -0.5012,
    "hour_cos": -0.4995,
    "hour": -0.4975
  }
}

We just hit a date with non-full cycle (first day of the dataset). Therefore the data was computed on the fly and neither daily_insights nor cumulative_mental_insights were updated. Let's try another date:
--

3. curl -X POST "http://127.0.0.1:3000/process-daily-mental-insights?date=2024-05-02"
{"message": "Historical insights were updated to include 2024-05-02 data.", "top_stress_features_shap": {"stress_level": 5.4068, "air_quality_index": 0.3312, "sleep_hours": 0.1824, "crowd_density_lag_13": 0.1585, "mood_score": 0.1401}, "correlations_pearson": {"stress_level": 0.8298, "air_quality_index": 0.4041, "sleep_hours": -0.3945, "noise_level_db_lag_34": -0.3877, "mood_score": -0.3232}}%

We successfully computed  metrics for one of the days, which should've triggered the cumulative mental-insights to be updated. Let's run the first command again and confirm it:
--

4. curl "http://127.0.0.1:3000/mental-insights" | jq
{
  "message": "Latest historic insights (computed using all data)",
  "created_at": "2025-06-29T02:32:01.373219",
  "time_range": "2024-05-01 to 2024-05-02",
  "days_analyzed": 2,
  "top_stress_features_shap": {
    "mood_score": 0.1404,
    "sleep_hours": 0.1414,
    "stress_level": 5.4453,
    "air_quality_index": 0.3557,
    "crowd_density_lag_13": 0.1624
  },
  "correlations_pearson": {
    "hour_cos": -0.5318,
    "hour_sin": -0.5227,
    "lighting_lux": -0.5142,
    "stress_level": 0.7709,
    "humidity_percent": -0.5548
  }
}

Indeed, the daily-mental-insights has triggered cumulative table to run on all data points. Let's check the created daily-insights record:
--

5. curl "http://127.0.0.1:3000/daily-mental-insights?date=2024-05-02" | jq
{
  "source": "database",
  "message": "Queried insights for 2024-05-02",
  "insight_date": "2024-05-02",
  "top_stress_features_shap": {
    "mood_score": 0.1401,
    "sleep_hours": 0.1824,
    "stress_level": 5.4068,
    "air_quality_index": 0.3312,
    "crowd_density_lag_13": 0.1585
  },
  "correlations_pearson": {
    "mood_score": 0.1401,
    "sleep_hours": 0.1824,
    "stress_level": 5.4068,
    "air_quality_index": 0.3312,
    "crowd_density_lag_13": 0.1585
  }
}
We confirmed that the data computed in 3. was read from the database.
--

6. curl http://127.0.0.1:3000/process-mental-insights -d '{}' 
Lastly, we will hit the endpoint that computes all of the data:
{
  "message": "Historical insights processed and saved to DB!",
  "time_range": "2024-05-01 - 2024-05-11",
  "days_analyzed": 11,
  "top_stress_features_shap": {
    "stress_level": 5.4521,
    "air_quality_index": 0.3299,
    "crowd_density_lag_13": 0.1525,
    "sleep_hours": 0.1468,
    "mood_score": 0.1425
  },
  "correlations_pearson": {
    "stress_level": 0.8093,
    "air_quality_index": 0.585,
    "sleep_hours": -0.424,
    "sleep_hours_lag_21": 0.322,
    "crowd_density": 0.2361
  }
}

--

7. curl http://127.0.0.1:3000/mental-insights |jq
And now let's read them. Despite mental_insights storing multiple rows of the cumulative data, this endpoint only returns the lates row.

{
  "message": "Latest historic insights (computed using all data)",
  "created_at": "2025-06-29T02:44:16.048307",
  "time_range": "2024-05-01 - 2024-05-11",
  "days_analyzed": 11,
  "top_stress_features_shap": {
    "mood_score": 0.1425,
    "sleep_hours": 0.1468,
    "stress_level": 5.4521,
    "air_quality_index": 0.3299,
    "crowd_density_lag_13": 0.1525
  },
  "correlations_pearson": {
    "sleep_hours": -0.424,
    "stress_level": 0.8093,
    "crowd_density": 0.2361,
    "air_quality_index": 0.585,
    "sleep_hours_lag_21": 0.322
  }
}



AREAS OF IMPROVEMENT & FUTURE WORK

1. Refactoring Repeated Code
Right now, there’s a large of repeated logic across different Lambda functions (between the endpoints that process and serve data). A lot of this could be moved into shared utility modules to clean up the code and make it easier to maintain.

2. Use Separate Models for Different Time Scales
At the moment, the same model is being used to generate both daily and historical insights. But realistically, these two use cases should probably have different models:

Daily Model: Should be optimized for short-term windows (e.g., 24 hours), using features with shorter lag structures.
Historical Model: Should be trained on larger temporal spans (e.g., weekly/monthly) with longer-term lagged features.
Maintaining separate models for these distinct tasks would increase flexibility and predictive performance.

3. API Security
The current API setup doesn’t have any access control in place. In a real production environment, it would be important to add authentication and authorization (like API keys or IAM roles) to protect the data and prevent unauthorized access.

4. Add Prediction Endpoint
One useful addition could be a separate endpoint that accepts incoming sensor data and returns a prediction of the mental health status.

5. Better Logging and Monitoring
Future versions should incorporate centralized logging (e.g., CloudWatch) and monitoring (e.g., error rates, request volumes) for better observability and maintenance.

6. Testing and Deployment Workflow
The project would benefit from test coverage .


