CREATE TABLE IF NOT EXISTS daily_insights (
  insight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  insight_date DATE NOT NULL,
  top_stress_features_shap JSONB,
  correlations_pearson JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS historical_insights (
  hinsight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  time_range TEXT NOT NULL, 
  days_analyzed INT NOT NULL, 
  top_stress_features_shap JSONB NOT NULL,  
  correlations_pearson JSONB NOT NULL, 
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS incoming_data (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL,
  location_id INT,
  temperature_celsius DOUBLE PRECISION,
  humidity_percent DOUBLE PRECISION,
  air_quality_index INT,
  noise_level_db DOUBLE PRECISION,
  lighting_lux DOUBLE PRECISION,
  crowd_density INT,
  stress_level INT,
  sleep_hours DOUBLE PRECISION,
  mood_score DOUBLE PRECISION,
  mental_health_status INT,
  processed BOOLEAN DEFAULT FALSE
);
