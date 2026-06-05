"""
AQI Predictor - Feature Pipeline
---------------------------------
This script:
1. Fetches raw AQI + weather data from AQICN API
2. Engineers features (time-based + derived)
3. Stores features in Hopsworks Feature Store

Run this script every hour via GitHub Actions (CI/CD step).
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import hopsworks

# ─────────────────────────────────────────────
# CONFIG — replace with your actual keys
# or set them as environment variables
# ─────────────────────────────────────────────
AQICN_TOKEN    = os.getenv("AQICN_TOKEN", "YOUR_AQICN_TOKEN_HERE")
HOPSWORKS_KEY  = os.getenv("HOPSWORKS_API_KEY", "YOUR_HOPSWORKS_KEY_HERE")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictorrr")
CITY           = "karachi"   # change to your city if needed


# ─────────────────────────────────────────────
# STEP 1: Fetch raw data from AQICN
# ─────────────────────────────────────────────
def fetch_aqi_data(city: str, token: str) -> dict:
    """Fetch current AQI and pollutant readings for a city."""
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data["status"] != "ok":
        raise ValueError(f"AQICN API error: {data.get('data', 'Unknown error')}")

    return data["data"]


# ─────────────────────────────────────────────
# STEP 2: Engineer features from raw data
# ─────────────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Extract and compute features from raw AQICN response.
    
    Model inputs (features):
      - pollutants: pm25, pm10, o3, no2, so2, co
      - weather:    temperature, humidity, wind speed, pressure
      - time-based: hour, day_of_week, month, is_weekend
      - derived:    aqi_category (encoded)

    Model output (target):
      - aqi: the current AQI value (used as target during training)
    """
    iaqi = raw.get("iaqi", {})
    now  = datetime.now(timezone.utc)

    # --- pollutants (use 0.0 if sensor not available) ---
    pm25    = float(iaqi.get("pm25",  {}).get("v", 0.0))
    pm10    = float(iaqi.get("pm10",  {}).get("v", 0.0))
    o3      = float(iaqi.get("o3",    {}).get("v", 0.0))
    no2     = float(iaqi.get("no2",   {}).get("v", 0.0))
    so2     = float(iaqi.get("so2",   {}).get("v", 0.0))
    co      = float(iaqi.get("co",    {}).get("v", 0.0))

    # --- weather conditions ---
    temperature = float(iaqi.get("t",  {}).get("v", 0.0))
    humidity    = float(iaqi.get("h",  {}).get("v", 0.0))
    wind_speed  = float(iaqi.get("w",  {}).get("v", 0.0))
    pressure    = float(iaqi.get("p",  {}).get("v", 0.0))

    # --- current AQI (this is the TARGET for ML) ---
    aqi = float(raw.get("aqi", 0))

    # --- time-based features ---
    hour        = now.hour
    day_of_week = now.weekday()        # 0=Monday … 6=Sunday
    month       = now.month
    is_weekend  = int(now.weekday() >= 5)

    # --- derived feature: AQI category (0-5 scale) ---
    # Good=0, Moderate=1, Unhealthy for Sensitive=2,
    # Unhealthy=3, Very Unhealthy=4, Hazardous=5
    if aqi <= 50:
        aqi_category = 0
    elif aqi <= 100:
        aqi_category = 1
    elif aqi <= 150:
        aqi_category = 2
    elif aqi <= 200:
        aqi_category = 3
    elif aqi <= 300:
        aqi_category = 4
    else:
        aqi_category = 5

    row = {
        # identifiers
        "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
        "city":         CITY,

        # pollutants
        "pm25":         pm25,
        "pm10":         pm10,
        "o3":           o3,
        "no2":          no2,
        "so2":          so2,
        "co":           co,

        # weather
        "temperature":  temperature,
        "humidity":     humidity,
        "wind_speed":   wind_speed,
        "pressure":     pressure,

        # time features
        "hour":         hour,
        "day_of_week":  day_of_week,
        "month":        month,
        "is_weekend":   is_weekend,

        # derived
        "aqi_category": aqi_category,

        # target
        "aqi":          aqi,
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# STEP 3: Store features in Hopsworks
# ─────────────────────────────────────────────
def store_features(df: pd.DataFrame):
    """Push the feature row into the Hopsworks Feature Store."""
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY
    )
    fs = project.get_feature_store()

    # Get or create the feature group
    # version=1 means it creates it on first run, reuses it after
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="Hourly AQI and weather features for Karachi",
        online_enabled=True,   # allows real-time serving later
    )

    print(f"📦 Inserting {len(df)} row(s) into feature group 'aqi_features'...")
    fg.insert(df, write_options={"wait_for_job": False})
    print("✅ Features stored successfully!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_pipeline():
    print(f"\n{'='*50}")
    print(f"  AQI Feature Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    print(f"📡 Fetching AQI data for '{CITY}'...")
    raw = fetch_aqi_data(CITY, AQICN_TOKEN)
    print(f"   Current AQI: {raw.get('aqi')} | Station: {raw.get('city', {}).get('name', '?')}")

    print("\n⚙️  Engineering features...")
    df = engineer_features(raw)
    print(df.T.to_string())   # pretty-print the single row

    store_features(df)

    print("\n🎉 Pipeline run complete!\n")


if __name__ == "__main__":
    run_pipeline()
