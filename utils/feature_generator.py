# utils/feature_generator.py
# ─────────────────────────────────────────────────────────────
# Converts user inputs into model-ready features
# Geolocation : Nominatim (free, no API key)
# Weather     : Open-Meteo Forecast + Archive (free, no API key)
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import calendar
import requests
import sys
import os
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(__file__))
from feature_labels import PRIMARY_USE_MAP, METER_MAP

# ── Feature Order ─────────────────────────────────────────────
# Must match exact order model was trained on

FEATURE_ORDER = [
    "meter", "site_id", "primary_use", "square_feet",
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction",
    "wind_speed", "hour", "month", "weekday", "is_weekend"
]

# ── Seasonal Weather Fallback ─────────────────────────────────
# Used when API call fails — based on general climate patterns

def get_seasonal_weather(month):
    """
    Returns seasonal weather defaults based on month.
    Used as fallback when API is unavailable.
    """
    if month in [3, 4, 5]:       # Spring/Summer
        return {
            "air_temperature":    35.0,
            "dew_temperature":    20.0,
            "wind_speed":         12.0,
            "wind_direction":     180.0,
            "sea_level_pressure": 1008.0,
            "cloud_coverage":     3.0,
            "precip_depth_1_hr":  0.0
        }
    elif month in [6, 7, 8]:     # Monsoon/Summer
        return {
            "air_temperature":    28.0,
            "dew_temperature":    25.0,
            "wind_speed":         18.0,
            "wind_direction":     200.0,
            "sea_level_pressure": 1005.0,
            "cloud_coverage":     7.0,
            "precip_depth_1_hr":  5.0
        }
    elif month in [9, 10, 11]:   # Autumn/Post-monsoon
        return {
            "air_temperature":    28.0,
            "dew_temperature":    18.0,
            "wind_speed":         10.0,
            "wind_direction":     160.0,
            "sea_level_pressure": 1012.0,
            "cloud_coverage":     3.0,
            "precip_depth_1_hr":  0.0
        }
    else:                        # Winter
        return {
            "air_temperature":    18.0,
            "dew_temperature":    10.0,
            "wind_speed":         8.0,
            "wind_direction":     90.0,
            "sea_level_pressure": 1018.0,
            "cloud_coverage":     2.0,
            "precip_depth_1_hr":  0.0
        }

# ── Geolocation API ───────────────────────────────────────────
# Nominatim: city name → lat/lon (free, no API key)

def get_coordinates(city_name):
    """
    Converts any city name to latitude and longitude.
    Works for any city on Earth.
    Returns: (lat, lon, display_name) or None if not found.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q":      city_name,
            "format": "json",
            "limit":  1
        }
        headers = {"User-Agent": "BuildingEnergyPredictor/1.0"}
        response = requests.get(
            url, params=params,
            headers=headers, timeout=5
        )
        data = response.json()

        if data:
            lat  = float(data[0]["lat"])
            lon  = float(data[0]["lon"])
            name = data[0]["display_name"]
            return lat, lon, name

        return None

    except Exception:
        return None


def get_city_suggestions(query):
    """
    Returns list of city name suggestions for autocomplete.
    Used in Streamlit UI as user types.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q":           query,
            "format":      "json",
            "limit":       5,
            "featuretype": "city"
        }
        headers = {"User-Agent": "BuildingEnergyPredictor/1.0"}
        response = requests.get(
            url, params=params,
            headers=headers, timeout=5
        )
        data = response.json()
        return [item["display_name"] for item in data]

    except Exception:
        return []


# ── Site ID Assignment ────────────────────────────────────────
# ASHRAE has 16 sites (0-15)
# We assign deterministically from coordinates

def assign_site_id(lat, lon):
    """
    Assigns a consistent site_id (0-15) from coordinates.
    Same city always gets same site_id.
    """
    rounded = f"{round(lat, 1)}_{round(lon, 1)}"
    return abs(hash(rounded)) % 16


# ── Time Feature Extractor ────────────────────────────────────

def extract_time_features(d, hour):
    """
    Extracts month, weekday, is_weekend from a date object.
    """
    month      = d.month
    weekday    = d.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    return month, weekday, is_weekend


# ── Weather — Forecast API (Current / Near Future) ────────────
# Open-Meteo forecast: works for today + next 16 days

def get_current_weather(lat, lon, target_date, hour):
    """
    Fetches real weather for a specific date and hour.
    Uses Open-Meteo forecast API.
    Works for current date and next 16 days.
    Falls back to seasonal defaults if API fails.
    """
    try:
        date_str = target_date.strftime("%Y-%m-%d")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "hourly": [
                "temperature_2m",
                "dewpoint_2m",
                "precipitation",
                "cloudcover",
                "surface_pressure",
                "windspeed_10m",
                "winddirection_10m"
            ],
            "start_date": date_str,
            "end_date":   date_str,
            "timezone":   "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if "hourly" not in data:
            raise ValueError("No hourly data returned")

        h = data["hourly"]

        return {
            "air_temperature":    h["temperature_2m"][hour],
            "dew_temperature":    h["dewpoint_2m"][hour],
            "precip_depth_1_hr":  h["precipitation"][hour],
            "cloud_coverage":     h["cloudcover"][hour] / 10,
            "sea_level_pressure": h["surface_pressure"][hour],
            "wind_speed":         h["windspeed_10m"][hour],
            "wind_direction":     h["winddirection_10m"][hour]
        }

    except Exception:
        return get_seasonal_weather(target_date.month)


# ── Weather — Archive API (Monthly Averages) ──────────────────
# Open-Meteo archive: works for any city, any month, any year

def get_monthly_weather_averages(lat, lon, year, month):
    """
    Fetches real monthly weather averages for any location.
    Uses previous year's same month as climate reference.
    Works for any city on Earth — no hardcoding.
    Falls back to seasonal defaults if API fails.
    """
    try:
        # Use previous year for historical reference
        ref_year = year - 1 if year >= 2024 else year
        num_days = calendar.monthrange(ref_year, month)[1]

        start_date = f"{ref_year}-{month:02d}-01"
        end_date   = f"{ref_year}-{month:02d}-{num_days:02d}"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly": [
                "temperature_2m",
                "dewpoint_2m",
                "precipitation",
                "cloudcover",
                "surface_pressure",
                "windspeed_10m",
                "winddirection_10m"
            ],
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=15)
        data     = response.json()

        if "hourly" not in data:
            raise ValueError("No hourly data returned")

        h = data["hourly"]

        def safe_avg(lst):
            vals = [v for v in lst if v is not None]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "air_temperature":    safe_avg(h["temperature_2m"]),
            "dew_temperature":    safe_avg(h["dewpoint_2m"]),
            "precip_depth_1_hr":  safe_avg(h["precipitation"]),
            "cloud_coverage":     safe_avg(h["cloudcover"]) / 10,
            "sea_level_pressure": safe_avg(h["surface_pressure"]),
            "wind_speed":         safe_avg(h["windspeed_10m"]),
            "wind_direction":     safe_avg(h["winddirection_10m"])
        }

    except Exception:
        return get_seasonal_weather(month)


# ── Single Hour Prediction Features ──────────────────────────

def generate_single_features(
    building_type, area, city_name,
    target_date, hour, meter_type
):
    """
    Generates a single row of features for one hour prediction.
    Uses Open-Meteo forecast for real current weather.

    Args:
        building_type : str   — e.g. "Office"
        area          : float — square footage
        city_name     : str   — any city on Earth
        target_date   : date  — prediction date
        hour          : int   — 0 to 23
        meter_type    : str   — e.g. "Electricity"

    Returns:
        pd.DataFrame with 1 row and 15 features
    """
    # Encode categorical inputs
    primary_use = PRIMARY_USE_MAP[building_type]
    meter       = METER_MAP[meter_type]

    # Get coordinates from city name
    coords = get_coordinates(city_name)
    if coords:
        lat, lon, _ = coords
    else:
        lat, lon = 20.5937, 78.9629  # India center fallback

    # Assign site_id from coordinates
    site_id = assign_site_id(lat, lon)

    # Fetch real weather via forecast API
    weather = get_current_weather(lat, lon, target_date, hour)

    # Extract time features
    month, weekday, is_weekend = extract_time_features(
        target_date, hour
    )

    row = {
        "meter":              meter,
        "site_id":            site_id,
        "primary_use":        primary_use,
        "square_feet":        area,
        "air_temperature":    weather["air_temperature"],
        "cloud_coverage":     weather["cloud_coverage"],
        "dew_temperature":    weather["dew_temperature"],
        "precip_depth_1_hr":  weather["precip_depth_1_hr"],
        "sea_level_pressure": weather["sea_level_pressure"],
        "wind_direction":     weather["wind_direction"],
        "wind_speed":         weather["wind_speed"],
        "hour":               hour,
        "month":              month,
        "weekday":            weekday,
        "is_weekend":         is_weekend
    }

    return pd.DataFrame([row])[FEATURE_ORDER]


# ── Monthly Feature Generator ─────────────────────────────────

def generate_monthly_features(
    building_type, area, city_name,
    year, month, meter_type
):
    """
    Generates 720 rows (30 days x 24 hours) for monthly prediction.
    Uses Open-Meteo archive for real monthly climate averages.
    Vectorized — model predicts all rows in one batch call.

    Returns:
        pd.DataFrame with 720 rows, 15 model features + tracking cols
    """
    primary_use = PRIMARY_USE_MAP[building_type]
    meter       = METER_MAP[meter_type]

    # Get coordinates
    coords = get_coordinates(city_name)
    if coords:
        lat, lon, _ = coords
    else:
        lat, lon = 20.5937, 78.9629

    site_id = assign_site_id(lat, lon)

    # Fetch real monthly climate averages via archive API
    weather  = get_monthly_weather_averages(lat, lon, year, month)
    num_days = calendar.monthrange(year, month)[1]

    rows = []
    for day in range(1, num_days + 1):
        current_date = date(year, month, day)
        m, weekday, is_weekend = extract_time_features(
            current_date, 0
        )
        for hour in range(24):
            rows.append({
                # Tracking columns (not fed to model)
                "date":    current_date,
                "day":     day,
                "hour":    hour,
                # Model features
                "meter":              meter,
                "site_id":            site_id,
                "primary_use":        primary_use,
                "square_feet":        area,
                "air_temperature":    weather["air_temperature"],
                "cloud_coverage":     weather["cloud_coverage"],
                "dew_temperature":    weather["dew_temperature"],
                "precip_depth_1_hr":  weather["precip_depth_1_hr"],
                "sea_level_pressure": weather["sea_level_pressure"],
                "wind_direction":     weather["wind_direction"],
                "wind_speed":         weather["wind_speed"],
                "month":              m,
                "weekday":            weekday,
                "is_weekend":         is_weekend
            })

    return pd.DataFrame(rows)


# ── Monthly Prediction Runner ─────────────────────────────────

def predict_monthly(
    building_type, area, city_name,
    year, month, meter_type, model
):
    """
    Runs full monthly prediction for any city.

    Returns dict:
        monthly_total_kwh : float
        daily_breakdown   : DataFrame (day, total_kwh)
        peak_day          : Series
        peak_row          : DataFrame (1 row — for SHAP)
        hourly_data       : full DataFrame
        site_id           : int
        weather           : dict of weather values used
        city_coords       : (lat, lon)
    """
    # Generate all 720 feature rows
    df = generate_monthly_features(
        building_type, area, city_name,
        year, month, meter_type
    )

    # Batch predict all rows in one model call
    preds_log = model.predict(df[FEATURE_ORDER])
    preds_kwh = np.expm1(preds_log)

    df["predicted_kwh"] = preds_kwh

    # Daily aggregation using reshape trick
    num_days     = calendar.monthrange(year, month)[1]
    daily_energy = preds_kwh[:num_days * 24].reshape(
        num_days, 24
    ).sum(axis=1)

    daily = pd.DataFrame({
        "day":       range(1, num_days + 1),
        "total_kwh": daily_energy
    })

    monthly_total = daily["total_kwh"].sum()
    peak_day      = daily.loc[daily["total_kwh"].idxmax()]

    # Get peak hour row for SHAP analysis
    peak_day_num  = int(peak_day["day"])
    peak_day_rows = df[df["day"] == peak_day_num]
    peak_hour_idx = peak_day_rows["predicted_kwh"].idxmax()
    peak_row      = df.loc[[peak_hour_idx], FEATURE_ORDER]

    return {
        "monthly_total_kwh": monthly_total,
        "daily_breakdown":   daily,
        "peak_day":          peak_day,
        "peak_row":          peak_row,
        "hourly_data":       df,
        "site_id":           int(df["site_id"].iloc[0])
    }