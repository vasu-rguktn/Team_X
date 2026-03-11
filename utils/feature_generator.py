import pandas as pd


PRIMARY_USE_MAP = {
    "Education": 0,
    "Office": 1,
    "Lodging": 2,
    "Retail": 3
}

CITY_MAP = {
    "Hyderabad": 0,
    "Delhi": 1,
    "Mumbai": 2,
    "Chennai": 3
}

METER_MAP = {
    "Electricity": 0,
    "Chilled Water": 1,
    "Steam": 2,
    "Hot Water": 3
}


def generate_single_features(building_type, area, city, date, hour, meter_type):

    data = {
        "meter": METER_MAP[meter_type],
        "site_id": CITY_MAP[city],
        "primary_use": PRIMARY_USE_MAP[building_type],
        "square_feet": area,

        "air_temperature": 30,
        "cloud_coverage": 3,
        "dew_temperature": 20,
        "precip_depth_1_hr": 0,
        "sea_level_pressure": 1012,
        "wind_direction": 180,   # added missing feature
        "wind_speed": 2,

        "hour": hour,
        "month": date.month,
        "weekday": date.weekday(),
        "is_weekend": 1 if date.weekday() >= 5 else 0
    }

    df = pd.DataFrame([data])

    # enforce exact feature order expected by model
    feature_order = [
        "meter",
        "site_id",
        "primary_use",
        "square_feet",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "hour",
        "month",
        "weekday",
        "is_weekend"
    ]

    df = df[feature_order]

    return df