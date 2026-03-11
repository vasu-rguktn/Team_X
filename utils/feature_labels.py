FEATURE_LABELS = {

    "primary_use": "Building Type",
    "square_feet": "Building Area (sq ft)",
    "site_id": "City",
    "meter": "Energy Meter Type",

    "air_temperature": "Outdoor Temperature (°C)",
    "dew_temperature": "Dew Point Temperature (°C)",
    "cloud_coverage": "Cloud Coverage",
    "precip_depth_1_hr": "Rainfall (mm/hr)",
    "wind_speed": "Wind Speed (m/s)",
    "sea_level_pressure": "Sea Level Pressure (hPa)",

    "hour": "Hour of Day",
    "day": "Day of Month",
    "month": "Month",
    "weekday": "Day of Week",

    "is_weekend": "Weekend Indicator"
}


def get_feature_label(name):
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def get_value_label(feature, value):
    return value