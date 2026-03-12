# utils/feature_labels.py
# Single source of truth — built from actual ASHRAE training data

# ── Human Readable Feature Names ─────────────────────────────

FEATURE_LABELS = {
    "primary_use":        "Building Type",
    "square_feet":        "Building Area (sq ft)",
    "site_id":            "Site Location",
    "meter":              "Energy Meter Type",
    "air_temperature":    "Outdoor Temperature (C)",
    "dew_temperature":    "Dew Point Temperature (C)",
    "cloud_coverage":     "Cloud Coverage",
    "precip_depth_1_hr":  "Rainfall (mm/hr)",
    "wind_speed":         "Wind Speed (m/s)",
    "wind_direction":     "Wind Direction",
    "sea_level_pressure": "Sea Level Pressure (hPa)",
    "hour":               "Hour of Day",
    "day":                "Day of Month",
    "month":              "Month",
    "weekday":            "Day of Week",
    "is_weekend":         "Weekend Indicator"
}

# ── Value Decoders ────────────────────────────────────────────
# Built from actual df['column'].value_counts() output

METER_LABELS = {
    0: "Electricity",
    1: "Chilled Water",
    2: "Steam"
}

PRIMARY_USE_LABELS = {
    0:  "Education",
    1:  "Entertainment/Public Assembly",
    2:  "Food Sales and Service",
    3:  "Healthcare",
    4:  "Lodging/Residential",
    5:  "Manufacturing/Industrial",
    6:  "Office",
    7:  "Other",
    8:  "Parking",
    9:  "Public Services",
    10: "Religious Worship",
    11: "Retail",
    12: "Services",
    13: "Technology/Science",
    14: "Utility",
    15: "Warehouse/Storage"
}

SITE_LABELS = {
    0:  "Site 0",
    1:  "Site 1",
    2:  "Site 2",
    3:  "Site 3",
    4:  "Site 4",
    5:  "Site 5",
    6:  "Site 6",
    7:  "Site 7",
    8:  "Site 8",
    9:  "Site 9",
    10: "Site 10",
    11: "Site 11",
    12: "Site 12",
    13: "Site 13",
    14: "Site 14",
    15: "Site 15"
}

MONTH_LABELS = {
    1:  "January",   2:  "February",  3:  "March",
    4:  "April",     5:  "May",       6:  "June",
    7:  "July",      8:  "August",    9:  "September",
    10: "October",   11: "November",  12: "December"
}

WEEKDAY_LABELS = {
    0: "Monday",    1: "Tuesday",   2: "Wednesday",
    3: "Thursday",  4: "Friday",    5: "Saturday",
    6: "Sunday"
}

WEEKEND_LABELS = {
    0: "No",
    1: "Yes"
}

VALUE_DECODERS = {
    "meter":       METER_LABELS,
    "primary_use": PRIMARY_USE_LABELS,
    "site_id":     SITE_LABELS,
    "month":       MONTH_LABELS,
    "weekday":     WEEKDAY_LABELS,
    "is_weekend":  WEEKEND_LABELS
}

# ── Also expose maps for feature_generator.py ─────────────────
# These are the REVERSE maps — label to number
# Used when converting user input to model input

PRIMARY_USE_MAP = {v: k for k, v in PRIMARY_USE_LABELS.items()}
METER_MAP = {
    "Electricity":   0,
    "Chilled Water": 1,
    "Steam":         2
}

# ── Helper Functions ──────────────────────────────────────────

def get_feature_label(name):
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def get_value_label(feature_name, value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)

    if feature_name in VALUE_DECODERS:
        decoded = VALUE_DECODERS[feature_name].get(int(value))
        return decoded if decoded else str(int(value))

    if feature_name == "square_feet":
        return f"{int(value):,}"

    if feature_name in ["air_temperature", "dew_temperature"]:
        return f"{value:.1f} C"

    if feature_name == "wind_speed":
        return f"{value:.1f} m/s"

    if feature_name == "sea_level_pressure":
        return f"{value:.1f} hPa"

    if feature_name == "precip_depth_1_hr":
        return f"{value:.1f} mm"

    if feature_name == "hour":
        return f"{int(value):02d}:00"

    if feature_name == "wind_direction":
        return f"{int(value)} deg"

    return str(int(value)) if value == int(value) else str(round(value, 2))