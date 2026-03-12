# utils/recommendations.py
# ─────────────────────────────────────────────────────────────
# Converts SHAP values into actionable energy recommendations
# ─────────────────────────────────────────────────────────────

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from feature_labels import get_feature_label, get_value_label

# ── Recommendation Knowledge Base ────────────────────────────
# Each feature maps to:
#   - advice: what to do
#   - saving_min: minimum % saving estimate
#   - saving_max: maximum % saving estimate
#   - condition: only show if SHAP is positive (increasing energy)

RECOMMENDATION_MAP = {
    "square_feet": {
        "advice": "Your building size is a major energy driver. "
                  "Consider implementing HVAC zoning to heat/cool "
                  "only occupied areas instead of the entire building.",
        "saving_min": 8,
        "saving_max": 15
    },
    "air_temperature": {
        "advice": "High outdoor temperature is significantly increasing "
                  "your cooling load. Improve building insulation and "
                  "optimize HVAC schedules during peak temperature hours.",
        "saving_min": 6,
        "saving_max": 12
    },
    "meter": {
        "advice": "Your meter type indicates high energy demand. "
                  "Review equipment schedules and consider switching "
                  "to more energy efficient systems where possible.",
        "saving_min": 5,
        "saving_max": 10
    },
    "primary_use": {
        "advice": "Buildings of this type typically benefit from "
                  "occupancy-based controls. Install smart sensors "
                  "to automatically reduce energy use in empty spaces.",
        "saving_min": 7,
        "saving_max": 13
    },
    "hour": {
        "advice": "Peak hour usage is driving up your consumption. "
                  "Shift non-critical operations like cleaning, "
                  "laundry or heavy equipment use to off-peak hours.",
        "saving_min": 5,
        "saving_max": 8
    },
    "month": {
        "advice": "Seasonal energy demand is high this month. "
                  "Pre-cool or pre-heat the building during off-peak "
                  "hours to reduce peak demand charges.",
        "saving_min": 4,
        "saving_max": 9
    },
    "dew_temperature": {
        "advice": "High humidity is increasing your cooling system "
                  "workload. Consider installing a dehumidification "
                  "system to reduce latent cooling load.",
        "saving_min": 4,
        "saving_max": 8
    },
    "wind_speed": {
        "advice": "Wind conditions are affecting your building envelope. "
                  "Check for air leaks around windows and doors and "
                  "improve weatherstripping to reduce infiltration losses.",
        "saving_min": 2,
        "saving_max": 6
    },
    "cloud_coverage": {
        "advice": "Current cloud conditions affect natural lighting. "
                  "Use daylight sensors to automatically adjust "
                  "artificial lighting levels and reduce electricity use.",
        "saving_min": 3,
        "saving_max": 7
    },
    "sea_level_pressure": {
        "advice": "Atmospheric pressure is influencing HVAC performance. "
                  "Ensure your system is calibrated for local conditions "
                  "to maintain optimal efficiency.",
        "saving_min": 2,
        "saving_max": 5
    },
    "wind_direction": {
        "advice": "Wind direction is impacting your building's thermal "
                  "performance. Consider windbreaks or facade shading "
                  "on the windward side to reduce heat gain or loss.",
        "saving_min": 2,
        "saving_max": 5
    },
    "precip_depth_1_hr": {
        "advice": "Rainfall patterns are affecting building conditions. "
                  "Ensure roof insulation and drainage systems are "
                  "maintained to prevent moisture-related energy losses.",
        "saving_min": 1,
        "saving_max": 4
    },
    "weekday": {
        "advice": "Day of week patterns show energy variation. "
                  "Create separate HVAC schedules for weekdays "
                  "and weekends to avoid unnecessary conditioning.",
        "saving_min": 3,
        "saving_max": 7
    },
    "is_weekend": {
        "advice": "Weekend energy usage can be significantly reduced. "
                  "Set up automatic setback temperatures and "
                  "lighting schedules for non-working days.",
        "saving_min": 5,
        "saving_max": 12
    },
    "site_id": {
        "advice": "Your location's climate characteristics are "
                  "influencing energy use. Consider local climate "
                  "adaptive strategies for your building envelope.",
        "saving_min": 2,
        "saving_max": 6
    }
}

# ── Core Functions ────────────────────────────────────────────

def get_top_shap_features(shap_values, feature_names, top_n=3):
    """
    Returns top N features sorted by absolute SHAP value.
    Only returns features with positive SHAP
    (features increasing energy consumption).
    These are the ones worth acting on.
    """
    pairs = list(zip(feature_names, shap_values))

    # Sort by absolute impact, highest first
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Only keep features that are INCREASING energy
    # (positive SHAP = pushing prediction higher)
    positive = [(f, s) for f, s in pairs if s > 0.01]

    return positive[:top_n]


def generate_recommendations(shap_values, feature_names, feature_values):
    """
    Main function. Takes SHAP values and returns
    list of recommendation dictionaries.

    Returns:
        List of dicts with keys:
        - feature_name  (raw)
        - feature_label (human readable)
        - feature_value (decoded)
        - shap_impact   (float)
        - advice        (string)
        - saving_min    (int %)
        - saving_max    (int %)
    """
    top_features = get_top_shap_features(shap_values, feature_names)

    recommendations = []

    for feature_name, shap_impact in top_features:

        if feature_name not in RECOMMENDATION_MAP:
            continue

        rec_data = RECOMMENDATION_MAP[feature_name]

        # Get feature value from input
        if feature_name in feature_names:
            idx = list(feature_names).index(feature_name)
            raw_value = feature_values[idx]
        else:
            raw_value = None

        recommendations.append({
            "feature_name":  feature_name,
            "feature_label": get_feature_label(feature_name),
            "feature_value": get_value_label(feature_name, raw_value),
            "shap_impact":   round(shap_impact, 3),
            "advice":        rec_data["advice"],
            "saving_min":    rec_data["saving_min"],
            "saving_max":    rec_data["saving_max"]
        })

    return recommendations


def format_recommendations(recommendations):
    """
    Returns a clean printable string of recommendations.
    Used in notebook testing.
    """
    if not recommendations:
        return "No significant recommendations found."

    lines = ["ENERGY SAVING RECOMMENDATIONS", "=" * 40]

    for i, rec in enumerate(recommendations, 1):
        lines.append(
            f"\n{i}. {rec['feature_label']}: {rec['feature_value']}"
        )
        lines.append(f"   Impact: +{rec['shap_impact']:.2f} on log scale")
        lines.append(f"   Action: {rec['advice']}")
        lines.append(
            f"   Estimated Saving: {rec['saving_min']}-"
            f"{rec['saving_max']}%"
        )

    total_min = sum(r['saving_min'] for r in recommendations)
    total_max = sum(r['saving_max'] for r in recommendations)
    lines.append("\n" + "=" * 40)
    lines.append(
        f"Total Potential Saving: {total_min}-{total_max}%"
    )

    return "\n".join(lines)