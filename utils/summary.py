# utils/summary.py
# ─────────────────────────────────────────────────────────────
# Generates human readable energy summary paragraph
# Takes: prediction + building info + recommendations
# Returns: clean paragraph string
# ─────────────────────────────────────────────────────────────

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from feature_labels import (
    PRIMARY_USE_LABELS,
    METER_LABELS,
    SITE_LABELS,
    MONTH_LABELS
)

# ── Helper ────────────────────────────────────────────────────

def get_efficiency_status(total_saving_min, total_saving_max):
    """
    Returns efficiency label based on potential saving range.
    Higher saving potential = more inefficient building.
    """
    avg = (total_saving_min + total_saving_max) / 2

    if avg >= 25:
        return "highly inefficient", "🔴"
    elif avg >= 15:
        return "moderately inefficient", "🟡"
    elif avg >= 8:
        return "slightly inefficient", "🟠"
    else:
        return "reasonably efficient", "🟢"


def get_period_label(period, month=None, year=None):
    """
    Returns human readable period string.
    """
    if period == "monthly" and month and year:
        month_name = MONTH_LABELS.get(month, str(month))
        return f"in {month_name} {year}"
    elif period == "daily":
        return "today"
    elif period == "weekly":
        return "this week"
    else:
        return "for this period"


# ── Main Summary Generator ────────────────────────────────────

def generate_summary(
    prediction_kwh,
    building_type_code,
    area,
    site_id,
    meter_code,
    recommendations,
    period="monthly",
    month=None,
    year=None
):
    """
    Generates a complete human readable summary.

    Args:
        prediction_kwh      : float — predicted energy in kWh
        building_type_code  : int   — primary_use encoded value
        area                : float — square footage
        site_id             : int   — site encoded value
        meter_code          : int   — meter type encoded value
        recommendations     : list  — from recommendations.py
        period              : str   — "monthly", "daily", "weekly"
        month               : int   — month number (for monthly)
        year                : int   — year (for monthly)

    Returns:
        str — complete summary paragraph
    """

    # ── Decode building info ──────────────────────────────────
    building_type = PRIMARY_USE_LABELS.get(
        int(building_type_code), "Unknown"
    )
    meter_type = METER_LABELS.get(int(meter_code), "Unknown")
    site_label = SITE_LABELS.get(int(site_id), f"Site {site_id}")
    area_formatted = f"{int(area):,}"
    period_label = get_period_label(period, month, year)

    # ── Calculate saving potential ────────────────────────────
    if recommendations:
        total_min = sum(r['saving_min'] for r in recommendations)
        total_max = sum(r['saving_max'] for r in recommendations)
        avg_saving = (total_min + total_max) / 2
        max_saving_kwh = prediction_kwh * (total_max / 100)
        min_saving_kwh = prediction_kwh * (total_min / 100)
    else:
        total_min = total_max = avg_saving = 0
        max_saving_kwh = min_saving_kwh = 0

    # ── Efficiency status ─────────────────────────────────────
    status_label, status_emoji = get_efficiency_status(
        total_min, total_max
    )

    # ── Top drivers ───────────────────────────────────────────
    if recommendations:
        top_drivers = [
            r['feature_label'] for r in recommendations[:3]
        ]
        if len(top_drivers) == 1:
            drivers_text = top_drivers[0]
        elif len(top_drivers) == 2:
            drivers_text = f"{top_drivers[0]} and {top_drivers[1]}"
        else:
            drivers_text = (
                f"{top_drivers[0]}, {top_drivers[1]} "
                f"and {top_drivers[2]}"
            )
    else:
        drivers_text = "various building factors"

    # ── Build summary paragraph ───────────────────────────────
    lines = []

    # Opening — what and where
    lines.append(
        f"This {building_type} building ({area_formatted} sq ft) "
        f"at {site_label} is predicted to consume "
        f"{prediction_kwh:,.1f} kWh {period_label} "
        f"using the {meter_type} meter."
    )

    # Efficiency status
    lines.append(
        f"Based on SHAP analysis, the building is currently "
        f"{status_label} {status_emoji}, with the primary energy "
        f"drivers being {drivers_text}."
    )

    # Recommendations impact
    if recommendations:
        lines.append(
            f"By implementing the {len(recommendations)} recommended "
            f"actions, an estimated {total_min}-{total_max}% reduction "
            f"in energy consumption is achievable, potentially saving "
            f"between {min_saving_kwh:,.0f} and "
            f"{max_saving_kwh:,.0f} kWh {period_label}."
        )

    # Closing action line
    if avg_saving >= 15:
        lines.append(
            "Immediate action is recommended to bring this building's "
            "energy consumption within acceptable efficiency ranges."
        )
    elif avg_saving >= 8:
        lines.append(
            "Moderate improvements to building operations could "
            "significantly reduce energy costs and environmental impact."
        )
    else:
        lines.append(
            "This building is performing reasonably well. "
            "Minor optimizations can further improve efficiency."
        )

    return " ".join(lines)


# ── Formatted Version For Display ────────────────────────────

def generate_full_report(
    prediction_kwh,
    building_type_code,
    area,
    site_id,
    meter_code,
    recommendations,
    shap_values=None,
    feature_names=None,
    period="monthly",
    month=None,
    year=None
):
    """
    Returns complete report as a dictionary with all sections.
    Used by app.py to display each section separately.
    """

    summary = generate_summary(
        prediction_kwh,
        building_type_code,
        area,
        site_id,
        meter_code,
        recommendations,
        period, month, year
    )

    total_min = sum(r['saving_min'] for r in recommendations) if recommendations else 0
    total_max = sum(r['saving_max'] for r in recommendations) if recommendations else 0

    return {
        "summary_text":     summary,
        "prediction_kwh":   prediction_kwh,
        "building_type":    PRIMARY_USE_LABELS.get(int(building_type_code), "Unknown"),
        "meter_type":       METER_LABELS.get(int(meter_code), "Unknown"),
        "area":             area,
        "site":             SITE_LABELS.get(int(site_id), f"Site {site_id}"),
        "total_saving_min": total_min,
        "total_saving_max": total_max,
        "recommendations":  recommendations,
        "period":           period,
        "month":            MONTH_LABELS.get(month, "") if month else "",
        "year":             year or ""
    }