
from datetime import datetime
from pathlib import Path
from math import isclose, isfinite

import pandas as pd

from agent.schemas import IncidentReport

# Import existing numerical analysis tools
from agent.tools import (
    analyze_battery_discharge,
    analyze_wheel_slip,
    analyze_unexpected_stop
)


# ============================================================
# 1. DEFINE THE INCIDENT CSV PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INCIDENTS_CSV = (
    PROJECT_ROOT / "data" / "classified_incidents_test_filtered.csv"
)


# ============================================================
# 2. DEFINE MEASUREMENT MAPPINGS
# ============================================================

# Map report measurement names to analysis-tool output names
MEASUREMENT_MAPPINGS = {

    "battery_discharge": {
        "initial_battery_level": "initial_battery_percent",
        "final_battery_level": "final_battery_percent",
        "total_battery_drop": "total_battery_drop_percentage_points",
        "average_discharge_rate": "average_discharge_rate_percentage_points_per_second",
        "average_velocity": "average_velocity_mps",
        "average_motor_current": "average_motor_current_A"
    },

    "wheel_slip": {
        "maximum_wheel_difference": "maximum_wheel_speed_difference_mps",
        "average_wheel_difference": "average_wheel_speed_difference_mps",
        "average_velocity": "average_velocity_mps",
        "average_motor_current": "average_motor_current_A"
    },

    "unexpected_stop": {
        "maximum_observed_deceleration": "maximum_observed_deceleration_mps2",
        "average_velocity": "average_velocity_mps",
        "average_motor_current": "average_motor_current_A",
        "maximum_motor_current": "maximum_motor_current_A"
    }
}


# Match each incident type to its analysis tool
ANALYSIS_TOOLS = {
    "battery_discharge": analyze_battery_discharge,
    "wheel_slip": analyze_wheel_slip,
    "unexpected_stop": analyze_unexpected_stop
}


# ============================================================
# 3. VERIFY THE STRUCTURED INCIDENT REPORT
# ============================================================

def verify_incident_report(report: IncidentReport) -> bool:

    # Load the original classified incidents
    incidents = pd.read_csv(INCIDENTS_CSV)

    # Find the incident matching the report's incident ID
    matching_incidents = incidents[
        incidents["incident_id"] == report.incident_id
    ]

    # Reject unknown or duplicate incident IDs
    if len(matching_incidents) != 1:
        raise ValueError(
            f"Expected exactly one source record for incident "
            f"{report.incident_id}, found {len(matching_incidents)}."
        )

    # Retrieve the original incident information
    source = matching_incidents.iloc[0]

    # ========================================================
    # 4. VERIFY INCIDENT IDENTIFICATION
    # ========================================================

    expected_values = {
        "robot_id": source["robot_id"],
        "incident_type": source["predicted_incident_type"]
    }

    for field, expected in expected_values.items():

        actual = getattr(report, field)

        if actual != expected:
            raise ValueError(
                f"Incident {report.incident_id}: "
                f"{field} mismatch. "
                f"Report={actual!r}, CSV={expected!r}"
            )

    # ========================================================
    # 5. VERIFY INCIDENT TIMESTAMPS
    # ========================================================

    report_start = datetime.fromisoformat(report.start_time)
    report_end = datetime.fromisoformat(report.end_time)

    source_start = pd.to_datetime(
        source["start_time"]
    ).to_pydatetime()

    source_end = pd.to_datetime(
        source["end_time"]
    ).to_pydatetime()

    if report_start != source_start:
        raise ValueError(
            f"Incident {report.incident_id}: "
            f"start_time mismatch. "
            f"Report={report.start_time}, "
            f"CSV={source['start_time']}"
        )

    if report_end != source_end:
        raise ValueError(
            f"Incident {report.incident_id}: "
            f"end_time mismatch. "
            f"Report={report.end_time}, "
            f"CSV={source['end_time']}"
        )

    # ========================================================
    # 6. RETRIEVE EXPECTED NUMERICAL MEASUREMENTS
    # ========================================================

    incident_type = report.incident_type

    # Ensure this incident type has a verification method
    if incident_type not in ANALYSIS_TOOLS:
        raise ValueError(
            f"Numerical verification is not supported "
            f"for incident type '{incident_type}'."
        )

    # Select the appropriate analysis tool
    analysis_tool = ANALYSIS_TOOLS[incident_type]

    # Recalculate expected values using the original telemetry
    analysis = analysis_tool.invoke({
        "incident_id": report.incident_id
    })

    # Reject reports when analysis cannot be completed
    if "error" in analysis:
        raise ValueError(
            f"Incident {report.incident_id}: "
            f"Numerical analysis failed: {analysis['error']}"
        )

    # ========================================================
    # 7. VERIFY NUMERICAL MEASUREMENTS
    # ========================================================

    # Get the required measurements for this incident type
    mapping = MEASUREMENT_MAPPINGS[incident_type]

    # Get measurements generated by Gemini
    reported_values = report.measurements.values

    # Check that all required measurements are present
    missing = set(mapping) - set(reported_values)

    if missing:
        raise ValueError(
            f"Incident {report.incident_id}: "
            f"Missing measurements: {sorted(missing)}"
        )

    # Reject unsupported measurements for this incident type
    unexpected = set(reported_values) - set(mapping)

    if unexpected:
        raise ValueError(
            f"Incident {report.incident_id}: "
            f"Unexpected measurements: {sorted(unexpected)}"
        )

    # Compare every reported value with its tool-calculated value
    for report_field, tool_field in mapping.items():

        expected = analysis.get(tool_field)
        actual = reported_values[report_field]

        # Reject unavailable measurements
        if expected is None:
            raise ValueError(
                f"Incident {report.incident_id}: "
                f"Expected measurement '{tool_field}' is unavailable."
            )

        # Reject non-finite values such as NaN or infinity
        if not isfinite(float(expected)) or not isfinite(float(actual)):
            raise ValueError(
                f"Incident {report.incident_id}: "
                f"Invalid numerical value for '{report_field}'."
            )

        # Allow small rounding differences
        if not isclose(
            float(actual),
            float(expected),
            rel_tol=0,
            abs_tol=0.0001
        ):
            raise ValueError(
                f"Incident {report.incident_id}: "
                f"{report_field} mismatch. "
                f"Report={actual}, Expected={expected}"
            )

    # ========================================================
    # 8. RETURN VERIFICATION RESULT
    # ========================================================

    # All checked identification, timestamp and numerical fields match
    return True
