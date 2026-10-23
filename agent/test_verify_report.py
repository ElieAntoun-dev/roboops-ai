
from agent.schemas import IncidentReport, IncidentMeasurements
from agent.verify_report import verify_incident_report
from agent.tools import analyze_wheel_slip, analyze_battery_discharge


# ============================================================
# 1. CREATE A VALID UNEXPECTED STOP REPORT (INCIDENT 6)
# ============================================================

report = IncidentReport(
    incident_id=6,
    robot_id="AMR-02",
    incident_type="unexpected_stop",
    start_time="2026-10-18T12:37:38.325324",
    end_time="2026-10-18T12:37:46.325324",

    summary="Unexpected stop investigation.",

    observations=[
        "Velocity remained at 0 m/s."
    ],

    # Numerical measurements from incident 6
    measurements=IncidentMeasurements(
        values={
            "average_velocity": 0.0,
            "average_motor_current": 6.55,
            "maximum_motor_current": 14.09,
            "maximum_observed_deceleration": 0.0
        }
    ),

    technical_assessment="Root cause is undetermined.",

    possible_causes=[],

    root_cause_confirmed=False,
    root_cause=None,

    recommended_actions=[
        "Review robot event logs."
    ],

    investigation_status="inconclusive"
)


# ============================================================
# 2. TEST A CORRECT UNEXPECTED STOP REPORT
# ============================================================

try:
    verify_incident_report(report)

    print("PASSED: Correct unexpected-stop report matches source data.")

except ValueError as error:
    print(f"FAILED: {error}")


# ============================================================
# 3. TEST AN INCORRECT ROBOT ID
# ============================================================

# Create a copy of the valid report
incorrect_data = report.model_dump()

# Deliberately change the robot ID
incorrect_data["robot_id"] = "AMR-99"

incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Incorrect robot ID was accepted.")

except ValueError as error:
    print(f"PASSED: Incorrect robot ID rejected. {error}")


# ============================================================
# 4. TEST AN INCORRECT END TIMESTAMP
# ============================================================

incorrect_data = report.model_dump()

# Chronologically valid but factually incorrect timestamp
incorrect_data["end_time"] = "2026-10-18T12:37:47.325324"

incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Incorrect end timestamp was accepted.")

except ValueError as error:
    print(f"PASSED: Incorrect end timestamp rejected. {error}")


# ============================================================
# 5. TEST AN INCORRECT UNEXPECTED STOP MEASUREMENT
# ============================================================

incorrect_data = report.model_dump()

# Correct average motor current: 6.55 A
# Deliberately incorrect value: 20.0 A
incorrect_data["measurements"]["values"]["average_motor_current"] = 20.0

incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Incorrect unexpected-stop measurement was accepted.")

except ValueError as error:
    print(f"PASSED: Incorrect unexpected-stop measurement rejected. {error}")


# ============================================================
# 6. CREATE A VALID WHEEL SLIP REPORT (INCIDENT 1)
# ============================================================

# Calculate measurements directly from telemetry
wheel_slip_analysis = analyze_wheel_slip.invoke(
    {"incident_id": 1}
)

wheel_slip_report = IncidentReport(
    incident_id=1,
    robot_id="AMR-01",
    incident_type="wheel_slip",
    start_time="2026-10-18T12:36:33.325324",
    end_time="2026-10-18T12:36:45.325324",

    summary="Wheel-slip incident investigation.",

    observations=[
        "A difference between left and right wheel speeds was detected."
    ],

    measurements=IncidentMeasurements(
        values={
            "maximum_wheel_difference": wheel_slip_analysis[
                "maximum_wheel_speed_difference_mps"
            ],

            "average_wheel_difference": wheel_slip_analysis[
                "average_wheel_speed_difference_mps"
            ],

            "average_velocity": wheel_slip_analysis[
                "average_velocity_mps"
            ],

            "average_motor_current": wheel_slip_analysis[
                "average_motor_current_A"
            ]
        }
    ),

    technical_assessment=(
        "Wheel-speed differences were observed. "
        "The underlying cause has not been confirmed."
    ),

    possible_causes=[
        "Loss of wheel traction",
        "Wheel-speed sensor inconsistency"
    ],

    root_cause_confirmed=False,
    root_cause=None,

    recommended_actions=[
        "Inspect the drive wheels.",
        "Review wheel-speed sensor measurements."
    ],

    investigation_status="inconclusive"
)


# ============================================================
# 7. TEST A CORRECT WHEEL SLIP REPORT
# ============================================================

try:
    verify_incident_report(wheel_slip_report)

    print("PASSED: Correct wheel-slip report matches source data.")

except ValueError as error:
    print(f"FAILED: {error}")


# ============================================================
# 8. TEST AN INCORRECT WHEEL SLIP MEASUREMENT
# ============================================================

incorrect_data = wheel_slip_report.model_dump()

# Deliberately change the maximum wheel-speed difference
incorrect_data["measurements"]["values"]["maximum_wheel_difference"] = 99.0

incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Incorrect wheel-slip measurement was accepted.")

except ValueError as error:
    print(f"PASSED: Incorrect wheel-slip measurement rejected. {error}")


# ============================================================
# 9. CREATE A VALID BATTERY DISCHARGE REPORT (INCIDENT 5)
# ============================================================

# Calculate battery measurements directly from telemetry
battery_analysis = analyze_battery_discharge.invoke(
    {"incident_id": 5}
)

battery_report = IncidentReport(
    incident_id=5,
    robot_id="AMR-02",
    incident_type="battery_discharge",
    start_time="2026-10-18T12:35:35.325324",
    end_time="2026-10-18T12:35:52.325324",

    summary="Battery-discharge incident investigation.",

    observations=[
        "The battery level decreased during the incident window."
    ],

    # Map the Python tool results to the report schema
    measurements=IncidentMeasurements(
        values={
            "initial_battery_level": battery_analysis[
                "initial_battery_percent"
            ],

            "final_battery_level": battery_analysis[
                "final_battery_percent"
            ],

            "total_battery_drop": battery_analysis[
                "total_battery_drop_percentage_points"
            ],

            "average_discharge_rate": battery_analysis[
                "average_discharge_rate_percentage_points_per_second"
            ],

            "average_velocity": battery_analysis[
                "average_velocity_mps"
            ],

            "average_motor_current": battery_analysis[
                "average_motor_current_A"
            ]
        }
    ),

    technical_assessment=(
        "Battery discharge was observed during the incident. "
        "The underlying cause has not been confirmed."
    ),

    possible_causes=[
        "High electrical power consumption",
        "Battery measurement inconsistency"
    ],

    root_cause_confirmed=False,
    root_cause=None,

    recommended_actions=[
        "Review battery voltage and current measurements.",
        "Inspect the battery management system logs."
    ],

    investigation_status="inconclusive"
)


# ============================================================
# 10. TEST A CORRECT BATTERY DISCHARGE REPORT
# ============================================================

try:
    verify_incident_report(battery_report)

    print("PASSED: Correct battery-discharge report matches source data.")

except ValueError as error:
    print(f"FAILED: {error}")


# ============================================================
# 11. TEST AN INCORRECT BATTERY DISCHARGE MEASUREMENT
# ============================================================

# Create a copy of the valid battery-discharge report
incorrect_data = battery_report.model_dump()

# Deliberately change the discharge rate to an incorrect value
incorrect_data["measurements"]["values"]["average_discharge_rate"] = 99.0

incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Incorrect battery-discharge measurement was accepted.")

except ValueError as error:
    print(f"PASSED: Incorrect battery-discharge measurement rejected. {error}")



# ============================================================
# 12. TEST AN INVALID INCIDENT ID
# ============================================================

# Create a copy of the valid unexpected-stop report
incorrect_data = report.model_dump()

# Deliberately use an incident ID that does not exist
incorrect_data["incident_id"] = 9999

# Validate the modified report using Pydantic
incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    # Attempt to verify the nonexistent incident
    verify_incident_report(incorrect_report)

    # The test fails if the report is accepted
    print("FAILED: Nonexistent incident ID was accepted.")

except ValueError as error:
    # Expected behavior: verification rejects the incident
    print(f"PASSED: Nonexistent incident ID rejected. {error}")



# ============================================================
# 13. TEST A MISSING NUMERICAL MEASUREMENT
# ============================================================

# Create a copy of the valid wheel-slip report
incorrect_data = wheel_slip_report.model_dump()

# Deliberately remove a required measurement
del incorrect_data["measurements"]["values"]["average_motor_current"]

# Validate the modified report using Pydantic
incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Report with missing measurement was accepted.")

except ValueError as error:
    print(f"PASSED: Missing measurement rejected. {error}")



# ============================================================
# 14. TEST AN UNEXPECTED NUMERICAL MEASUREMENT
# ============================================================

# Create a copy of the valid battery-discharge report
incorrect_data = battery_report.model_dump()

# Add a numerical measurement that is not expected
# for a battery-discharge incident
incorrect_data["measurements"]["values"]["maximum_wheel_difference"] = 1.5

# Validate the modified report using Pydantic
incorrect_report = IncidentReport.model_validate(incorrect_data)

try:
    verify_incident_report(incorrect_report)

    print("FAILED: Unexpected measurement was accepted.")

except ValueError as error:
    print(f"PASSED: Unexpected measurement rejected. {error}")

