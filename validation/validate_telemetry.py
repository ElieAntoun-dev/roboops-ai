
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD GENERATED DATASETS
# ============================================================

telemetry_df = pd.read_csv("data/robot_telemetry.csv")
incident_df = pd.read_csv("data/incident_log.csv")


# ============================================================
# 2. VALIDATE DATASET STRUCTURE
# ============================================================

expected_columns = [
    "timestamp",
    "robot_id",
    "velocity",
    "left_wheel_speed",
    "right_wheel_speed",
    "motor_current",
    "battery_level",
    "status"
]

# Check whether all required columns exist
missing_columns = [
    column
    for column in expected_columns
    if column not in telemetry_df.columns
]

# Check expected number of records
expected_records = 5 * 1000

print("\n--- TELEMETRY VALIDATION ---")

print(f"Expected records: {expected_records}")
print(f"Actual records: {len(telemetry_df)}")

print(f"Missing columns: {missing_columns}")

if len(telemetry_df) == expected_records and not missing_columns:
    print("PASS: Dataset structure is valid.")
else:
    print("FAIL: Dataset structure needs attention.")

print(f"\nTotal incidents in ground-truth log: {len(incident_df)}")

# ============================================================
# 3. VALIDATE TELEMETRY VALUES
# ============================================================

# Count missing values in each column
missing_values = telemetry_df.isnull().sum()

# Check battery levels are between 0% and 100%
invalid_battery = ~telemetry_df["battery_level"].between(0, 100)

print("\n--- TELEMETRY VALUE VALIDATION ---")

print(f"Total missing values: {missing_values.sum()}")
print(f"Invalid battery records: {invalid_battery.sum()}")

if missing_values.sum() == 0 and invalid_battery.sum() == 0:
    print("PASS: Telemetry values are valid.")
else:
    print("FAIL: Invalid telemetry values detected.")

# ============================================================
# 4. VALIDATE INCIDENTS AGAINST TELEMETRY
# ============================================================

print("\n--- INCIDENT VALIDATION ---")

failed_incidents = 0

for _, incident in incident_df.iterrows():

    robot_id = incident["robot_id"]
    incident_type = incident["incident_type"]
    start = int(incident["start_second"])
    duration = int(incident["duration"])

    # Select telemetry belonging to this robot
    robot_data = telemetry_df[
        telemetry_df["robot_id"] == robot_id
    ].reset_index(drop=True)

    # Extract records during the incident
    incident_data = robot_data.iloc[start:start + duration]

    # Validate incident-specific measurements
    if incident_type == "unexpected_stop":

        valid = (
            (incident_data["status"] == "Unexpected Stop").all()
            and (incident_data["velocity"] == 0).all()
        )

    elif incident_type == "wheel_slip":

        valid = (
            (incident_data["status"] == "Moving").all()
            and (incident_data["left_wheel_speed"] > 1.5).all()
            and (incident_data["right_wheel_speed"] < 1.0).all()
        )

    elif incident_type == "battery_discharge":

        # Battery is recorded before each second's consumption
        battery_values = robot_data["battery_level"].iloc[
            start:start + duration + 1
        ]

        battery_drops = -battery_values.diff().dropna()

        expected_rate = float(incident["discharge_rate"])

        valid = (
            len(battery_drops) == duration
            and np.allclose(
                battery_drops,
                expected_rate,
                atol=0.001
            )
        )

    else:
        valid = False

    if not valid:
        failed_incidents += 1
        print(f"FAIL: {robot_id} - {incident_type} at second {start}")

print(f"Total incidents: {len(incident_df)}")
print(f"Failed incidents: {failed_incidents}")

if failed_incidents == 0:
    print("PASS: All incidents match the telemetry.")
else:
    print("FAIL: Some incidents do not match the telemetry.")
