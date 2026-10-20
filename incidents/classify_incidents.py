
import pandas as pd


# ============================================================
# 1. LOAD INCIDENTS AND TELEMETRY
# ============================================================

# Load filtered grouped incidents and telemetry predictions
incident_df = pd.read_csv("data/grouped_incidents_test_filtered.csv")
telemetry_df = pd.read_csv("data/telemetry_predictions_test.csv")

# Convert timestamps to datetime objects
incident_df["start_time"] = pd.to_datetime(incident_df["start_time"])
incident_df["end_time"] = pd.to_datetime(incident_df["end_time"])
telemetry_df["timestamp"] = pd.to_datetime(telemetry_df["timestamp"])

print(f"Grouped incidents: {len(incident_df)}")
print(f"Telemetry records: {len(telemetry_df)}")


# ============================================================
# 2. CALCULATE BATTERY DROP RATE
# ============================================================

# Sort telemetry by robot and timestamp
telemetry_df = telemetry_df.sort_values(
    by=["robot_id", "timestamp"]
).reset_index(drop=True)

# Calculate battery decrease between consecutive measurements
telemetry_df["battery_drop_rate"] = (
    -telemetry_df.groupby("robot_id")["battery_level"].diff()
).fillna(0)


# ============================================================
# 3. PREPARE INCIDENT CLASSIFICATION
# ============================================================

# Store the classification results for all incidents
classified_incidents = []


# Loop through every grouped incident
for _, incident in incident_df.iterrows():

    # Select telemetry belonging to the robot and incident time window
    incident_telemetry = telemetry_df[
        (telemetry_df["robot_id"] == incident["robot_id"])
        & (telemetry_df["timestamp"] >= incident["start_time"])
        & (telemetry_df["timestamp"] <= incident["end_time"])
    ]

    print(f"\n--- INCIDENT {incident['incident_id']} ---")
    print(f"Robot: {incident['robot_id']}")
    print(f"Telemetry records: {len(incident_telemetry)}")

    # Skip incidents without matching telemetry
    if incident_telemetry.empty:
        print("No matching telemetry found. Skipping incident.")
        continue


    # ========================================================
    # 4. CALCULATE INCIDENT FEATURES
    # ========================================================

    # Calculate average robot velocity
    average_velocity = incident_telemetry["velocity"].mean()

    # Calculate maximum difference between left and right wheel speeds
    maximum_wheel_difference = (
        incident_telemetry["left_wheel_speed"]
        - incident_telemetry["right_wheel_speed"]
    ).abs().max()

    # Calculate average motor current
    average_motor_current = incident_telemetry["motor_current"].mean()

    # Calculate maximum battery decrease per second
    maximum_battery_drop = incident_telemetry["battery_drop_rate"].max()

    # Display calculated features
    print(f"Average velocity: {average_velocity:.3f}")
    print(f"Maximum wheel difference: {maximum_wheel_difference:.3f}")
    print(f"Average motor current: {average_motor_current:.3f}")
    print(f"Maximum battery drop: {maximum_battery_drop:.3f} %/s")


    # ========================================================
    # 5. CLASSIFY INCIDENT
    # ========================================================

    # Identify wheel slip based on wheel-speed difference
    if maximum_wheel_difference > 0.5:
        incident_type = "wheel_slip"

    # Identify abnormal battery discharge
    elif maximum_battery_drop > 0.1:
        incident_type = "battery_discharge"

    # Identify unexpected stops
    elif average_velocity < 0.1 and average_motor_current > 3:
        incident_type = "unexpected_stop"

    # No matching incident pattern
    else:
        incident_type = "unknown"

    print(f"Predicted incident type: {incident_type}")


    # ========================================================
    # 6. STORE CLASSIFIED INCIDENT
    # ========================================================

    # Save incident details, calculated features, and prediction
    classified_incidents.append({
        "incident_id": incident["incident_id"],
        "robot_id": incident["robot_id"],
        "start_time": incident["start_time"],
        "end_time": incident["end_time"],
        "duration_seconds": incident["duration_seconds"],
        "anomaly_count": incident["anomaly_count"],
        "average_velocity": average_velocity,
        "maximum_wheel_difference": maximum_wheel_difference,
        "average_motor_current": average_motor_current,
        "maximum_battery_drop": maximum_battery_drop,
        "predicted_incident_type": incident_type
    })


# ============================================================
# 7. SAVE CLASSIFIED INCIDENTS TO CSV
# ============================================================

# Convert the list of dictionaries into a DataFrame
classified_df = pd.DataFrame(classified_incidents)

# Save filtered classification results separately
classified_df.to_csv(
    "data/classified_incidents_test_filtered.csv",
    index=False
)

print("\nFiltered classified incidents saved successfully.")

# Display the number of incidents per category
print("\nClassification summary:")
print(classified_df["predicted_incident_type"].value_counts())
