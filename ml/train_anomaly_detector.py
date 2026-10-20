
import pandas as pd


# ============================================================
# 1. LOAD EVALUATION, INCIDENTS, AND TELEMETRY
# ============================================================

# Load the evaluation results
evaluation_df = pd.read_csv(
    "data/incident_evaluation_test.csv"
)

# Load the classified incidents
incidents_df = pd.read_csv(
    "data/classified_incidents_test.csv"
)

# Load telemetry containing anomaly predictions
telemetry_df = pd.read_csv(
    "data/telemetry_predictions_test.csv"
)

# Convert timestamps to datetime
incidents_df["start_time"] = pd.to_datetime(
    incidents_df["start_time"]
)

incidents_df["end_time"] = pd.to_datetime(
    incidents_df["end_time"]
)

telemetry_df["timestamp"] = pd.to_datetime(
    telemetry_df["timestamp"]
)

print("Evaluation records:", len(evaluation_df))
print("Classified incidents:", len(incidents_df))
print("Telemetry records:", len(telemetry_df))


# ============================================================
# 2. RECREATE ENGINEERED FEATURES
# ============================================================

# Sort telemetry by robot and timestamp
telemetry_df = telemetry_df.sort_values(
    by=["robot_id", "timestamp"]
).reset_index(drop=True)

# Calculate the absolute difference between wheel speeds
telemetry_df["wheel_speed_difference"] = abs(
    telemetry_df["left_wheel_speed"]
    - telemetry_df["right_wheel_speed"]
)

# Calculate battery decrease between consecutive measurements
telemetry_df["battery_drop_rate"] = (
    -telemetry_df.groupby("robot_id")["battery_level"].diff()
).fillna(0)

# Calculate average battery drop over the last 5 measurements
telemetry_df["battery_drop_rolling"] = (
    telemetry_df.groupby("robot_id")["battery_drop_rate"]
    .transform(
        lambda x: x.rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)


# ============================================================
# 3. IDENTIFY UNMATCHED PREDICTIONS
# ============================================================

# Select predictions that did not match actual incidents
false_positives = evaluation_df[
    evaluation_df["matched"] == False
].copy()

# Add incident details using the incident ID
false_positives = false_positives.merge(
    incidents_df,
    left_on="predicted_incident_id",
    right_on="incident_id",
    how="left"
)

# Display the unmatched incidents
print("\n========== UNMATCHED PREDICTIONS ==========")

print(
    false_positives[
        [
            "incident_id",
            "robot_id_x",
            "predicted_type",
            "start_time",
            "end_time"
        ]
    ].to_string(index=False)
)

print(f"\nTotal unmatched predictions: {len(false_positives)}")


# ============================================================
# 4. ANALYZE ROBOT OPERATING STATES
# ============================================================

# Loop through every unmatched prediction
for _, incident in false_positives.iterrows():

    # Retrieve telemetry for the incident's robot and time window
    incident_telemetry = telemetry_df[
        (telemetry_df["robot_id"] == incident["robot_id_x"])
        & (telemetry_df["timestamp"] >= incident["start_time"])
        & (telemetry_df["timestamp"] <= incident["end_time"])
    ]

    # Count operating states during the incident
    status_counts = incident_telemetry["status"].value_counts()

    print(
        f"\nIncident {incident['incident_id']} "
        f"- {incident['robot_id_x']}"
    )

    print(status_counts)


# ============================================================
# 5. ANALYZE ANOMALIES BY OPERATING STATE
# ============================================================

# Count total measurements for each operating state
status_counts = telemetry_df.groupby("status").size()

# Count anomalous measurements for each operating state
anomaly_counts = (
    telemetry_df[telemetry_df["anomaly"] == -1]
    .groupby("status")
    .size()
)

# Create comparison table
status_analysis = pd.DataFrame({
    "total_measurements": status_counts,
    "anomalies": anomaly_counts
}).fillna(0)

# Calculate anomaly percentage for each operating state
status_analysis["anomaly_percentage"] = (
    status_analysis["anomalies"]
    / status_analysis["total_measurements"]
    * 100
)

print("\n========== ANOMALIES BY OPERATING STATE ==========")
print(status_analysis.round(3))


# ============================================================
# 6. INVESTIGATE CHARGING ANOMALIES
# ============================================================

# Select charging telemetry
charging_df = telemetry_df[
    telemetry_df["status"] == "Charging"
]

# Separate normal and anomalous charging measurements
charging_normal = charging_df[
    charging_df["anomaly"] == 1
]

charging_anomalies = charging_df[
    charging_df["anomaly"] == -1
]

# Features used by Isolation Forest
features = [
    "velocity",
    "left_wheel_speed",
    "right_wheel_speed",
    "motor_current",
    "battery_level",
    "wheel_speed_difference",
    "battery_drop_rate",
    "battery_drop_rolling"
]

# Compare average telemetry features
comparison = pd.DataFrame({
    "Normal Charging": charging_normal[features].mean(),
    "Anomalous Charging": charging_anomalies[features].mean()
})

print("\n========== CHARGING TELEMETRY COMPARISON ==========")
print(comparison.round(4))
# ============================================================
# 7. INSPECT INDIVIDUAL CHARGING ANOMALIES
# ============================================================

columns = [
    "timestamp",
    "robot_id",
    "status",
    "battery_level",
    "battery_drop_rate",
    "battery_drop_rolling",
    "anomaly"
]

print("\n========== CHARGING ANOMALY DETAILS ==========")

print(
    charging_anomalies[columns]
    .to_string(index=False)
)