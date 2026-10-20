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
# 2. IDENTIFY UNMATCHED PREDICTIONS
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
# 3. ANALYZE ROBOT OPERATING STATES
# ============================================================

for _, incident in false_positives.iterrows():

    # Retrieve telemetry for the incident's robot and time window
    incident_telemetry = telemetry_df[
        (telemetry_df["robot_id"] == incident["robot_id_x"])
        & (telemetry_df["timestamp"] >= incident["start_time"])
        & (telemetry_df["timestamp"] <= incident["end_time"])
    ]

    # Count the operating states during the incident
    status_counts = incident_telemetry["status"].value_counts()

    print(f"\nIncident {incident['incident_id']} - {incident['robot_id_x']}")
    print(status_counts)

    # ============================================================
# 4. ANALYZE ANOMALIES BY OPERATING STATE
# ============================================================

# Count total telemetry measurements per operating state
status_counts = telemetry_df.groupby("status").size()

# Count anomalous measurements per operating state
anomaly_counts = (
    telemetry_df[telemetry_df["anomaly"] == -1]
    .groupby("status")
    .size()
)

# Calculate anomaly percentage for each operating state
status_analysis = pd.DataFrame({
    "total_measurements": status_counts,
    "anomalies": anomaly_counts
}).fillna(0)

status_analysis["anomaly_percentage"] = (
    status_analysis["anomalies"]
    / status_analysis["total_measurements"]
    * 100
)

print("\n========== ANOMALIES BY OPERATING STATE ==========")
print(status_analysis)

# ============================================================
# 5. INVESTIGATE CHARGING ANOMALIES
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

# Compare average telemetry features
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

comparison = pd.DataFrame({
    "Normal Charging": charging_normal[features].mean(),
    "Anomalous Charging": charging_anomalies[features].mean()
})

print("\n========== CHARGING TELEMETRY COMPARISON ==========")
print(comparison.round(3))