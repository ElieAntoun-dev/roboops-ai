
import pandas as pd


# ============================================================
# 1. LOAD TELEMETRY PREDICTIONS
# ============================================================

# Load telemetry records with Isolation Forest predictions
telemetry_df = pd.read_csv("data/telemetry_predictions_test.csv")

# Convert timestamps from strings to datetime objects
telemetry_df["timestamp"] = pd.to_datetime(
    telemetry_df["timestamp"]
)

# Sort records by robot and timestamp
telemetry_df = telemetry_df.sort_values(
    by=["robot_id", "timestamp"]
).reset_index(drop=True)

print("\n--- TELEMETRY PREDICTIONS ---")
print(telemetry_df.head(10))

print(f"\nTotal records: {len(telemetry_df)}")


# ============================================================
# 2. EXTRACT ANOMALOUS RECORDS
# ============================================================

# Select records classified as anomalies by Isolation Forest
anomalies_df = telemetry_df[
    telemetry_df["anomaly"] == -1
].copy()

print("\n--- DETECTED ANOMALIES ---")
print(anomalies_df.head(10))

print(f"\nTotal anomalous records: {len(anomalies_df)}")

# Count anomalies for each robot
print("\n--- ANOMALIES PER ROBOT ---")
print(anomalies_df["robot_id"].value_counts())


# ============================================================
# 3. IDENTIFY ANOMALY GROUPS
# ============================================================

# Calculate the time difference between consecutive anomalies
anomalies_df["time_gap"] = (
    anomalies_df.groupby("robot_id")["timestamp"]
    .diff()
    .dt.total_seconds()
)

# Maximum allowed gap between consecutive anomalies
max_gap_seconds = 3

# Start a new group if the gap exceeds 3 seconds
# or if this is the first anomaly for a robot
anomalies_df["new_group"] = (
    anomalies_df["time_gap"].isna()
    | (anomalies_df["time_gap"] > max_gap_seconds)
)

# Assign a unique group number to each incident
anomalies_df["incident_id"] = (
    anomalies_df["new_group"].cumsum()
)

print("\n--- GROUPED ANOMALIES ---")
print(
    anomalies_df[
        ["robot_id", "timestamp", "time_gap", "incident_id"]
    ].head(20)
)


# ============================================================
# 4. SUMMARIZE GROUPED INCIDENTS
# ============================================================

incident_summary = (
    anomalies_df.groupby(["incident_id", "robot_id"])
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        anomaly_count=("timestamp", "count"),
        minimum_anomaly_score=("anomaly_score", "min")
    )
    .reset_index()
)

# Calculate incident duration, including start and end seconds
incident_summary["duration_seconds"] = (
    incident_summary["end_time"] - incident_summary["start_time"]
).dt.total_seconds() + 1

print("\n--- INCIDENT SUMMARY ---")
print(incident_summary.head(10))

print(f"\nTotal grouped incidents: {len(incident_summary)}")


# ============================================================
# 5. FILTER SHORT ANOMALY GROUPS
# ============================================================

# Minimum number of anomalous measurements required
min_anomaly_count = 4

# Keep only incidents containing at least 4 anomalies
filtered_incidents = incident_summary[
    incident_summary["anomaly_count"] >= min_anomaly_count
].copy()

print("\n--- FILTERED INCIDENTS ---")
print(f"Before filtering: {len(incident_summary)}")
print(f"After filtering: {len(filtered_incidents)}")
print(f"Removed: {len(incident_summary) - len(filtered_incidents)}")


# ============================================================
# 6. SAVE GROUPED INCIDENTS
# ============================================================

# Save original grouped incidents (baseline)
incident_summary.to_csv(
    "data/grouped_incidents_test.csv",
    index=False
)

# Save filtered incidents separately (experiment)
filtered_incidents.to_csv(
    "data/grouped_incidents_test_filtered.csv",
    index=False
)

print("\nOriginal and filtered incidents saved successfully.")
