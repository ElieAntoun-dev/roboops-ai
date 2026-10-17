
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. LOAD ROBOT TELEMETRY
# ============================================================

# Load the generated telemetry dataset
telemetry_df = pd.read_csv("data/robot_telemetry.csv")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Calculate the absolute difference between wheel speeds
telemetry_df["wheel_speed_difference"] = abs(
    telemetry_df["left_wheel_speed"]
    - telemetry_df["right_wheel_speed"]
)

# Calculate battery percentage decrease per second, per robot
telemetry_df["battery_drop_rate"] = (
    -telemetry_df.groupby("robot_id")["battery_level"].diff()
).fillna(0)


# ============================================================
# 3. SELECT MACHINE LEARNING FEATURES
# ============================================================

# Select the measurements used by the ML model
features = [
    "velocity",
    "left_wheel_speed",
    "right_wheel_speed",
    "motor_current",
    "battery_level",
    "wheel_speed_difference",
    "battery_drop_rate"
]

# Create the machine learning input dataset
X = telemetry_df[features]

print("\nSelected ML features:")
print(X.head())

print(f"\nDataset shape: {X.shape}")


# ============================================================
# 4. TRAIN ISOLATION FOREST
# ============================================================

# Initialize the anomaly detection model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Train the model
model.fit(X)

# Predict anomalies for each telemetry record
telemetry_df["anomaly"] = model.predict(X)

print("\n--- ANOMALY DETECTION RESULTS ---")
print(telemetry_df["anomaly"].value_counts())


# ============================================================
# 5. LABEL ACTUAL INCIDENTS
# ============================================================

# Load the ground-truth incident log
incident_df = pd.read_csv("data/incident_log.csv")

# Initially assume all telemetry records are normal
telemetry_df["actual_incident"] = 0

# Go through each recorded incident
for _, incident in incident_df.iterrows():

    robot_id = incident["robot_id"]
    start = int(incident["start_second"])
    end = start + int(incident["duration"])

    # Find telemetry indices belonging to this robot
    robot_indices = telemetry_df.index[
        telemetry_df["robot_id"] == robot_id
    ]

    # Select the records during the incident
    incident_indices = robot_indices[start:end]

    # Mark these records as actual incidents
    telemetry_df.loc[incident_indices, "actual_incident"] = 1


# ============================================================
# 6. EVALUATE MODEL PREDICTIONS
# ============================================================

# Convert predictions: 0 = Normal, 1 = Anomaly
telemetry_df["predicted_incident"] = (
    telemetry_df["anomaly"] == -1
).astype(int)

# Compare actual incidents with model predictions
print("\n--- MODEL EVALUATION ---")

print(pd.crosstab(
    telemetry_df["actual_incident"],
    telemetry_df["predicted_incident"],
    rownames=["Actual"],
    colnames=["Predicted"]
))

precision = precision_score(
    telemetry_df["actual_incident"],
    telemetry_df["predicted_incident"]
)

recall = recall_score(
    telemetry_df["actual_incident"],
    telemetry_df["predicted_incident"]
)

f1 = f1_score(
    telemetry_df["actual_incident"],
    telemetry_df["predicted_incident"]
)

print("\n--- MODEL PERFORMANCE METRICS ---")

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")


# ============================================================
# 7. EVALUATE EACH INCIDENT TYPE
# ============================================================

# Initially label all telemetry records as normal
telemetry_df["incident_type"] = "normal"

# Loop through every actual incident
for _, incident in incident_df.iterrows():

    robot_id = incident["robot_id"]
    incident_type = incident["incident_type"]

    start = int(incident["start_second"])
    end = start + int(incident["duration"])

    # Find all telemetry indices belonging to this robot
    robot_indices = telemetry_df.index[
        telemetry_df["robot_id"] == robot_id
    ]

    # Select the indices corresponding to the incident
    incident_indices = robot_indices[start:end]

    # Assign the actual incident type to these records
    telemetry_df.loc[incident_indices, "incident_type"] = incident_type


    
# ============================================================
# 8. DETECTION RATE BY INCIDENT TYPE
# ============================================================

print("\n--- DETECTION RATE BY INCIDENT TYPE ---")

incident_types = [
    "unexpected_stop",
    "wheel_slip",
    "battery_discharge"
]

for incident_type in incident_types:

    incident_records = telemetry_df[
        telemetry_df["incident_type"] == incident_type
    ]

    total_records = len(incident_records)

    detected_records = (
        incident_records["predicted_incident"] == 1
    ).sum()

    detection_rate = detected_records / total_records

    print(
        f"{incident_type}: "
        f"{detected_records}/{total_records} detected "
        f"({detection_rate:.2%})"
    )


# ============================================================
# 9. INVESTIGATE MISSED BATTERY-DISCHARGE RECORDS
# ============================================================

# Select only battery-discharge records
battery_records = telemetry_df[
    telemetry_df["incident_type"] == "battery_discharge"
]

# Separate detected and missed records
detected_battery = battery_records[
    battery_records["predicted_incident"] == 1
]

missed_battery = battery_records[
    battery_records["predicted_incident"] == 0
]

# Compare the battery-drop rates
print("\n--- BATTERY DISCHARGE ANALYSIS ---")

print("\nDetected battery-discharge records:")
print(detected_battery["battery_drop_rate"].describe())

print("\nMissed battery-discharge records:")
print(missed_battery["battery_drop_rate"].describe())

# Compare other telemetry measurements
comparison_features = [
    "battery_drop_rate",
    "battery_level",
    "velocity",
    "motor_current"
]

print("\n--- AVERAGE FEATURE COMPARISON ---")

print("Detected records:")
print(detected_battery[comparison_features].mean())

print("\nMissed records:")
print(missed_battery[comparison_features].mean())




# ============================================================
# 10. INVESTIGATE INDIVIDUAL MISSED BATTERY INCIDENTS
# ============================================================

# Display the missed battery-discharge records
columns_to_display = [
    "robot_id",
    "timestamp",
    "battery_drop_rate",
    "velocity",
    "motor_current"
]

print("\n--- MISSED BATTERY-DISCHARGE RECORDS ---")
print(missed_battery[columns_to_display].to_string(index=True))

# Check whether missed records occur at incident starts
missed_at_start = 0

for _, incident in incident_df.iterrows():

    if incident["incident_type"] != "battery_discharge":
        continue

    robot_indices = telemetry_df.index[
        telemetry_df["robot_id"] == incident["robot_id"]
    ]

    start_index = robot_indices[int(incident["start_second"])]

    if start_index in missed_battery.index:
        missed_at_start += 1

print("\n--- MISSED INCIDENT STARTS ---")
print(f"Missed records at incident starts: {missed_at_start}")
print(f"Total missed battery records: {len(missed_battery)}")


