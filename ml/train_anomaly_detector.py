
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