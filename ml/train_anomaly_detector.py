import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

telemetry_df = pd.read_csv("data/robot_telemetry.csv")
incident_df = pd.read_csv("data/incident_log.csv")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Calculate the absolute difference between wheel speeds
telemetry_df["wheel_speed_difference"] = abs(
    telemetry_df["left_wheel_speed"] -
    telemetry_df["right_wheel_speed"]
)

# Calculate battery percentage decrease per second
telemetry_df["battery_drop_rate"] = (
    -telemetry_df.groupby("robot_id")["battery_level"].diff()
).fillna(0)

# Calculate average battery drop over the last 5 records
telemetry_df["battery_drop_rolling"] = (
    telemetry_df.groupby("robot_id")["battery_drop_rate"]
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)


# ============================================================
# 3. SELECT MACHINE LEARNING FEATURES
# ============================================================

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

X = telemetry_df[features]


# ============================================================
# 4. TRAIN AND SAVE ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X)

joblib.dump(model, "ml/isolation_forest.pkl")

print("Isolation Forest trained and saved successfully.")


# ============================================================
# 5. PREDICT ANOMALIES
# ============================================================

# Isolation Forest returns 1 for normal and -1 for anomaly
telemetry_df["anomaly"] = model.predict(X)

# Convert predictions to 0 = normal, 1 = anomaly
telemetry_df["predicted_incident"] = (
    telemetry_df["anomaly"] == -1
).astype(int)

# Calculate anomaly scores
telemetry_df["anomaly_score"] = model.decision_function(X)


# ============================================================
# 6. LABEL ACTUAL INCIDENTS
# ============================================================

# Initially assume all records are normal
telemetry_df["actual_incident"] = 0
telemetry_df["incident_type"] = "normal"

# Label incidents using the ground-truth incident log
for _, incident in incident_df.iterrows():

    robot_id = incident["robot_id"]
    start = int(incident["start_second"])
    end = start + int(incident["duration"])

    # Find the telemetry records belonging to this robot
    robot_indices = telemetry_df.index[
        telemetry_df["robot_id"] == robot_id
    ]

    # Select the records corresponding to the incident
    incident_indices = robot_indices[start:end]

    # Assign the actual incident label and type
    telemetry_df.loc[incident_indices, "actual_incident"] = 1

    telemetry_df.loc[
        incident_indices, "incident_type"
    ] = incident["incident_type"]


# ============================================================
# 7. EVALUATE MODEL PERFORMANCE
# ============================================================

y_true = telemetry_df["actual_incident"]
y_pred = telemetry_df["predicted_incident"]

# Display confusion matrix
print("\n--- CONFUSION MATRIX ---")

print(
    pd.crosstab(
        y_true,
        y_pred,
        rownames=["Actual"],
        colnames=["Predicted"]
    )
)

# Calculate performance metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- MODEL PERFORMANCE ---")

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")


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

    # Select actual records belonging to this incident type
    incident_records = telemetry_df[
        telemetry_df["incident_type"] == incident_type
    ]

    total_records = len(incident_records)

    # Count correctly detected anomaly records
    detected_records = (
        incident_records["predicted_incident"] == 1
    ).sum()

    detection_rate = (
        detected_records / total_records
        if total_records > 0 else 0
    )

    print(
        f"{incident_type}: "
        f"{detected_records}/{total_records} detected "
        f"({detection_rate:.2%})"
    )