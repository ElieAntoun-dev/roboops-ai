
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. LOAD TRAINED MODEL AND TEST DATA
# ============================================================

model = joblib.load("ml/isolation_forest.pkl")

telemetry_df = pd.read_csv("data/robot_telemetry_test.csv")
incident_df = pd.read_csv("data/incident_log_test.csv")

print(f"Testing records: {len(telemetry_df)}")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Calculate wheel-speed difference
telemetry_df["wheel_speed_difference"] = abs(
    telemetry_df["left_wheel_speed"] -
    telemetry_df["right_wheel_speed"]
)

# Calculate battery drop per second for each robot
telemetry_df["battery_drop_rate"] = (
    -telemetry_df.groupby("robot_id")["battery_level"].diff()
).fillna(0)

# Calculate rolling average battery drop over 5 records
telemetry_df["battery_drop_rolling"] = (
    telemetry_df.groupby("robot_id")["battery_drop_rate"]
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)


# ============================================================
# 3. SELECT MODEL FEATURES
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

X_test = telemetry_df[features]


# ============================================================
# 4. PREDICT ANOMALIES
# ============================================================

# 1 = Normal, -1 = Anomaly
telemetry_df["anomaly"] = model.predict(X_test)

# Calculate anomaly scores
telemetry_df["anomaly_score"] = model.decision_function(X_test)

# Convert predictions to 0 = Normal, 1 = Anomaly
telemetry_df["predicted_incident"] = (
    telemetry_df["anomaly"] == -1
).astype(int)

print("\n--- TESTING ANOMALY RESULTS ---")
print(telemetry_df["anomaly"].value_counts())


# ============================================================
# 5. SAVE PREDICTIONS FOR INCIDENT GROUPING
# ============================================================

# Save telemetry and model predictions without ground-truth labels
prediction_columns = [
    "timestamp",
    "robot_id",
    "velocity",
    "left_wheel_speed",
    "right_wheel_speed",
    "motor_current",
    "battery_level",
    "status",
    "anomaly",
    "anomaly_score",
    "predicted_incident"
]

telemetry_df[prediction_columns].to_csv(
    "data/telemetry_predictions_test.csv",
    index=False
)

print("\nTesting predictions saved successfully.")


# ============================================================
# 6. LABEL ACTUAL TESTING INCIDENTS
# ============================================================

# Initially assume all records are normal
telemetry_df["actual_incident"] = 0
telemetry_df["incident_type"] = "normal"

# Assign ground-truth incident labels and types
for _, incident in incident_df.iterrows():

    robot_id = incident["robot_id"]
    start = int(incident["start_second"])
    end = start + int(incident["duration"])

    # Find telemetry indices belonging to this robot
    robot_indices = telemetry_df.index[
        telemetry_df["robot_id"] == robot_id
    ]

    # Select records corresponding to the incident
    incident_indices = robot_indices[start:end]

    telemetry_df.loc[incident_indices, "actual_incident"] = 1

    telemetry_df.loc[
        incident_indices, "incident_type"
    ] = incident["incident_type"]


# ============================================================
# 7. EVALUATE TESTING PERFORMANCE
# ============================================================

y_true = telemetry_df["actual_incident"]
y_pred = telemetry_df["predicted_incident"]

print("\n--- TESTING CONFUSION MATRIX ---")

print(
    pd.crosstab(
        y_true,
        y_pred,
        rownames=["Actual"],
        colnames=["Predicted"]
    )
)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- TESTING PERFORMANCE ---")

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")


# ============================================================
# 8. DETECTION RATE BY INCIDENT TYPE
# ============================================================

print("\n--- TESTING DETECTION BY INCIDENT TYPE ---")

incident_types = [
    "unexpected_stop",
    "wheel_slip",
    "battery_discharge"
]

for incident_type in incident_types:

    records = telemetry_df[
        telemetry_df["incident_type"] == incident_type
    ]

    total = len(records)

    detected = (
        records["predicted_incident"] == 1
    ).sum()

    detection_rate = detected / total if total > 0 else 0

    print(
        f"{incident_type}: "
        f"{detected}/{total} detected "
        f"({detection_rate:.2%})"
    )
