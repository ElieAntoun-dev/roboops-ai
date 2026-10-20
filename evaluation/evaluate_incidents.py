
"""Evaluate RoboOps AI incidents using one-to-one matching."""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. FILE PATHS AND CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

PREDICTED_PATH = DATA_DIR / "classified_incidents_test_filtered.csv"
ACTUAL_PATH = DATA_DIR / "incident_log_test.csv"
TELEMETRY_PATH = DATA_DIR / "robot_telemetry_test.csv"

OUTPUT_PATH = DATA_DIR / "incident_evaluation_test_one_to_one.csv"

RESULT_COLUMNS = [
    "predicted_incident_id",
    "robot_id",
    "predicted_type",
    "actual_type",
    "matched",
    "correct_type"
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    """Load predicted incidents, actual incidents, and telemetry."""

    predicted = pd.read_csv(PREDICTED_PATH)
    actual = pd.read_csv(ACTUAL_PATH)
    telemetry = pd.read_csv(TELEMETRY_PATH)

    # Convert timestamps into datetime objects
    predicted["start_time"] = pd.to_datetime(predicted["start_time"])
    predicted["end_time"] = pd.to_datetime(predicted["end_time"])
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])

    return predicted, actual, telemetry


# ============================================================
# 3. PREPARE ACTUAL INCIDENT TIMESTAMPS
# ============================================================

def prepare_actual_incidents(actual, telemetry):
    """Convert simulation-relative seconds into timestamps."""

    actual = actual.copy()

    # Find the simulation start time for each robot
    simulation_starts = telemetry.groupby(
        "robot_id"
    )["timestamp"].min()

    # Assign the corresponding start time to each incident
    actual["simulation_start"] = actual["robot_id"].map(
        simulation_starts
    )

    # Calculate the incident's ending second
    actual["end_second"] = (
        actual["start_second"]
        + actual["duration"]
        - 1
    )

    # Convert start and end seconds into timestamps
    actual["start_time"] = (
        actual["simulation_start"]
        + pd.to_timedelta(actual["start_second"], unit="s")
    )

    actual["end_time"] = (
        actual["simulation_start"]
        + pd.to_timedelta(actual["end_second"], unit="s")
    )

    return actual


# ============================================================
# 4. ONE-TO-ONE INCIDENT MATCHING
# ============================================================

def match_incidents(predicted, actual):
    """Match incidents by maximizing unique matches and overlap."""

    n_predicted = len(predicted)
    n_actual = len(actual)

    if n_predicted == 0 or n_actual == 0:
        return {}

    # Create an empty overlap matrix
    overlap_matrix = np.zeros((n_predicted, n_actual))

    # Calculate overlaps between predicted and actual incidents
    for predicted_id, prediction in enumerate(
        predicted.itertuples(index=False)
    ):

        for actual_id, incident in enumerate(
            actual.itertuples(index=False)
        ):

            # Only compare incidents belonging to the same robot
            if prediction.robot_id != incident.robot_id:
                continue

            # Calculate the overlapping time interval
            overlap_start = max(
                prediction.start_time,
                incident.start_time
            )

            overlap_end = min(
                prediction.end_time,
                incident.end_time
            )

            # Store the overlap duration if the intervals intersect
            if overlap_start <= overlap_end:

                overlap_seconds = (
                    overlap_end - overlap_start
                ).total_seconds() + 1

                overlap_matrix[
                    predicted_id, actual_id
                ] = overlap_seconds

    # Prioritize the number of matches over overlap duration
    match_bonus = overlap_matrix.sum() + 1

    # Add dummy columns to allow unmatched predictions
    score_matrix = np.zeros(
        (n_predicted, n_actual + n_predicted)
    )

    # Assign scores to valid overlapping pairs
    score_matrix[:, :n_actual] = np.where(
        overlap_matrix > 0,
        match_bonus + overlap_matrix,
        0
    )

    # Find the optimal one-to-one assignments
    assigned_predictions, assigned_actuals = (
        linear_sum_assignment(
            score_matrix,
            maximize=True
        )
    )

    # Keep only valid matches with real incidents
    selected_matches = {}

    for predicted_id, actual_id in zip(
        assigned_predictions,
        assigned_actuals
    ):

        if (
            actual_id < n_actual
            and overlap_matrix[predicted_id, actual_id] > 0
        ):
            selected_matches[predicted_id] = actual_id

    return selected_matches


# ============================================================
# 5. BUILD EVALUATION RESULTS
# ============================================================

def build_results(predicted, actual, selected_matches):
    """Record matches and check incident classification."""

    results = []

    for predicted_id, prediction in enumerate(
        predicted.itertuples(index=False)
    ):

        actual_id = selected_matches.get(predicted_id)

        if actual_id is not None:

            # Retrieve the matched actual incident
            actual_type = actual.iloc[actual_id]["incident_type"]

            matched = True

            correct_type = (
                prediction.predicted_incident_type == actual_type
            )

        else:

            # Prediction did not match any actual incident
            actual_type = "none"
            matched = False
            correct_type = False

        results.append({
            "predicted_incident_id": prediction.incident_id,
            "robot_id": prediction.robot_id,
            "predicted_type": prediction.predicted_incident_type,
            "actual_type": actual_type,
            "matched": matched,
            "correct_type": correct_type
        })

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


# ============================================================
# 6. CALCULATE EVALUATION METRICS
# ============================================================

def calculate_metrics(results, n_actual):
    """Calculate detection and classification performance."""

    # True positives: uniquely matched predictions
    tp = int(results["matched"].sum())

    # False positives: unmatched predictions
    fp = len(results) - tp

    # False negatives: actual incidents not detected
    fn = n_actual - tp

    # Detection precision
    precision = tp / (tp + fp) if tp + fp else 0.0

    # Detection recall
    recall = tp / (tp + fn) if tp + fn else 0.0

    # Detection F1 score
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # Classification accuracy among matched predictions
    accuracy = (
        results["correct_type"].sum() / tp
        if tp
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1, accuracy


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    # Load the datasets
    predicted, actual, telemetry = load_data()

    # Prepare actual incident timestamps
    actual = prepare_actual_incidents(actual, telemetry)

    # Perform one-to-one incident matching
    selected_matches = match_incidents(predicted, actual)

    # Build evaluation results
    results = build_results(predicted, actual, selected_matches)

    # Calculate evaluation metrics
    tp, fp, fn, precision, recall, f1, accuracy = (
        calculate_metrics(results, len(actual))
    )

    # Display evaluation results
    print("\n========== EVALUATION RESULTS ==========")

    print(f"Actual incidents: {len(actual)}")
    print(f"Predicted incidents: {len(predicted)}")

    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    print(f"Detection precision: {precision:.2%}")
    print(f"Detection recall: {recall:.2%}")
    print(f"Detection F1 score: {f1:.2%}")

    print(
        "Classification accuracy (matched predictions): "
        f"{accuracy:.2%}"
    )

    # Save the evaluation results
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nEvaluation results saved to {OUTPUT_PATH}")


# ============================================================
# 8. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
