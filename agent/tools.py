
from pathlib import Path

import pandas as pd
from langchain.tools import tool


# ============================================================
# 1. FILE PATHS
# ============================================================

# Locate the data folder relative to this Python file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TELEMETRY_PATH = DATA_DIR / "robot_telemetry_test.csv"

INCIDENTS_PATH = DATA_DIR / "classified_incidents_test_filtered.csv"


# ============================================================
# 2. TOOL 1 - GET ROBOT TELEMETRY
# ============================================================

@tool
def get_robot_telemetry(robot_id: str) -> dict:
    """Retrieve the latest recorded telemetry measurement for a specific robot.

Use this tool when the user asks about the robot's latest known
state, including battery level, velocity, wheel speeds, motor
current, operating status, or timestamp.

This tool returns only the most recent recorded measurement.
It does not provide telemetry history or incident-specific data.

Args:
    robot_id: Robot identifier, such as "AMR-02".
"""

    # Load the telemetry dataset
    df = pd.read_csv(TELEMETRY_PATH)

    # Filter the telemetry for the requested robot
    robot_data = df[df["robot_id"] == robot_id].copy()

    # Check whether the robot exists
    if robot_data.empty:
        return {
            "error": f"No telemetry found for robot {robot_id}"
        }

    # Convert timestamps into datetime objects
    robot_data["timestamp"] = pd.to_datetime(
        robot_data["timestamp"]
    )

    # Find the latest recorded telemetry
    latest = robot_data.sort_values("timestamp").iloc[-1]

    # Convert telemetry measurements into a dictionary
    telemetry = {
        column: (
            value.item() if hasattr(value, "item") else value
        )
        for column, value in latest.items()
        if column not in ["robot_id", "timestamp"]
    }

    # Return structured telemetry data
    return {
        "robot_id": robot_id,
        "timestamp": latest["timestamp"].isoformat(),
        "telemetry": telemetry
    }


# ============================================================
# 3. TOOL 2 - GET INCIDENT DETAILS
# ============================================================

@tool
def get_incident_details(incident_id: int) -> dict:
    """Retrieve summary information about a detected robot incident.

Use this tool to identify the affected robot, incident type,
start and end timestamps, duration, anomaly count, and
aggregated telemetry statistics.

This tool provides a summary of an incident, not the individual
telemetry measurements recorded during it.

For detailed investigation of telemetry changes during an
incident, also use get_incident_telemetry.

Args:
    incident_id: Numeric identifier of the detected incident.
"""

    # Load the classified incidents dataset
    df = pd.read_csv(INCIDENTS_PATH)

    # Filter the dataset for the requested incident
    incident_data = df[df["incident_id"] == incident_id]

    # Check whether the incident exists
    if incident_data.empty:
        return {
            "error": f"Incident {incident_id} was not found."
        }

    # Select the matching incident
    incident = incident_data.iloc[0]

    # Convert the incident row into a dictionary
    result = incident.to_dict()

    # Convert NumPy values into standard Python values
    result = {
        key: (
            value.item() if hasattr(value, "item") else value
        )
        for key, value in result.items()
    }

    return result


# ============================================================
# 4. TOOL 3 - GET INCIDENT TELEMETRY
# ============================================================

@tool
def get_incident_telemetry(incident_id: int) -> dict:
    """Retrieve timestamped robot telemetry recorded during an incident.

Use this tool when investigating how measurements changed
during an incident, including battery level, velocity,
wheel speeds, motor current, and robot operating status.

This tool retrieves all available telemetry records belonging
to the affected robot between the incident start and end times.

Use get_incident_details for the incident classification and
summary statistics. Use this tool for detailed evidence.

Args:
    incident_id: Numeric identifier of the detected incident.
"""

    # 1. Load the classified incidents dataset
    incidents_df = pd.read_csv(INCIDENTS_PATH)

    # 2. Find the requested incident
    incident_data = incidents_df[
        incidents_df["incident_id"] == incident_id
    ]

    # 3. Check whether the incident exists
    if incident_data.empty:
        return {
            "error": f"Incident {incident_id} was not found."
        }

    # 4. Extract incident information
    incident = incident_data.iloc[0]

    robot_id = incident["robot_id"]

    start_time = pd.to_datetime(incident["start_time"])
    end_time = pd.to_datetime(incident["end_time"])

    # 5. Load the telemetry dataset
    telemetry_df = pd.read_csv(TELEMETRY_PATH)

    # 6. Convert timestamps into datetime objects
    telemetry_df["timestamp"] = pd.to_datetime(
        telemetry_df["timestamp"]
    )

    # 7. Filter telemetry by robot ID and incident timeframe
    incident_telemetry = telemetry_df[
        (telemetry_df["robot_id"] == robot_id)
        & (telemetry_df["timestamp"] >= start_time)
        & (telemetry_df["timestamp"] <= end_time)
    ].copy()

    # 8. Check whether telemetry was found
    if incident_telemetry.empty:
        return {
            "error": f"No telemetry found for incident {incident_id}."
        }

    # 9. Sort measurements chronologically
    incident_telemetry = incident_telemetry.sort_values(
        "timestamp"
    )

    # 10. Convert timestamps to strings
    incident_telemetry["timestamp"] = (
        incident_telemetry["timestamp"].dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )
    )

    # 11. Convert telemetry rows into dictionaries
    records = incident_telemetry.to_dict(orient="records")

    # 12. Return structured incident telemetry
    return {
        "incident_id": int(incident_id),
        "robot_id": str(robot_id),
        "incident_type": str(
            incident["predicted_incident_type"]
        ),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "telemetry_count": len(records),
        "telemetry": records
    }

# ============================================================
# TOOL 4 - ANALYZE BATTERY DISCHARGE
# ============================================================

@tool
def analyze_battery_discharge(incident_id: int) -> dict:
    """Calculate battery discharge statistics for a detected incident.

    Use this tool when the user asks how much or how quickly
    the battery level changed during an incident.

    Returns the initial and final battery levels, total change,
    elapsed time, average discharge rate, and measurement count.

    This tool performs numerical calculations only. It does not
    establish whether the discharge is abnormal or determine
    the underlying root cause.

    Args:
        incident_id: Numeric identifier of the detected incident.
    """

    # 1. Retrieve the incident's telemetry using our existing tool
    result = get_incident_telemetry.invoke({
        "incident_id": incident_id
    })

    # 2. Check for errors
    if "error" in result:
        return result

    # 3. Convert the telemetry records into a DataFrame
    df = pd.DataFrame(result["telemetry"])

    # 4. Check that sufficient battery data is available
    if "battery_level" not in df.columns:
        return {"error": "Battery level data is unavailable."}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["battery_level"] = pd.to_numeric(
        df["battery_level"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "battery_level"]
    ).sort_values("timestamp")

    if len(df) < 2:
        return {
            "error": "At least two valid battery measurements are required."
        }

    # 5. Get the first and last battery measurements
    initial_battery = float(df["battery_level"].iloc[0])
    final_battery = float(df["battery_level"].iloc[-1])

    # 6. Calculate elapsed time between the measurements
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    elapsed_seconds = (
        end_time - start_time
    ).total_seconds()

    if elapsed_seconds <= 0:
        return {
            "error": "Elapsed time must be greater than zero."
        }

    # 7. Calculate battery decrease
    total_battery_drop = initial_battery - final_battery

    # 8. Calculate average discharge rate
    average_discharge_rate = (
        total_battery_drop / elapsed_seconds
    )

    # 9. Return the calculated results
    return {
        "incident_id": int(incident_id),
        "robot_id": result["robot_id"],
        "initial_battery_percent": round(initial_battery, 3),
        "final_battery_percent": round(final_battery, 3),
        "battery_change_percentage_points": round(
            final_battery - initial_battery, 3
        ),
        "total_battery_drop_percentage_points": round(
            total_battery_drop, 3
        ),
        "elapsed_seconds": elapsed_seconds,
        "average_discharge_rate_percentage_points_per_second": round(
            average_discharge_rate, 4
        ),
        "measurement_count": len(df)
    }

# ============================================================
# TOOL 5 - ANALYZE WHEEL SLIP
# ============================================================

@tool
def analyze_wheel_slip(incident_id: int) -> dict:
    """Analyze wheel-speed differences during a detected incident.

    Use this tool to investigate possible wheel slip or wheel-speed
    inconsistencies using recorded left wheel speed, right wheel
    speed, and robot velocity.

    Returns maximum and average wheel-speed differences and
    wheel-to-robot velocity discrepancies.

    These indicators do not independently confirm physical slip.

    Args:
        incident_id: Numeric identifier of the detected incident.
    """

    # 1. Retrieve incident telemetry
    result = get_incident_telemetry.invoke({
        "incident_id": incident_id
    })

    if "error" in result:
        return result

    # 2. Convert telemetry records into a DataFrame
    df = pd.DataFrame(result["telemetry"])

    required_columns = [
        "timestamp",
        "velocity",
        "left_wheel_speed",
        "right_wheel_speed"
    ]

    # 3. Verify required measurements exist
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        return {
            "error": f"Missing telemetry columns: {missing}"
        }

    # 4. Convert measurements to numeric values
    for column in required_columns[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required_columns
    ).sort_values("timestamp")

    if df.empty:
        return {"error": "No valid wheel telemetry available."}

    # 5. Calculate absolute wheel-speed difference
    df["wheel_speed_difference"] = (
        df["left_wheel_speed"] - df["right_wheel_speed"]
    ).abs()

    # 6. Compare wheel speeds with reported robot velocity
    df["left_velocity_difference"] = (
        df["left_wheel_speed"] - df["velocity"]
    ).abs()

    df["right_velocity_difference"] = (
        df["right_wheel_speed"] - df["velocity"]
    ).abs()

    # 7. Identify the measurement with maximum wheel difference
    max_index = df["wheel_speed_difference"].idxmax()

    # 8. Return numerical indicators
    return {
        "incident_id": int(incident_id),
        "robot_id": result["robot_id"],
        "maximum_wheel_speed_difference_mps": round(
            float(df["wheel_speed_difference"].max()), 4
        ),
        "average_wheel_speed_difference_mps": round(
            float(df["wheel_speed_difference"].mean()), 4
        ),
        "maximum_left_velocity_difference_mps": round(
            float(df["left_velocity_difference"].max()), 4
        ),
        "maximum_right_velocity_difference_mps": round(
            float(df["right_velocity_difference"].max()), 4
        ),
        "timestamp_of_maximum_difference": (
            df.loc[max_index, "timestamp"].isoformat()
        ),
        "measurement_count": len(df),
        "assessment": (
            "Wheel-speed discrepancies were calculated. "
            "Physical wheel slip cannot be confirmed from "
            "these measurements alone."
        )
    }

# ============================================================
# TOOL 6 - ANALYZE UNEXPECTED STOP
# ============================================================

@tool
def analyze_unexpected_stop(incident_id: int) -> dict:
    """Analyze velocity changes and operating status during an incident.

    Use this tool to investigate unexpected stops, rapid
    deceleration, or interruptions in robot motion.

    Returns velocity statistics, maximum observed deceleration,
    motor-current statistics, and recorded operating statuses.

    This tool does not establish the cause of a stop or
    determine whether a safety system was triggered.

    Args:
        incident_id: Numeric identifier of the detected incident.
    """

    # 1. Retrieve incident telemetry
    result = get_incident_telemetry.invoke({
        "incident_id": incident_id
    })

    if "error" in result:
        return result

    # 2. Convert telemetry records into a DataFrame
    df = pd.DataFrame(result["telemetry"])

    required_columns = [
        "timestamp",
        "velocity",
        "motor_current",
        "status"
    ]

    # 3. Verify required columns
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        return {
            "error": f"Missing telemetry columns: {missing}"
        }

    # 4. Convert timestamps and numerical measurements
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    for column in ["velocity", "motor_current"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).sort_values("timestamp")

    if len(df) < 2:
        return {
            "error": "At least two valid telemetry records are required."
        }

    # 5. Calculate time difference between measurements
    df["time_difference"] = (
        df["timestamp"].diff().dt.total_seconds()
    )

    # 6. Calculate change in velocity
    df["velocity_change"] = df["velocity"].diff()

    # 7. Calculate acceleration / deceleration
    df["acceleration"] = (
        df["velocity_change"] / df["time_difference"]
    )

    # 8. Ignore invalid or zero time intervals
    valid_acceleration = df.loc[
        df["time_difference"] > 0,
        "acceleration"
    ].dropna()

    if valid_acceleration.empty:
        return {
            "error": "No valid time intervals for deceleration analysis."
        }

    # 9. Identify maximum observed deceleration
    minimum_acceleration = float(valid_acceleration.min())

    maximum_deceleration = max(
        0.0,
        -minimum_acceleration
    )

    # 10. Return analysis results
    return {
        "incident_id": int(incident_id),
        "robot_id": result["robot_id"],
        "initial_velocity_mps": round(
            float(df["velocity"].iloc[0]), 4
        ),
        "final_velocity_mps": round(
            float(df["velocity"].iloc[-1]), 4
        ),
        "minimum_velocity_mps": round(
            float(df["velocity"].min()), 4
        ),
        "maximum_velocity_mps": round(
            float(df["velocity"].max()), 4
        ),
        "maximum_observed_deceleration_mps2": round(
            maximum_deceleration, 4
        ),
        "average_motor_current_A": round(
            float(df["motor_current"].mean()), 4
        ),
        "maximum_motor_current_A": round(
            float(df["motor_current"].max()), 4
        ),
        "recorded_statuses": sorted(
            df["status"].astype(str).unique().tolist()
        ),
        "measurement_count": len(df),
        "assessment": (
            "Motion and current measurements were analyzed. "
            "The cause of any stopping event is not established."
        )
    }





# ============================================================
# 4. TEST THE TOOLS
# ============================================================



if __name__ == "__main__":

    # Test wheel slip using an actual wheel-slip incident
    print("\n========== WHEEL SLIP - INCIDENT 1 ==========")

    wheel_result = analyze_wheel_slip.invoke({
        "incident_id": 1
    })

    for key, value in wheel_result.items():
        print(f"{key}: {value}")

    # Test unexpected stop using an actual stopping incident
    print("\n========== UNEXPECTED STOP - INCIDENT 6 ==========")

    stop_result = analyze_unexpected_stop.invoke({
        "incident_id": 6
    })

    for key, value in stop_result.items():
        print(f"{key}: {value}")


