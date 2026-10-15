
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# 1. SELECT THE NEXT NORMAL ROBOT STATUS
# ============================================================

def get_next_status(previous_status):

    # Allowed transitions and their probabilities
    status_transitions = {
        "Moving": {
            "Moving": 0.95,
            "Idle": 0.05
        },
        "Idle": {
            "Moving": 0.15,
            "Idle": 0.80,
            "Charging": 0.05
        },
        "Charging": {
            "Charging": 0.95,
            "Idle": 0.05
        }
    }

    # Get possible statuses based on previous status
    possible_statuses = status_transitions[previous_status]

    # Randomly select the next normal status
    next_status = str(np.random.choice(
        list(possible_statuses.keys()),
        p=list(possible_statuses.values())
    ))

    return next_status


# ============================================================
# 2. GENERATE ONE TELEMETRY RECORD
# ============================================================

def generate_telemetry(
    robot_id,
    timestamp,
    battery_level,
    status,
    incident_type=None
):

    # Generate normal measurements based on status
    if status == "Moving":

        velocity = round(np.random.normal(1.2, 0.02), 3)

        wheel_difference = np.random.normal(0, 0.01)

        left_wheel_speed = round(velocity - wheel_difference, 3)
        right_wheel_speed = round(velocity + wheel_difference, 3)

        motor_current = round(np.random.normal(8.5, 0.3), 2)

    else:

        velocity = 0.0
        left_wheel_speed = 0.0
        right_wheel_speed = 0.0

        if status == "Idle":
            motor_current = round(np.random.normal(0.5, 0.1), 2)
        else:
            motor_current = 0.0

    # Apply abnormal measurements during incidents
    if incident_type == "unexpected_stop":

        velocity = 0.0
        left_wheel_speed = 0.0
        right_wheel_speed = 0.0

        motor_current = round(np.random.uniform(10.0, 15.0), 2)

    elif incident_type == "recovery":

        velocity = 0.0
        left_wheel_speed = 0.0
        right_wheel_speed = 0.0

        motor_current = 2.0

    elif incident_type == "wheel_slip":

        velocity = round(np.random.uniform(0.4, 0.8), 3)

        left_wheel_speed = round(np.random.uniform(1.5, 2.0), 3)
        right_wheel_speed = round(np.random.uniform(0.6, 1.0), 3)

        motor_current = round(np.random.uniform(12.0, 16.0), 2)

    # Return telemetry measurements
    return {
        "timestamp": timestamp,
        "robot_id": robot_id,
        "velocity": velocity,
        "left_wheel_speed": left_wheel_speed,
        "right_wheel_speed": right_wheel_speed,
        "motor_current": motor_current,
        "battery_level": battery_level,
        "status": status
    }


# ============================================================
# 3. DEFINE ROBOT FLEET AND SIMULATION SETTINGS
# ============================================================

robots = {
    "AMR-01": 85.0,
    "AMR-02": 70.0,
    "AMR-03": 95.0,
    "AMR-04": 60.0,
    "AMR-05": 90.0
}

simulation_seconds = 100
start_time = datetime.now()

incident_types = [
    "unexpected_stop",
    "wheel_slip",
    "battery_discharge"
]


# ============================================================
# 4. CREATE INCIDENT SCHEDULES
# ============================================================

# Create 100 incident states for each robot
incident_schedules = {
    robot_id: [None] * simulation_seconds
    for robot_id in robots
}

# Store ground-truth incidents and discharge rates
incident_log = []
robots_discharge_rates = {}

# Randomly assign at most one incident to each robot
for robot_id in robots:

    # Select an incident or no incident
    selected_incident = np.random.choice(
        [None] + incident_types
    )

    if selected_incident is None:
        continue

    # Randomize incident duration
    if selected_incident == "battery_discharge":
        duration = int(np.random.randint(15, 26))
    else:
        duration = int(np.random.randint(4, 9))

    # Randomize battery discharge severity
    discharge_rate = None

    if selected_incident == "battery_discharge":

        discharge_rate = round(np.random.uniform(0.3, 0.7), 3)

        robots_discharge_rates[robot_id] = discharge_rate

    # Reserve recovery time after unexpected stops
    recovery_duration = 5 if selected_incident == "unexpected_stop" else 0

    latest_start = simulation_seconds - duration - recovery_duration

    # Randomly select incident start time
    start_second = int(np.random.randint(10, latest_start + 1))

    # Assign the incident to consecutive seconds
    for second in range(start_second, start_second + duration):

        incident_schedules[robot_id][second] = selected_incident

    # Schedule recovery after an unexpected stop
    if selected_incident == "unexpected_stop":

        for second in range(
            start_second + duration,
            start_second + duration + recovery_duration
        ):

            incident_schedules[robot_id][second] = "recovery"

    # Save actual incident information
    incident_log.append({
        "robot_id": robot_id,
        "incident_type": selected_incident,
        "start_second": start_second,
        "duration": duration,
        "discharge_rate": discharge_rate
    })


# ============================================================
# 5. GENERATE TELEMETRY FOR ALL ROBOTS
# ============================================================

telemetry_records = []

for robot_id, battery_level in robots.items():

    # Each robot starts in Idle mode
    previous_status = "Idle"

    # Track whether the robot has just recovered
    just_recovered = False

    for i in range(simulation_seconds):

        # Generate timestamp
        timestamp = start_time + timedelta(seconds=i)

        # Check whether an incident is active
        incident_type = incident_schedules[robot_id][i]

        # Determine status based on incident or normal operation
        if incident_type == "unexpected_stop":

            status = "Unexpected Stop"

        elif incident_type == "recovery":

            status = "Recovering"
            just_recovered = True

        elif incident_type == "wheel_slip":

            status = "Moving"

        elif just_recovered:

            # Force one Idle second after recovery
            status = "Idle"
            just_recovered = False

        else:

            # Normal operation uses transition probabilities
            status = get_next_status(previous_status)

        # Generate telemetry measurements
        robot_data = generate_telemetry(
            robot_id,
            timestamp,
            battery_level,
            status,
            incident_type=incident_type
        )

        # Save telemetry record
        telemetry_records.append(robot_data)

        # Update previous status for next iteration
        if incident_type == "unexpected_stop":

            previous_status = "Idle"

        elif incident_type == "recovery":

            previous_status = "Idle"

        else:

            previous_status = status

        # Update battery level
        if incident_type == "battery_discharge":

            battery_level -= robots_discharge_rates[robot_id]

        elif status == "Moving":

            battery_level -= 0.01

        elif status == "Idle":

            battery_level -= 0.002

        elif status == "Charging":

            battery_level += 0.02

        # Keep battery between 0% and 100%
        battery_level = float(round(
            np.clip(battery_level, 0, 100), 3
        ))

    # Save final battery level for this robot
    robots[robot_id] = battery_level


# ============================================================
# 6. SAVE TELEMETRY AND INCIDENT DATASETS
# ============================================================

# Convert telemetry records into a DataFrame
df = pd.DataFrame(telemetry_records)

# Save telemetry dataset
df.to_csv("data/robot_telemetry.csv", index=False)

# Convert incident log into a DataFrame
incident_df = pd.DataFrame(
    incident_log,
    columns=[
        "robot_id",
        "incident_type",
        "start_second",
        "duration",
        "discharge_rate"
    ]
)

# Save incident log
incident_df.to_csv("data/incident_log.csv", index=False)


# ============================================================
# 7. DISPLAY SIMULATION RESULTS
# ============================================================

print("\nTelemetry generated successfully!")

print(f"Total telemetry records: {len(df)}")

print("\nIncident summary:")

if incident_df.empty:

    print("No incidents generated in this simulation.")

else:

    print(incident_df.to_string(index=False))

print("\nFinal battery levels:")

for robot_id, battery_level in robots.items():

    print(f"{robot_id}: {battery_level}%")
