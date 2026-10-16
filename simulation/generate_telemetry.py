
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# 1. SELECT THE NEXT NORMAL ROBOT STATUS
# ============================================================

def get_next_status(previous_status):

    status_transitions = {
        "Moving": {"Moving": 0.95, "Idle": 0.05},
        "Idle": {"Moving": 0.15, "Idle": 0.80, "Charging": 0.05},
        "Charging": {"Charging": 0.95, "Idle": 0.05}
    }

    possible_statuses = status_transitions[previous_status]

    return str(np.random.choice(
        list(possible_statuses.keys()),
        p=list(possible_statuses.values())
    ))


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

    # Generate normal measurements
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

        motor_current = (
            round(np.random.normal(0.5, 0.1), 2)
            if status == "Idle" else 0.0
        )

    # Override measurements during incidents
    if incident_type == "unexpected_stop":

        velocity = left_wheel_speed = right_wheel_speed = 0.0
        motor_current = round(np.random.uniform(10, 15), 2)

    elif incident_type == "recovery":

        velocity = left_wheel_speed = right_wheel_speed = 0.0
        motor_current = 2.0

    elif incident_type == "wheel_slip":

        velocity = round(np.random.uniform(0.4, 0.8), 3)
        left_wheel_speed = round(np.random.uniform(1.5, 2.0), 3)
        right_wheel_speed = round(np.random.uniform(0.6, 1.0), 3)
        motor_current = round(np.random.uniform(12, 16), 2)

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
# 3. UPDATE ROBOT BATTERY LEVEL
# ============================================================

def update_battery_level(
    battery_level,
    status,
    incident_type,
    discharge_rate=None
):

    if incident_type == "battery_discharge":
        battery_level -= discharge_rate

    elif status == "Moving":
        battery_level -= 0.01

    elif status == "Idle":
        battery_level -= 0.002

    elif status == "Charging":
        battery_level += 0.02

    return float(round(np.clip(battery_level, 0, 100), 3))


# ============================================================
# 4. CREATE INCIDENT SCHEDULES
# ============================================================

def create_incident_schedules(robots, simulation_seconds, incident_types):

    # One incident state per second for each robot
    incident_schedules = {
        robot_id: [None] * simulation_seconds
        for robot_id in robots
    }

    # Store battery discharge rates per second
    discharge_schedules = {
        robot_id: [None] * simulation_seconds
        for robot_id in robots
    }

    incident_log = []

    for robot_id in robots:

        # Track which seconds are already reserved
        occupied = np.zeros(simulation_seconds, dtype=bool)

        # Generate 3 to 5 incidents per robot
        number_of_incidents = int(np.random.randint(3, 6))

        for _ in range(number_of_incidents):

            incident_type = str(np.random.choice(incident_types))

            # Determine incident duration
            duration = (
                int(np.random.randint(15, 26))
                if incident_type == "battery_discharge"
                else int(np.random.randint(4, 9))
            )

            # Reserve recovery and Idle time after unexpected stops
            recovery_duration = 5 if incident_type == "unexpected_stop" else 0
            idle_duration = 1 if incident_type == "unexpected_stop" else 0

            total_duration = duration + recovery_duration + idle_duration

            # Find available starting seconds
            available_starts = [
                start
                for start in range(10, simulation_seconds - total_duration + 1)
                if not occupied[start:start + total_duration].any()
            ]

            if not available_starts:
                continue

            start_second = int(np.random.choice(available_starts))
            end_second = start_second + duration

            # Reserve the complete time window
            occupied[start_second:start_second + total_duration] = True

            # Assign incident states
            incident_schedules[robot_id][start_second:end_second] = (
                [incident_type] * duration
            )

            # Assign battery discharge rate
            discharge_rate = None

            if incident_type == "battery_discharge":

                discharge_rate = round(np.random.uniform(0.3, 0.7), 3)

                discharge_schedules[robot_id][start_second:end_second] = (
                    [discharge_rate] * duration
                )

            # Schedule recovery and Idle after unexpected stops
            if incident_type == "unexpected_stop":

                recovery_end = end_second + recovery_duration

                incident_schedules[robot_id][end_second:recovery_end] = (
                    ["recovery"] * recovery_duration
                )

                incident_schedules[robot_id][recovery_end] = (
                    "post_recovery_idle"
                )

            # Save ground-truth incident
            incident_log.append({
                "robot_id": robot_id,
                "incident_type": incident_type,
                "start_second": start_second,
                "duration": duration,
                "discharge_rate": discharge_rate
            })

    return incident_schedules, discharge_schedules, incident_log


# ============================================================
# 5. RUN THE ROBOT FLEET SIMULATION
# ============================================================

def run_simulation(robots, simulation_seconds, incident_schedules,
                   discharge_schedules):

    telemetry_records = []
    start_time = datetime.now()
    final_battery_levels = {}

    for robot_id, initial_battery in robots.items():

        battery_level = initial_battery
        previous_status = "Idle"

        for second in range(simulation_seconds):

            timestamp = start_time + timedelta(seconds=second)
            incident_type = incident_schedules[robot_id][second]

            # Determine current robot status
            if incident_type == "unexpected_stop":
                status = "Unexpected Stop"

            elif incident_type == "recovery":
                status = "Recovering"

            elif incident_type == "post_recovery_idle":
                status = "Idle"

            elif incident_type == "wheel_slip":
                status = "Moving"

            else:
                status = get_next_status(previous_status)

            # Generate and store telemetry
            telemetry_records.append(
                generate_telemetry(
                    robot_id,
                    timestamp,
                    battery_level,
                    status,
                    incident_type
                )
            )

            # Update previous status
            previous_status = (
                "Idle"
                if incident_type in (
                    "unexpected_stop",
                    "recovery",
                    "post_recovery_idle"
                )
                else status
            )

            # Update battery level
            battery_level = update_battery_level(
                battery_level,
                status,
                incident_type,
                discharge_schedules[robot_id][second]
            )

        final_battery_levels[robot_id] = battery_level

    return telemetry_records, final_battery_levels


# ============================================================
# 6. DEFINE SIMULATION SETTINGS
# ============================================================

robots = {
    "AMR-01": 85.0,
    "AMR-02": 70.0,
    "AMR-03": 95.0,
    "AMR-04": 60.0,
    "AMR-05": 90.0
}

simulation_seconds = 1000

incident_types = [
    "unexpected_stop",
    "wheel_slip",
    "battery_discharge"
]


# ============================================================
# 7. EXECUTE SIMULATION AND SAVE RESULTS
# ============================================================

if __name__ == "__main__":

    # Create incident schedules
    incident_schedules, discharge_schedules, incident_log = (
        create_incident_schedules(
            robots,
            simulation_seconds,
            incident_types
        )
    )

    # Generate robot telemetry
    telemetry_records, final_battery_levels = run_simulation(
        robots,
        simulation_seconds,
        incident_schedules,
        discharge_schedules
    )

    # Save telemetry dataset
    df = pd.DataFrame(telemetry_records)
    df.to_csv("data/robot_telemetry.csv", index=False)

    # Save ground-truth incident dataset
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

    incident_df = incident_df.sort_values(
        by=["robot_id", "start_second"]
    )

    incident_df.to_csv("data/incident_log.csv", index=False)

    # Display results
    print("\nTelemetry generated successfully!")
    print(f"Total telemetry records: {len(df)}")
    print(f"Total incidents generated: {len(incident_df)}")

    print("\nIncident summary:")
    print(incident_df.to_string(index=False))

    print("\nFinal battery levels:")

    for robot_id, battery_level in final_battery_levels.items():
        print(f"{robot_id}: {battery_level}%")
