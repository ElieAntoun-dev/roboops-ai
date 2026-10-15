from datetime import datetime
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def generate_telemetry(robot_id, timestamp, battery_level,incident=False):
    # Simulate the robot's operational status
    status = str(np.random.choice(
        ["Moving", "Idle", "Charging"],
        p=[0.7, 0.2, 0.1]
    ))
    ## Generate velocity based on robot status
    if status == "Moving":
        velocity = round(np.random.normal(1.2, 0.02), 3)
    else:
        velocity = 0.0

    # Generate wheel speeds
    if status == "Moving":
        wheel_difference = np.random.normal(0, 0.01)

        left_wheel_speed = round(velocity - wheel_difference, 3)
        right_wheel_speed = round(velocity + wheel_difference, 3)
    else:
        left_wheel_speed = 0.0
        right_wheel_speed = 0.0

    # Generate drive motor current based on robot status
    if status == "Moving":
        motor_current = round(np.random.normal(8.5, 0.3), 2)

    elif status == "Idle":
        motor_current = round(np.random.normal(0.5, 0.1), 2)

    else:  # Charging
        motor_current = 0.0

    # Simulate an unexpected robot stop
    if incident:
        status = "Unexpected Stop"
        velocity = 0.0
        left_wheel_speed = 0.0
        right_wheel_speed = 0.0
        motor_current = 12.0

    # Generate one simulated telemetry record
    robot_data = {
        "timestamp": timestamp,
        "robot_id": robot_id,
        "velocity": velocity,
        "left_wheel_speed": left_wheel_speed,
        "right_wheel_speed": right_wheel_speed,
        "motor_current": motor_current,
        "battery_level": battery_level,
        "status": status
    }
    return robot_data

telemetry_records = []

# Starting timestamp
start_time = datetime.now()

# Define our robot fleet and initial battery levels
robots = {
    "AMR-01": 85.0,
    "AMR-02": 70.0,
    "AMR-03": 95.0,
    "AMR-04": 60.0,
    "AMR-05": 90.0
}

# Generate 100 measurements for each robot
for robot_id, battery_level in robots.items():

    for i in range(100):
        timestamp = start_time + timedelta(seconds=i)

        # Trigger an unexpected stop for AMR-03 at second 50
        incident = robot_id == "AMR-03" and i == 50

        robot_data = generate_telemetry(
        robot_id, timestamp, battery_level, incident=incident
    )

        telemetry_records.append(robot_data)

        # Update battery level based on status
        if robot_data["status"] == "Moving":
            battery_level -= 0.01

        elif robot_data["status"] == "Idle":
            battery_level -= 0.002

        elif robot_data["status"] == "Charging":
            battery_level += 0.02

        battery_level = float(round(np.clip(battery_level, 0, 100), 3))

    # Save the updated battery level for this robot
    robots[robot_id] = battery_level

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(telemetry_records)

# Display the first five records
print(df.head())
# Save the DataFrame to a CSV file
df.to_csv("data/robot_telemetry.csv", index=False)

print(f"Total records: {len(df)}")
print(df["robot_id"].value_counts())
print(df[df["status"] == "Unexpected Stop"])

print("Telemetry successfully saved to CSV!")