from datetime import datetime
import numpy as np
from datetime import datetime, timedelta

def generate_telemetry(timestamp,batterylevel):
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

    


    # Generate one simulated telemetry record
    robot_data = {
        "timestamp": timestamp,
        "robot_id": "AMR-01",
        "velocity": velocity,
        "left_wheel_speed": left_wheel_speed,
        "right_wheel_speed": right_wheel_speed,
        "motor_current": motor_current,
        "battery_level": battery_level,
        "status": status
    }
    return robot_data



# Starting timestamp
start_time = datetime.now()

battery_level = 85.0

# Generate 10 telemetry records, one second apart
for i in range(100):
    timestamp = start_time + timedelta(seconds=i)

    robot_data = generate_telemetry(timestamp,battery_level)
    
    # Update battery level based on robot status
    if robot_data["status"] == "Moving":
        battery_level -= 0.01

    elif robot_data["status"] == "Idle":
        battery_level -= 0.002

    elif robot_data["status"] == "Charging":
        battery_level += 0.02

    # Keep battery between 0% and 100%
    battery_level = float(round(np.clip(battery_level, 0, 100), 3))

    print(f"Record {i + 1}: {robot_data}")