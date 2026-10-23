
# Import the Pydantic model we created
from agent.schemas import IncidentReport
from pydantic import ValidationError


# 1. Create a sample incident report
report = IncidentReport(
    incident_id=6,
    robot_id="AMR-02",
    incident_type="unexpected_stop",

    start_time="2026-10-18T12:37:38.325324",
    end_time="2026-10-18T12:37:46.325324",

    summary=(
        "AMR-02 experienced an unexpected stop and remained "
        "stationary throughout the recorded incident window."
    ),

    observations=[
        "Robot velocity remained at 0 m/s.",
        "Maximum observed deceleration was 0 m/s².",
        "Average motor current was 6.55 A.",
        "Maximum motor current was 14.09 A.",
        "Recorded statuses included Unexpected Stop and Recovering."
    ],

    technical_assessment=(
        "The robot was already stationary at the beginning "
        "of the incident window. The telemetry does not "
        "capture the deceleration leading up to the stop."
    ),

    possible_causes=[],

    root_cause_confirmed=False,
    root_cause=None,

    recommended_actions=[
        "Inspect telemetry recorded before the incident.",
        "Review robot event logs.",
        "Check safety-system diagnostics."
    ],

    investigation_status="inconclusive"
)


print("\n=== INVALID DATA TEST ===")

# Start with the valid report and change one field
invalid_data = report.model_dump()

# Assign a value not allowed by our Literal definition
invalid_data["investigation_status"] = "pending"

try:
    # Attempt to create a report with invalid data
    invalid_report = IncidentReport(**invalid_data)

except ValidationError as error:
    # Display the validation error instead of crashing
    print("Validation failed as expected!")
    print(error)
