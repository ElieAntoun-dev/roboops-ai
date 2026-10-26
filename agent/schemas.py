import json
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Literal, Dict
from typing_extensions import Self


# ============================================================
# 1. DEFINE NUMERICAL MEASUREMENTS
# ============================================================

class IncidentMeasurements(BaseModel):

    # Numerical measurements returned by the analysis tools
    # Examples:
    # average_motor_current: 8.5394
    # total_battery_drop: 9.945
    # maximum_wheel_difference: 1.356

    values: Dict[str, float] = Field(
        description=(
            "A JSON object containing measurement names as keys "
            "and numerical measurements as values. "
            "This field MUST be an object/dictionary, NOT a string. "
            "Do not serialize the dictionary into JSON text. "
            "For wheel_slip, use exactly these keys: "
            "maximum_wheel_difference, average_wheel_difference, "
            "average_velocity, average_motor_current. "
            "Example: "
            '{"maximum_wheel_difference": 1.356, '
            '"average_wheel_difference": 0.7694, '
            '"average_velocity": 0.7148, '
            '"average_motor_current": 13.0862}. '
            "For other incident types, use the exact keys "
            "specified in the system prompt. "
            "Use numerical values returned by the analysis tools. "
            "Never invent measurements."
        )
    )

    # Convert JSON strings into dictionaries when necessary
    @field_validator("values", mode="before")
    @classmethod
    def parse_measurements(cls, value):

        # If Gemini returns a dictionary, keep it unchanged
        if isinstance(value, dict):
            return value

        # If Gemini returns a JSON string, convert it
        if isinstance(value, str):

            try:
                parsed = json.loads(value)

            except json.JSONDecodeError:
                raise ValueError(
                    "Measurements must be a valid JSON object."
                )

            # Ensure the parsed result is a dictionary
            if not isinstance(parsed, dict):
                raise ValueError(
                    "Measurements must contain a dictionary."
                )

            return parsed

        # Reject unsupported formats
        raise ValueError(
            "Measurements must be a dictionary or JSON object string."
        )


# ============================================================
# 2. DEFINE THE INCIDENT REPORT
# ============================================================

class IncidentReport(BaseModel):

    # 1. Incident identification
    incident_id: int = Field(
        description="Unique identifier of the incident"
    )

    robot_id: str = Field(
        description="Identifier of the affected robot"
    )

    incident_type: str = Field(
        description="Classification of the incident"
    )

    # 2. Incident timeframe
    start_time: str = Field(
        description="Incident start timestamp"
    )

    end_time: str = Field(
        description="Incident end timestamp"
    )

    # 3. Investigation results
    summary: str = Field(
        description="Brief summary of the incident"
    )

    observations: List[str] = Field(
        description="Telemetry observations supported by data"
    )

    # Structured numerical measurements
    measurements: IncidentMeasurements = Field(
        description=(
            "Numerical telemetry measurements used "
            "in the investigation"
        )
    )

    technical_assessment: str = Field(
        description="Engineering interpretation of the observations"
    )

    possible_causes: List[str] = Field(
        description="Possible explanations, not confirmed causes"
    )

    # 4. Root cause information
    root_cause_confirmed: bool = Field(
        description="Whether the root cause is supported by evidence"
    )

    root_cause: Optional[str] = Field(
        default=None,
        description="Confirmed root cause, or None if undetermined"
    )

    # 5. Recommended investigation actions
    recommended_actions: List[str] = Field(
        description="Recommended next investigation steps"
    )

    # 6. Investigation status
    investigation_status: Literal[
        "completed",
        "inconclusive",
        "insufficient_data"
    ] = Field(
        description="Status of the investigation"
    )

    # ========================================================
    # 3. VALIDATE ROOT CAUSE CONSISTENCY
    # ========================================================

    @model_validator(mode="after")
    def validate_root_cause(self) -> Self:

        # A confirmed root cause must have a description
        if self.root_cause_confirmed and not self.root_cause:
            raise ValueError(
                "A confirmed root cause must have a description."
            )

        # An unconfirmed root cause must be None
        if not self.root_cause_confirmed and self.root_cause is not None:
            raise ValueError(
                "Root cause must be None when it is not confirmed."
            )

        return self

    # ========================================================
    # 4. VALIDATE INCIDENT TIMESTAMPS
    # ========================================================

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:

        try:
            # Convert ISO timestamp strings to datetime objects
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)

        except ValueError:
            raise ValueError(
                "Incident timestamps must be valid ISO 8601 values."
            )

        # Ensure the incident end time follows the start time
        if end < start:
            raise ValueError(
                "Incident end_time cannot be earlier than start_time."
            )

        return self