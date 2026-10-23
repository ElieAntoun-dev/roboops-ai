
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
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
            "Numerical telemetry measurements from the tools. "
            "Use descriptive measurement names and numeric values. "
            "Do not invent measurements."
        )
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

    # NEW: Structured numerical measurements
    measurements: IncidentMeasurements = Field(
        description="Numerical telemetry measurements used in the investigation"
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

    # 7. Validate consistency between root cause fields
    @model_validator(mode="after")
    def validate_root_cause(self) -> Self:

        if self.root_cause_confirmed and not self.root_cause:
            raise ValueError(
                "A confirmed root cause must have a description."
            )

        if not self.root_cause_confirmed and self.root_cause is not None:
            raise ValueError(
                "Root cause must be None when it is not confirmed."
            )

        return self

    # 8. Validate incident timestamps
    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:

        try:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)

        except ValueError:
            raise ValueError(
                "Incident timestamps must be valid ISO 8601 values."
            )

        if end < start:
            raise ValueError(
                "Incident end_time cannot be earlier than start_time."
            )

        return self
