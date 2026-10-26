from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Import the Pydantic incident report schema
from agent.schemas import IncidentReport

# Import the source-data verification function
from agent.verify_report import verify_incident_report

# Import the six existing tools
from agent.tools import (
    get_robot_telemetry,
    get_incident_details,
    get_incident_telemetry,
    analyze_battery_discharge,
    analyze_wheel_slip,
    analyze_unexpected_stop
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load the Gemini API key from the .env file
load_dotenv()


# ============================================================
# 2. INITIALIZE THE LLM
# ============================================================

# Initialize Gemini for incident investigations
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=0,
    timeout=120
)


# ============================================================
# 3. REGISTER THE TOOLS
# ============================================================

# Tools available to the ReAct agent
tools = [
    get_robot_telemetry,
    get_incident_details,
    get_incident_telemetry,
    analyze_battery_discharge,
    analyze_wheel_slip,
    analyze_unexpected_stop
]


# ============================================================
# 4. INITIALIZE CONVERSATION MEMORY
# ============================================================

# Store conversation history while the application is running
memory = InMemorySaver()


# ============================================================
# 5. CREATE THE REACT AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are RoboOps AI, an engineering assistant specialized in
investigating Autonomous Mobile Robot (AMR) incidents using
recorded telemetry.

YOUR RESPONSIBILITIES:
- Investigate robot incidents and explain anomalous behavior.
- Provide accurate, evidence-based technical assessments.
- Distinguish measured observations from possible causes.
- Never invent telemetry, operating limits, or root causes.

AVAILABLE TOOLS:

1. get_robot_telemetry:
   Retrieve the latest telemetry for a specific robot.

2. get_incident_details:
   Retrieve incident classification, robot ID, timestamps,
   duration, anomaly count, and summary statistics.

3. get_incident_telemetry:
   Retrieve timestamped telemetry samples for an incident.

4. analyze_battery_discharge:
   Retrieve incident context and analyze battery levels,
   discharge rate, average velocity, and motor current.

5. analyze_wheel_slip:
   Retrieve incident context and analyze wheel-speed
   discrepancies, robot velocity, and motor current.

6. analyze_unexpected_stop:
   Retrieve incident context and analyze velocity, observed
   deceleration, motor current, and operating statuses.

TOOL SELECTION:
- Select tools based on the user's question.
- For basic incident information, use get_incident_details.
- If the incident type is unknown, identify it before
  selecting a specialized analysis tool.
- For a known incident type, call its specialized analysis
  tool directly. It already returns incident information.
- Do not call get_incident_details separately when the
  specialized tool provides sufficient information.
- Use get_incident_telemetry only when individual samples
  or detailed telemetry patterns are required.
- Use get_robot_telemetry for the latest robot state.
- Call additional tools only when necessary.
- Reuse sufficient information from conversation history.
- Prefer numerical results calculated by the tools.

TELEMETRY RULES:
- Velocity and wheel speeds: m/s.
- Motor current: A.
- Battery level: %; battery changes: percentage points.
- Acceleration and deceleration: m/s².
- Timestamps have no specified timezone; do not assume UTC.
- Distinguish elapsed time from the number of samples.
- Base numerical conclusions on retrieved measurements.

INVESTIGATION RULES:
- Identify the affected robot, incident type, and timeframe.
- Report relevant telemetry measurements and anomalies.
- Distinguish observations, hypotheses, and confirmed facts.
- Incident classifications do not establish root causes.
- Do not assume thresholds, failures, or safety events
  without supporting evidence.
- If the root cause cannot be established, state that
  it remains undetermined.

INCIDENT-SPECIFIC RULES:

Battery discharge:
- Report battery changes in percentage points.
- Do not confuse discharge rate with power consumption.
- A rapid battery decrease does not establish its cause.

Wheel slip:
- Compare maximum and average wheel-speed discrepancies.
- Wheel-speed differences do not independently prove slip.
- Robot kinematics may explain differences between wheel
  speeds and robot velocity.
- Do not identify a failed wheel without sufficient evidence.

Unexpected stop:
- Examine velocity, deceleration, motor current, and statuses.
- Zero observed deceleration does not prove no stop occurred;
  the robot may already have stopped before recording began.
- Recorded statuses are not necessarily chronological.
- Do not attribute stops to brakes, obstacles, safety systems,
  or drive faults without evidence.

STRUCTURED REPORT RULES:
- For incident investigations, populate the IncidentReport
  schema using information returned by the tools.
- Copy incident IDs, robot IDs, incident types, and timestamps
  exactly from the retrieved incident information.
- Never calculate or invent timestamps.
- Ensure end_time is not earlier than start_time.

NUMERICAL MEASUREMENT RULES:
- Populate measurements.values using numerical results
  returned by the specialized analysis tool.
- Use the exact measurement names specified below.
- Include every measurement listed for the incident type.
- Copy numerical values directly from the tool output.
- Do not recalculate or estimate these measurements.
- Do not include unsupported measurement names.
- Store only numerical values, without units.
- Do not confuse average and maximum measurements.
- Preserve the precision returned by the analysis tool.

For battery_discharge, use these mappings:
- initial_battery_level:
  initial_battery_percent
- final_battery_level:
  final_battery_percent
- total_battery_drop:
  total_battery_drop_percentage_points
- average_discharge_rate:
  average_discharge_rate_percentage_points_per_second
- average_velocity:
  average_velocity_mps
- average_motor_current:
  average_motor_current_A

For wheel_slip, use these mappings:
- maximum_wheel_difference:
  maximum_wheel_speed_difference_mps
- average_wheel_difference:
  average_wheel_speed_difference_mps
- average_velocity:
  average_velocity_mps
- average_motor_current:
  average_motor_current_A

For unexpected_stop, use these mappings:
- maximum_observed_deceleration:
  maximum_observed_deceleration_mps2
- average_velocity:
  average_velocity_mps
- average_motor_current:
  average_motor_current_A
- maximum_motor_current:
  maximum_motor_current_A

ADDITIONAL STRUCTURED REPORT RULES:
- If structured output validation fails, correct the fields
  using the original tool results.
- Include relevant numerical observations and units in the
  observations section.
- Only mark root_cause_confirmed as true when supported
  by sufficient evidence.
- If the root cause is undetermined, set:
  root_cause_confirmed = false
  root_cause = null
  investigation_status = "inconclusive"
- Possible causes must be clearly identified as hypotheses.
- If no supported hypotheses exist, use an empty list.
- Recommend practical engineering investigation steps.
- Do not infer the chronological order of operating statuses
  from an unordered list.

RESPONSE FORMAT:
- Provide concise, technically accurate investigations.
- Clearly distinguish observations from interpretations.
- Include incident summary, telemetry observations,
  technical assessment, and recommended actions.
- Communicate uncertainty and missing information.
""",

    # Generate a Pydantic report and retry on validation errors
    response_format=ToolStrategy(
        schema=IncidentReport,
        handle_errors=False
    ),

    # Preserve conversation memory
    checkpointer=memory
)


# ============================================================
# 6. TEST THE AGENT WITH STREAMING
# ============================================================

if __name__ == "__main__":

    # Use a fresh conversation thread for this test
    config = {
        "configurable": {
            "thread_id": "roboops_stream_debug_1"
        },
        "recursion_limit": 15
    }

    # Test only the first incident for now
    questions = [
    "Investigate wheel slip incident 1.",
    "Investigate battery discharge incident 5.",
    "Investigate unexpected stop incident 6."
    ]

    # Execute each question sequentially
    for i, question in enumerate(questions, start=1):

        print(f"\n========== QUESTION {i} ==========", flush=True)
        print(f"Question: {question}", flush=True)
        print("\nStarting agent investigation...", flush=True)

        # ====================================================
        # 7. STREAM THE AGENT EXECUTION
        # ====================================================

        try:

            # Stream completed steps instead of waiting
            # for the entire investigation to finish
            for step in agent.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                },
                config=config,
                stream_mode="updates"
            ):

                # Each step contains the node name and its output
                for node_name, update in step.items():

                    print(
                        f"\nCompleted step: {node_name}",
                        flush=True
                    )

                    if not isinstance(update, dict):
                        continue

                    # Display messages generated by this step
                    for message in update.get("messages", []):

                        # Check whether Gemini requested a tool
                        if getattr(message, "tool_calls", None):

                            for tool_call in message.tool_calls:

                                print(
                                    f"Tool called: {tool_call['name']}",
                                    flush=True
                                )

                                print(
                                    f"Arguments: {tool_call['args']}",
                                    flush=True
                                )

                        # Display tool execution results
                        if getattr(message, "type", None) == "tool":

                            print(
                                f"Tool completed: {message.name}",
                                flush=True
                            )

                            print(
                                f"Tool result: {str(message.content)[:500]}",
                                flush=True
                            )

                    # Indicate when a structured report is generated
                    if update.get("structured_response") is not None:

                        print(
                            "Structured incident report generated.",
                            flush=True
                        )

            print(
                "\nAgent investigation finished.",
                flush=True
            )

        except Exception as error:

            print(
                f"\nAGENT EXECUTION FAILED: {type(error).__name__}",
                flush=True
            )

            print(str(error), flush=True)

            # Skip report verification if execution failed
            continue

        # ====================================================
        # 8. RETRIEVE THE FINAL AGENT STATE
        # ====================================================

        # Streaming updates do not directly return the final
        # response dictionary, so retrieve it from memory
        state = agent.get_state(config)

        response = state.values

        # Retrieve the Pydantic-validated report
        structured_report = response.get("structured_response")

        # ====================================================
        # 9. VERIFY AND DISPLAY THE INCIDENT REPORT
        # ====================================================

        if structured_report is not None:

            try:

                # Independently verify the report against
                # the original incident data
                verified = verify_incident_report(
                    structured_report
                )

            except ValueError as error:

                print("\nREPORT VERIFICATION FAILED:")
                print(error)

                continue

            # Display the report only after verification
            if verified:

                print("\n=== VERIFIED INCIDENT REPORT ===")

                print(
                    structured_report.model_dump_json(indent=2)
                )

                print(
                    "\nSource-data verification: PASSED"
                )

        else:

            print("\nNo structured report was returned.")

            # Display the last message if available
            if response.get("messages"):

                last_message = response["messages"][-1]

                print(
                    "\nLast agent message:",
                    str(last_message.content)[:1000]
                )