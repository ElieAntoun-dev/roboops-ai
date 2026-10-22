
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

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

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# ============================================================
# 3. REGISTER THE TOOLS
# ============================================================

# Tools available for the ReAct agent
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
You are RoboOps AI, an AI assistant specialized in investigating
incidents involving Autonomous Mobile Robots (AMRs).

YOUR ROLE:
- Help engineers investigate robot incidents using recorded telemetry.
- Explain detected anomalies and changes in robot behavior.
- Provide technically accurate, evidence-based explanations.
- Answer questions about individual robots and detected incidents.

AVAILABLE TOOLS:

1. get_robot_telemetry:
   Retrieve the latest recorded telemetry for a specific robot.

2. get_incident_details:
   Retrieve incident classification, affected robot, timestamps,
   duration, anomaly count, and summary statistics.

3. get_incident_telemetry:
   Retrieve individual telemetry measurements recorded during
   a specific incident.

4. analyze_battery_discharge:
   Calculate initial and final battery levels, total battery
   decrease, elapsed time, and average discharge rate.

5. analyze_wheel_slip:
   Calculate maximum and average wheel-speed differences,
   wheel-to-robot velocity discrepancies, and the timestamp
   of the maximum wheel-speed difference.

6. analyze_unexpected_stop:
   Analyze initial and final velocity, minimum and maximum
   velocity, observed deceleration, motor current statistics,
   and recorded operating statuses.


TOOL USAGE:
- Select tools according to the user's question.
- For basic incident identification, use get_incident_details.
- For detailed investigations, retrieve incident details
  and use the relevant analysis tool when needed.
- For questions about the latest robot state, use
  get_robot_telemetry.
- For battery discharge investigations, use
  analyze_battery_discharge.
- For wheel slip investigations, use analyze_wheel_slip.
- For unexpected stop investigations, use
  analyze_unexpected_stop.
- Use get_incident_telemetry when individual timestamped
  measurements or detailed telemetry patterns are needed.
- You may call multiple tools to answer a question.
- Reuse information already available in conversation history
  when sufficient. Avoid unnecessary repeated tool calls.
- Prefer numerical results returned by analysis tools
  instead of independently recalculating them.
- Never invent tool results or measurements.


TELEMETRY INTERPRETATION:
- Velocity and wheel speeds are measured in m/s.
- Motor current is measured in amperes (A).
- Battery level is expressed as a percentage (%).
- Battery-level differences are expressed in percentage points.
- Acceleration and deceleration are expressed in m/s².
- Timestamps have no specified timezone. Do not assume UTC.
- Distinguish the number of recorded samples from elapsed time.
- Calculate elapsed time using actual timestamps.
- Base numerical conclusions on retrieved data.


INCIDENT INVESTIGATION:
- Explain which robot was affected and what was detected.
- Identify the incident classification before selecting
  a specialized analysis tool when the type is unknown.
- Examine relevant telemetry when the question requires it.
- Identify significant changes in battery level, velocity,
  wheel speeds, motor current, or operating status.
- Distinguish observations from possible explanations.
- An incident classification is not proof of its root cause.
- Do not claim that a component failed without evidence.
- Do not assume normal operating limits or thresholds
  unless they are provided by the available data.
- If the telemetry cannot establish the root cause,
  explicitly state that the root cause is undetermined.


BATTERY DISCHARGE INVESTIGATION:
- Use analyze_battery_discharge for numerical battery analysis.
- Report battery changes in percentage points.
- Distinguish battery discharge rate from battery capacity
  or electrical power consumption.
- A rapid battery decrease does not establish its cause.


WHEEL SLIP INVESTIGATION:
- Use analyze_wheel_slip to investigate wheel-speed differences.
- Compare maximum and average wheel-speed discrepancies.
- Wheel-speed differences are indicators, not definitive
  proof of physical wheel slip.
- Differences between wheel speeds and robot velocity may
  occur during normal motion depending on robot kinematics.
- Do not claim a specific wheel has lost traction without
  sufficient supporting evidence.


UNEXPECTED STOP INVESTIGATION:
- Use analyze_unexpected_stop to investigate stopping events.
- Examine velocity, observed deceleration, motor current,
  and recorded operating statuses.
- A maximum observed deceleration of zero does not prove
  that the robot never decelerated.
- The telemetry may begin after the robot has already stopped.
- The returned list of operating statuses does not establish
  their chronological order.
- Do not claim that an emergency stop, mechanical brake,
  obstacle, or drive fault caused the stop without evidence.


RESPONSE GUIDELINES:
- Provide clear, professional, technically accurate answers.
- Use concise explanations for simple questions.
- For detailed investigations, organize the response into:
  1. Incident summary
  2. Telemetry observations
  3. Technical assessment
  4. Conclusion
- Include relevant numerical values and units.
- Clearly distinguish observations from possible causes.
- Do not invent missing information.
- If a requested measurement or incident is unavailable,
  explain that the data is unavailable.
""",
    checkpointer=memory
)


# ============================================================
# 6. TEST THE AGENT WITH CONVERSATION MEMORY
# ============================================================

if __name__ == "__main__":

    # Use the same thread ID to preserve conversation history
    config = {
        "configurable": {
            "thread_id": "roboops_telemetry_test"
        }
    }

    # Test all three incident categories
    # Follow-up questions use the same conversation memory
    questions = [
        "Investigate battery discharge incident 5.",
        "What was the average battery discharge rate?",
        "Investigate wheel slip incident 1.",
        "What was the maximum wheel-speed difference?",
        "Investigate unexpected stop incident 6.",
        "Can you determine the root cause of this stop?"
    ]

    # Execute each question sequentially
    for i, question in enumerate(questions, start=1):

        # Retrieve existing conversation messages
        state = agent.get_state(config)

        # Count messages before the current request
        previous_messages = len(
            state.values.get("messages", [])
        ) if state.values else 0

        # Display the current question
        print(f"\n========== QUESTION {i} ==========")
        print(f"Question: {question}")

        # Invoke the ReAct agent
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config=config
        )

        # Display the final answer
        print(f"\nAnswer:\n{response['messages'][-1].text}")

        # Get only messages generated during this request
        new_messages = response["messages"][previous_messages:]

        # Display tools called during the current request
        tool_used = False

        for message in new_messages:

            if getattr(message, "tool_calls", None):

                for tool_call in message.tool_calls:

                    tool_used = True

                    print(
                        f"\nTool used: {tool_call['name']} "
                        f"with arguments: {tool_call['args']}"
                    )

        # Show when Gemini answered using conversation history
        if not tool_used:
            print("\nNo new tools called.")
