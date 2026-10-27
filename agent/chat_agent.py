
# ============================================================
# 1. IMPORTS
# ============================================================

import json
from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Reuse the exact Gemini configuration and tools from Agent 1
from agent.agent import llm, tools


# ============================================================
# 2. CHATBOT SYSTEM PROMPT
# ============================================================

CHAT_SYSTEM_PROMPT = """
You are RoboOps AI, a technical investigation assistant
specializing in Autonomous Mobile Robots (AMRs).

Your role is to answer follow-up questions about an incident
after the initial investigation report has been generated.

INCIDENT CONTEXT:
- The engineer provides a verified incident report at the
  beginning of the conversation.
- Use this report as the initial investigation context.
- Stay focused on the active incident.
- Do not mix information from different incidents.

AVAILABLE TOOLS:
1. get_robot_telemetry
2. get_incident_details
3. get_incident_telemetry
4. analyze_battery_discharge
5. analyze_wheel_slip
6. analyze_unexpected_stop

TOOL SELECTION:
- Use the report for questions already answered by the report.
- Use tools when additional measurements or evidence are needed.
- Prefer specialized analysis tools for known incident types.
- Use get_incident_telemetry for detailed sample-level analysis.
- Do not call unnecessary tools.
- Prefer numerical results returned by tools.

ENGINEERING RULES:
- Distinguish measurements from interpretations.
- Never invent telemetry, operating limits, or root causes.
- An incident classification does not confirm a root cause.
- If evidence is insufficient, explain what is missing.
- Battery changes are expressed in percentage points.
- Velocity and wheel speeds are measured in m/s.
- Motor current is measured in A.
- Do not assume timestamps are in UTC.

RESPONSE FORMAT:
- Answer the engineer's specific question.
- Keep answers concise, technical, and understandable.
- Explain supporting evidence when relevant.
- Clearly identify uncertain conclusions.
- Do not generate a complete incident report unless requested.
"""


# ============================================================
# 3. INITIALIZE CONVERSATION MEMORY
# ============================================================

# Store chat history while the application is running
chat_memory = InMemorySaver()


# ============================================================
# 4. CREATE THE CONVERSATIONAL AGENT
# ============================================================

# Agent 2 uses the same Gemini model and investigation tools,
# but returns conversational responses rather than a
# structured Pydantic IncidentReport.

chat_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=CHAT_SYSTEM_PROMPT,
    checkpointer=chat_memory
)


# ============================================================
# 5. CREATE INCIDENT CHAT CONTEXT
# ============================================================

def create_incident_chat_context(
    incident_report,
    incident_id
):
    """
    Prepare a new conversation using the verified report.

    Inputs:
        incident_report: Verified Pydantic IncidentReport
        incident_id: Selected incident ID

    Output:
        Dictionary containing the thread ID,
        incident ID, and initial report context.
    """

    # Convert the Pydantic report into a dictionary
    report_data = incident_report.model_dump(
        mode="json"
    )

    # Convert the dictionary into formatted JSON text
    report_json = json.dumps(
        report_data,
        indent=2
    )

    # Prepare the initial context for Gemini
    initial_context = f"""
ACTIVE INCIDENT ID:
{incident_id}

INITIAL INVESTIGATION REPORT:
{report_json}

The report has passed independent metadata and numerical
verification against the source data.

However, possible causes and technical interpretations
are not automatically confirmed.

Use this report to answer the engineer's questions.

Retrieve additional telemetry using the available tools
whenever necessary.
"""

    return {
        "thread_id": str(uuid4()),
        "incident_id": incident_id,
        "initial_context": initial_context
    }


# ============================================================
# 6. ASK FOLLOW-UP QUESTIONS
# ============================================================

def ask_incident_question(
    chat_context,
    question,
    first_message=False
):
    """
    Send an engineer's question to Agent 2.

    The first question includes the verified incident report.
    Later questions reuse the same conversation thread.

    Inputs:
        chat_context: Conversation context dictionary
        question: Engineer's question
        first_message: True for the first question

    Output:
        Gemini's conversational answer as text
    """

    # --------------------------------------------------------
    # 6.1 CONFIGURE CONVERSATION THREAD
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": chat_context["thread_id"]
        },
        "recursion_limit": 15
    }

    # --------------------------------------------------------
    # 6.2 PREPARE THE USER MESSAGE
    # --------------------------------------------------------

    if first_message:

        message_content = (
            chat_context["initial_context"]
            + "\n\nENGINEER QUESTION:\n"
            + question
        )

    else:

        message_content = question

    # --------------------------------------------------------
    # 6.3 CALL THE CONVERSATIONAL AGENT
    # --------------------------------------------------------

    response = chat_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        },
        config=config
    )

    # --------------------------------------------------------
    # 6.4 EXTRACT THE FINAL ANSWER
    # --------------------------------------------------------

    final_message = response["messages"][-1]

    content = final_message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

        return "\n".join(text_parts)

    return str(content)
