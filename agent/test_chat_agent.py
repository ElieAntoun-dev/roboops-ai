
# ============================================================
# 1. IMPORTS
# ============================================================

from uuid import uuid4

from agent.agent import agent
from agent.verify_report import verify_incident_report

from agent.chat_agent import (
    create_incident_chat_context,
    ask_incident_question
)


# ============================================================
# 2. GENERATE AN INCIDENT REPORT USING AGENT 1
# ============================================================

incident_id = 5

print(f"\nInvestigating Incident {incident_id}...")

# Create a new conversation for Agent 1
config = {
    "configurable": {
        "thread_id": str(uuid4())
    },
    "recursion_limit": 15
}

# Ask Agent 1 to investigate the incident
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Investigate battery discharge incident "
                    f"{incident_id}."
                )
            }
        ]
    },
    config=config
)

# Retrieve the structured Pydantic report
incident_report = response.get(
    "structured_response"
)

if incident_report is None:
    raise RuntimeError(
        "Agent 1 did not generate an incident report."
    )


# ============================================================
# 3. VERIFY THE INCIDENT REPORT
# ============================================================

print("\nVerifying the incident report...")

verified = verify_incident_report(
    incident_report
)

if not verified:
    raise RuntimeError(
        "The incident report failed verification."
    )

print("Report verification passed.")


# ============================================================
# 4. INITIALIZE AGENT 2 CONVERSATION
# ============================================================

chat_context = create_incident_chat_context(
    incident_report=incident_report,
    incident_id=incident_id
)

print("\nAgent 2 conversation initialized.")


# ============================================================
# 5. FIRST FOLLOW-UP QUESTION
# ============================================================

question_1 = (
    "Why was this incident classified as battery discharge?"
)

print("\n========== QUESTION 1 ==========")
print(question_1)

answer_1 = ask_incident_question(
    chat_context=chat_context,
    question=question_1,
    first_message=True
)

print("\nAGENT 2 ANSWER:")
print(answer_1)


# ============================================================
# 6. SECOND FOLLOW-UP QUESTION
# ============================================================

question_2 = (
    "Could motor current explain this behavior? "
    "Use the available telemetry evidence."
)

print("\n========== QUESTION 2 ==========")
print(question_2)

answer_2 = ask_incident_question(
    chat_context=chat_context,
    question=question_2,
    first_message=False
)

print("\nAGENT 2 ANSWER:")
print(answer_2)


# ============================================================
# 7. TEST CONVERSATION MEMORY
# ============================================================

question_3 = (
    "Which incident and robot are we currently discussing?"
)

print("\n========== QUESTION 3 ==========")
print(question_3)

answer_3 = ask_incident_question(
    chat_context=chat_context,
    question=question_3,
    first_message=False
)

print("\nAGENT 2 ANSWER:")
print(answer_3)

print("\n========== TEST COMPLETED ==========")
