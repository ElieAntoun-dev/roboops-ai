
# ============================================================
# 1. IMPORTS AND PROJECT CONFIGURATION
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from uuid import uuid4
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Agent 1: Structured incident investigation
from agent.agent import agent

# Independent report verification
from agent.verify_report import verify_incident_report

# Agent 2: Conversational incident investigation
from agent.chat_agent import (
    create_incident_chat_context,
    ask_incident_question
)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoboOps AI",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 3. LOAD FLEET DATA
# ============================================================

TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "robot_telemetry_test.csv"
)

INCIDENTS_PATH = (
    PROJECT_ROOT / "data" / "classified_incidents_test_filtered.csv"
)


@st.cache_data
def load_data():

    telemetry = pd.read_csv(TELEMETRY_PATH)
    incidents = pd.read_csv(INCIDENTS_PATH)

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"]
    )

    incidents["start_time"] = pd.to_datetime(
        incidents["start_time"]
    )

    incidents["end_time"] = pd.to_datetime(
        incidents["end_time"]
    )

    return telemetry, incidents


telemetry_df, incidents_df = load_data()


# ============================================================
# 4. INITIALIZE SESSION STATE
# ============================================================

# Store the verified investigation report
if "investigation_report" not in st.session_state:
    st.session_state.investigation_report = None

# Store the incident ID associated with the report
if "investigation_incident_id" not in st.session_state:
    st.session_state.investigation_incident_id = None

# Store the conversational agent context
if "chat_context" not in st.session_state:
    st.session_state.chat_context = None

# Store the chat messages displayed in Streamlit
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Track whether the initial report context has been sent
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False


# ============================================================
# 5. DISPLAY STRUCTURED INCIDENT REPORT
# ============================================================

def display_incident_report(report):

    report_data = report.model_dump(mode="json")

    st.subheader("Incident Investigation Report")

    # --------------------------------------------------------
    # 5.1 INCIDENT INFORMATION
    # --------------------------------------------------------

    st.markdown("### 1. Incident Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Incident ID",
            report_data["incident_id"]
        )

    with col2:
        st.metric(
            "Robot ID",
            report_data["robot_id"]
        )

    with col3:
        st.metric(
            "Incident Type",
            report_data["incident_type"].replace("_", " ").title()
        )

    st.write("**Start Time:**", report_data["start_time"])
    st.write("**End Time:**", report_data["end_time"])

    st.divider()

    # --------------------------------------------------------
    # 5.2 INCIDENT SUMMARY
    # --------------------------------------------------------

    st.markdown("### 2. Incident Summary")

    st.write(report_data["summary"])

    st.divider()

    # --------------------------------------------------------
    # 5.3 OBSERVATIONS
    # --------------------------------------------------------

    st.markdown("### 3. Observations")

    for observation in report_data["observations"]:
        st.markdown(f"- {observation}")

    st.divider()

    # --------------------------------------------------------
    # 5.4 MEASUREMENTS
    # --------------------------------------------------------

    st.markdown("### 4. Measurements")

    measurements = report_data["measurements"]["values"]

    measurement_rows = []

    for key, value in measurements.items():

        measurement_rows.append({
            "Measurement": key.replace("_", " ").title(),
            "Value": value
        })

    st.dataframe(
        pd.DataFrame(
            measurement_rows,
            columns=["Measurement", "Value"]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Numerical measurements are independently checked "
        "against the incident source data."
    )

    st.divider()

    # --------------------------------------------------------
    # 5.5 TECHNICAL ASSESSMENT
    # --------------------------------------------------------

    st.markdown("### 5. Technical Assessment")

    st.write(report_data["technical_assessment"])

    st.divider()

    # --------------------------------------------------------
    # 5.6 POSSIBLE CAUSES
    # --------------------------------------------------------

    st.markdown("### 6. Possible Causes")

    for cause in report_data["possible_causes"]:
        st.markdown(f"- {cause}")

    st.divider()

    # --------------------------------------------------------
    # 5.7 ROOT CAUSE
    # --------------------------------------------------------

    st.markdown("### 7. Root Cause")

    if report_data["root_cause_confirmed"]:
        st.success("Root cause confirmed.")
    else:
        st.warning("Root cause not confirmed.")

    if report_data.get("root_cause"):
        st.write(report_data["root_cause"])

    st.divider()

    # --------------------------------------------------------
    # 5.8 RECOMMENDED ACTIONS
    # --------------------------------------------------------

    st.markdown("### 8. Recommended Actions")

    for action in report_data["recommended_actions"]:
        st.markdown(f"- {action}")

    st.divider()

    # --------------------------------------------------------
    # 5.9 INVESTIGATION STATUS
    # --------------------------------------------------------

    st.markdown("### 9. Investigation Status")

    st.write(report_data["investigation_status"])

    st.divider()

    # --------------------------------------------------------
    # 5.10 RAW JSON
    # --------------------------------------------------------

    with st.expander("View Raw JSON Report"):
        st.json(report_data)


# ============================================================
# 6. CREATE INTERACTIVE TELEMETRY CHART
# ============================================================

def create_telemetry_chart(
    telemetry,
    columns,
    title,
    y_axis_label,
    incident_start,
    incident_end,
    display_start,
    display_end
):

    fig = go.Figure()

    # Add telemetry signals
    for column in columns:

        fig.add_trace(
            go.Scatter(
                x=telemetry["timestamp"],
                y=telemetry[column],
                mode="lines+markers",
                name=column.replace("_", " ").title(),
                line=dict(width=2),
                marker=dict(size=4)
            )
        )

    # Highlight the incident interval
    fig.add_vrect(
        x0=incident_start,
        x1=incident_end,
        fillcolor="red",
        opacity=0.15,
        layer="below",
        line_width=0,
        annotation_text="Incident",
        annotation_position="top left"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_axis_label,
        hovermode="x unified",
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        range=[display_start, display_end]
    )

    return fig


# ============================================================
# 7. DISPLAY INCIDENT TELEMETRY
# ============================================================

def display_incident_telemetry(incident, telemetry):

    st.subheader("Incident Telemetry Analysis")

    st.write(
        "Analyze the robot's behavior before, during, "
        "and after the detected incident."
    )

    robot_id = incident["robot_id"]

    incident_start = pd.to_datetime(
        incident["start_time"]
    )

    incident_end = pd.to_datetime(
        incident["end_time"]
    )

    # --------------------------------------------------------
    # 7.1 SELECT TIME WINDOW
    # --------------------------------------------------------

    st.markdown("### Telemetry Time Window")

    context_seconds = st.slider(
        "Seconds before and after the incident",
        min_value=0,
        max_value=60,
        value=10,
        step=5,
        key=f"telemetry_window_{incident['incident_id']}"
    )

    display_start = (
        incident_start
        - pd.Timedelta(seconds=context_seconds)
    )

    display_end = (
        incident_end
        + pd.Timedelta(seconds=context_seconds)
    )

    # --------------------------------------------------------
    # 7.2 FILTER TELEMETRY
    # --------------------------------------------------------

    incident_telemetry = telemetry[
        (telemetry["robot_id"] == robot_id)
        & (telemetry["timestamp"] >= display_start)
        & (telemetry["timestamp"] <= display_end)
    ].copy()

    incident_telemetry = incident_telemetry.sort_values(
        "timestamp"
    )

    if incident_telemetry.empty:

        st.warning(
            "No telemetry records were found "
            "for the selected time window."
        )

        return

    # --------------------------------------------------------
    # 7.3 DISPLAY TELEMETRY INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Robot ID", robot_id)

    with col2:
        st.metric(
            "Telemetry Records",
            len(incident_telemetry)
        )

    with col3:
        st.metric(
            "Context Window",
            f"±{context_seconds} s"
        )

    st.write("**Incident Start:**", incident_start)
    st.write("**Incident End:**", incident_end)

    st.caption(
        "The shaded red region indicates the detected "
        "incident interval."
    )

    # --------------------------------------------------------
    # 7.4 VELOCITY GRAPH
    # --------------------------------------------------------

    st.markdown("### 1. Robot Velocity")

    velocity_chart = create_telemetry_chart(
        telemetry=incident_telemetry,
        columns=["velocity"],
        title="Robot Velocity Over Time",
        y_axis_label="Velocity (m/s)",
        incident_start=incident_start,
        incident_end=incident_end,
        display_start=display_start,
        display_end=display_end
    )

    st.plotly_chart(
        velocity_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 7.5 WHEEL SPEED GRAPH
    # --------------------------------------------------------

    st.markdown("### 2. Wheel Speeds")

    wheel_chart = create_telemetry_chart(
        telemetry=incident_telemetry,
        columns=[
            "left_wheel_speed",
            "right_wheel_speed"
        ],
        title="Left and Right Wheel Speeds",
        y_axis_label="Wheel Speed (m/s)",
        incident_start=incident_start,
        incident_end=incident_end,
        display_start=display_start,
        display_end=display_end
    )

    st.plotly_chart(
        wheel_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 7.6 MOTOR CURRENT GRAPH
    # --------------------------------------------------------

    st.markdown("### 3. Motor Current")

    current_chart = create_telemetry_chart(
        telemetry=incident_telemetry,
        columns=["motor_current"],
        title="Motor Current Over Time",
        y_axis_label="Motor Current (A)",
        incident_start=incident_start,
        incident_end=incident_end,
        display_start=display_start,
        display_end=display_end
    )

    st.plotly_chart(
        current_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 7.7 BATTERY LEVEL GRAPH
    # --------------------------------------------------------

    st.markdown("### 4. Battery Level")

    battery_chart = create_telemetry_chart(
        telemetry=incident_telemetry,
        columns=["battery_level"],
        title="Battery Level Over Time",
        y_axis_label="Battery Level (%)",
        incident_start=incident_start,
        incident_end=incident_end,
        display_start=display_start,
        display_end=display_end
    )

    st.plotly_chart(
        battery_chart,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 7.8 RAW TELEMETRY DATA
    # --------------------------------------------------------

    with st.expander("View Incident Telemetry Data"):

        st.dataframe(
            incident_telemetry,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 8. DISPLAY FOLLOW-UP INVESTIGATION CHATBOT
# ============================================================

def display_investigation_chatbot():
    """
    Display the Agent 2 chatbot.

    The chatbot receives the verified incident report
    and maintains conversation history using session state.
    """

    st.subheader("RoboOps AI Investigation Assistant")

    st.write(
        "Ask follow-up questions about the incident, "
        "investigate possible causes, or request "
        "additional telemetry analysis."
    )

    st.caption(
        "The assistant uses the verified incident report "
        "and the existing investigation tools."
    )

    # --------------------------------------------------------
    # 8.1 DISPLAY PREVIOUS CHAT MESSAGES
    # --------------------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------------
    # 8.2 RECEIVE USER QUESTION
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about this incident..."
    )

    if question:

        # Display and store the user's question
        st.session_state.chat_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # ----------------------------------------------------
        # 8.3 CALL AGENT 2
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Analyzing incident..."):

                try:

                    # Include the report only in the first question
                    first_message = (
                        not st.session_state.chat_initialized
                    )

                    answer = ask_incident_question(
                        chat_context=st.session_state.chat_context,
                        question=question,
                        first_message=first_message
                    )

                    # Mark the conversation as initialized
                    # only after the agent responds successfully
                    st.session_state.chat_initialized = True

                    st.markdown(answer)

                    # Store the assistant response
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    st.error(
                        f"Chatbot failed: {error}"
                    )


# ============================================================
# 9. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("🤖 RoboOps AI")

    st.caption("AMR Fleet Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Fleet Overview",
            "Incident Explorer",
            "AI Investigation"
        ]
    )

    st.divider()

    st.caption("RoboOps AI | Version 1.0")


# ============================================================
# 10. FLEET OVERVIEW PAGE
# ============================================================

if page == "Fleet Overview":

    st.title("Fleet Overview")

    st.write(
        "Overview of AMR fleet performance and detected incidents."
    )

    st.divider()

    total_robots = telemetry_df["robot_id"].nunique()
    total_records = len(telemetry_df)
    total_incidents = len(incidents_df)

    incident_types = (
        incidents_df["predicted_incident_type"].nunique()
    )

    # --------------------------------------------------------
    # 10.1 DISPLAY FLEET METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Robots Monitored", total_robots)

    with col2:
        st.metric(
            "Telemetry Records",
            f"{total_records:,}"
        )

    with col3:
        st.metric(
            "Detected Incidents",
            total_incidents
        )

    with col4:
        st.metric(
            "Incident Types",
            incident_types
        )

    # --------------------------------------------------------
    # 10.2 INCIDENT DISTRIBUTION
    # --------------------------------------------------------

    st.divider()

    st.subheader("Incident Distribution")

    incident_counts = (
        incidents_df["predicted_incident_type"]
        .value_counts()
        .rename_axis("Incident Type")
        .reset_index(name="Number of Incidents")
    )

    st.bar_chart(
        incident_counts,
        x="Incident Type",
        y="Number of Incidents"
    )

    # --------------------------------------------------------
    # 10.3 INCIDENTS PER ROBOT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Incidents per Robot")

    robot_incident_counts = (
        incidents_df["robot_id"]
        .value_counts()
        .rename_axis("Robot ID")
        .reset_index(name="Number of Incidents")
    )

    st.bar_chart(
        robot_incident_counts,
        x="Robot ID",
        y="Number of Incidents"
    )


# ============================================================
# 11. INCIDENT EXPLORER PAGE
# ============================================================

elif page == "Incident Explorer":

    st.title("Incident Explorer")

    st.write(
        "Browse and analyze incidents detected "
        "by the anomaly detection pipeline."
    )

    st.divider()

    st.metric(
        "Total Detected Incidents",
        len(incidents_df)
    )

    # --------------------------------------------------------
    # 11.1 INCIDENT FILTERS
    # --------------------------------------------------------

    st.subheader("Filter Incidents")

    col1, col2 = st.columns(2)

    with col1:

        selected_robot = st.selectbox(
            "Filter by Robot",
            ["All Robots"] + sorted(
                incidents_df["robot_id"].unique().tolist()
            )
        )

    with col2:

        selected_type = st.selectbox(
            "Filter by Incident Type",
            ["All Types"] + sorted(
                incidents_df["predicted_incident_type"]
                .unique()
                .tolist()
            )
        )

    # --------------------------------------------------------
    # 11.2 APPLY FILTERS
    # --------------------------------------------------------

    filtered_incidents = incidents_df.copy()

    if selected_robot != "All Robots":

        filtered_incidents = filtered_incidents[
            filtered_incidents["robot_id"] == selected_robot
        ]

    if selected_type != "All Types":

        filtered_incidents = filtered_incidents[
            filtered_incidents["predicted_incident_type"]
            == selected_type
        ]

    # --------------------------------------------------------
    # 11.3 DISPLAY INCIDENT TABLE
    # --------------------------------------------------------

    st.subheader("Detected Incidents")

    display_columns = [
        "incident_id",
        "robot_id",
        "predicted_incident_type",
        "start_time",
        "end_time",
        "duration_seconds",
        "anomaly_count"
    ]

    incident_table = filtered_incidents[
        display_columns
    ].copy()

    incident_table = incident_table.rename(
        columns={
            "incident_id": "Incident ID",
            "robot_id": "Robot ID",
            "predicted_incident_type": "Incident Type",
            "start_time": "Start Time",
            "end_time": "End Time",
            "duration_seconds": "Duration (s)",
            "anomaly_count": "Anomaly Count"
        }
    )

    st.caption(
        f"Showing {len(filtered_incidents)} of "
        f"{len(incidents_df)} incidents"
    )

    st.dataframe(
        incident_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 11.4 DISPLAY SELECTED INCIDENT DETAILS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Incident Details")

    if filtered_incidents.empty:

        st.info(
            "No incidents match the selected filters."
        )

    else:

        incident_ids = sorted(
            filtered_incidents["incident_id"].tolist()
        )

        selected_incident_id = st.selectbox(
            "Select an Incident ID",
            incident_ids
        )

        selected_incident = filtered_incidents[
            filtered_incidents["incident_id"]
            == selected_incident_id
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Incident ID",
                int(selected_incident["incident_id"])
            )

        with col2:
            st.metric(
                "Robot ID",
                selected_incident["robot_id"]
            )

        with col3:
            st.metric(
                "Incident Type",
                selected_incident["predicted_incident_type"]
            )

        st.write(
            "**Start Time:**",
            selected_incident["start_time"]
        )

        st.write(
            "**End Time:**",
            selected_incident["end_time"]
        )

        st.write(
            "**Duration (s):**",
            selected_incident["duration_seconds"]
        )

        st.write(
            "**Anomaly Count:**",
            selected_incident["anomaly_count"]
        )


# ============================================================
# 12. AI INVESTIGATION PAGE
# ============================================================

elif page == "AI Investigation":

    st.title("AI Incident Investigation")

    st.write(
        "Select an incident and investigate it "
        "using the Gemini AI agent."
    )

    st.divider()

    # --------------------------------------------------------
    # 12.1 SELECT INCIDENT
    # --------------------------------------------------------

    incident_ids = sorted(
        incidents_df["incident_id"].tolist()
    )

    selected_incident_id = st.selectbox(
        "Select an Incident ID",
        incident_ids
    )

    selected_incident = incidents_df[
        incidents_df["incident_id"]
        == selected_incident_id
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Robot ID",
            selected_incident["robot_id"]
        )

    with col2:
        st.metric(
            "Incident Type",
            selected_incident["predicted_incident_type"]
        )

    # --------------------------------------------------------
    # 12.2 RUN AGENT 1 INVESTIGATION
    # --------------------------------------------------------

    if st.button(
        "🔍 Investigate Incident",
        type="primary"
    ):

        # Clear previous investigation and conversation
        st.session_state.investigation_report = None
        st.session_state.investigation_incident_id = None

        st.session_state.chat_context = None
        st.session_state.chat_messages = []
        st.session_state.chat_initialized = False

        with st.spinner(
            f"Gemini is investigating Incident "
            f"{selected_incident_id}..."
        ):

            try:

                config = {
                    "configurable": {
                        "thread_id": str(uuid4())
                    },
                    "recursion_limit": 15
                }

                question = (
                    f"Investigate incident {selected_incident_id}, "
                    f"classified as "
                    f"{selected_incident['predicted_incident_type']}."
                )

                # Execute Agent 1
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

                structured_report = response.get(
                    "structured_response"
                )

                # ------------------------------------------------
                # 12.3 VERIFY THE REPORT
                # ------------------------------------------------

                if structured_report is None:

                    st.error(
                        "Gemini did not return a structured "
                        "incident report."
                    )

                else:

                    verified = verify_incident_report(
                        structured_report
                    )

                    if verified:

                        # Save the verified report
                        st.session_state.investigation_report = (
                            structured_report
                        )

                        st.session_state.investigation_incident_id = (
                            selected_incident_id
                        )

                        # ----------------------------------------
                        # 12.4 INITIALIZE AGENT 2 CONTEXT
                        # ----------------------------------------

                        st.session_state.chat_context = (
                            create_incident_chat_context(
                                incident_report=structured_report,
                                incident_id=selected_incident_id
                            )
                        )

                        st.session_state.chat_messages = []
                        st.session_state.chat_initialized = False

                    else:

                        st.error(
                            "The incident report did not "
                            "pass verification."
                        )

            except Exception as error:

                st.error(
                    f"Investigation failed: {error}"
                )

    # ========================================================
    # 12.5 DISPLAY SAVED INVESTIGATION RESULTS
    # ========================================================

    if (
        st.session_state.investigation_report is not None
        and st.session_state.investigation_incident_id
        == selected_incident_id
    ):

        st.divider()

        st.success(
            "Investigation completed. "
            "Source-data verification passed."
        )

        # Display the structured engineering report
        display_incident_report(
            st.session_state.investigation_report
        )

        st.divider()

        # Display interactive telemetry graphs
        display_incident_telemetry(
            selected_incident,
            telemetry_df
        )

        st.divider()

        # Display the conversational investigation assistant
        if st.session_state.chat_context is not None:

            display_investigation_chatbot()
