# 🤖 RoboOps AI
### Agentic AI Platform for Autonomous Mobile Robot Fleet Monitoring and Incident Investigation

RoboOps AI is an end-to-end intelligent monitoring and investigation platform for Autonomous Mobile Robots (AMRs). It combines machine learning-based anomaly detection, incident classification, AI-powered investigation agents, and interactive telemetry visualization to help engineers analyze abnormal robot behavior.

The platform uses **Google Gemini, LangChain, and LangGraph** to investigate incidents through specialized Python tools, generate structured engineering reports, verify numerical findings against source telemetry, and support conversational follow-up investigations.

The application is built with **Python and Streamlit**, containerized using **Docker**, and supports exporting investigation reports as PDF documents.

---

## 1. Project Overview

Autonomous Mobile Robots generate continuous operational telemetry, including velocity, wheel speeds, motor current, battery level, and operating status.

Identifying abnormal behavior and investigating potential failures can require engineers to manually inspect large telemetry datasets, compare measurements, and determine possible causes.

**RoboOps AI aims to simplify this process through an automated investigation workflow.**

The platform provides:

- **Robot Fleet Monitoring:** Overview of monitored robots and detected incidents.
- **Anomaly Detection:** Machine learning-based identification of abnormal telemetry behavior.
- **Incident Classification:** Categorization and filtering of detected anomalies into incidents.
- **AI Incident Investigation:** Automated engineering analysis using an LLM agent and specialized Python tools.
- **Independent Report Verification:** Validation of report metadata and numerical measurements against source data.
- **Conversational AI Assistant:** Follow-up investigation through a second AI agent with conversation memory.
- **Telemetry Visualization:** Interactive graphs showing robot behavior before, during, and after incidents.
- **PDF Report Export:** Downloadable engineering investigation reports.
- **Docker Support:** Containerized execution for reproducible deployment.

## 2. System Architecture

RoboOps AI combines a machine learning pipeline with a two-agent investigation system.

```text
                 AMR Telemetry Data
                         |
                         v
               Data Processing Pipeline
                         |
                         v
                Anomaly Detection
                         |
                         v
              Incident Classification
                         |
                         v
                 Incident Database
                         |
                         v
                Streamlit Interface
                         |
                         v
               Engineer Selects Incident
                         |
                         v
              AGENT 1: AI Investigation
                  Google Gemini LLM
                         |
                         v
               Specialized Python Tools
                  |      |      |
                  v      v      v
              Telemetry Analysis Tools
                         |
                         v
             Structured Incident Report
                         |
                         v
             Independent Report Verification
                         |
                         v
                  Verified Report
                    /         \
                   v           v
              PDF Export     AGENT 2
                              |
                              v
                       Conversational
                      Investigation AI
                              |
                              v
                      Follow-up Analysis
```

### Agentic AI Architecture

The investigation system consists of two agents with different responsibilities.

**Agent 1 — Incident Investigation Agent**

Responsible for investigating a selected incident, retrieving relevant telemetry, executing specialized analysis tools, and generating a structured engineering report.

**Agent 2 — Conversational Investigation Agent**

Responsible for answering follow-up questions using the verified investigation report, conversation history, and additional telemetry analysis tools when required.

Both agents use Google Gemini and share the available investigation tools.

## 3. Machine Learning and Anomaly Detection

The machine learning pipeline processes robot telemetry to identify abnormal operating behavior.

### Telemetry Features

The dataset includes operational signals such as:

| Feature | Description |
|---|---|
| Robot ID | Identifier of the monitored AMR |
| Timestamp | Telemetry recording time |
| Velocity | Robot velocity (m/s) |
| Left Wheel Speed | Left wheel velocity (m/s) |
| Right Wheel Speed | Right wheel velocity (m/s) |
| Motor Current | Motor current (A) |
| Battery Level | Battery state of charge (%) |
| Robot Status | Recorded operating state |

### Detection Pipeline

The anomaly detection workflow consists of:

1. Loading and preprocessing robot telemetry.
2. Applying machine learning-based anomaly detection.
3. Identifying abnormal operating samples.
4. Grouping and classifying detected incidents.
5. Filtering incidents to reduce false positives.
6. Preparing the resulting incidents for AI investigation.

The machine learning implementation is organized within the `ml/`, `simulation/`, `incidents/`, and `validation/` directories.

## 4. Agent 1 — Automated Incident Investigation

Agent 1 is implemented using **LangChain, LangGraph, and Google Gemini**.

The engineer selects an incident through the Streamlit interface, and the agent receives an investigation request containing the incident ID and classification.

The agent can retrieve relevant information through Python tools and generate a structured engineering report.

### Available Investigation Tools

| Tool | Purpose |
|---|---|
| `get_robot_telemetry` | Retrieve telemetry for a selected robot |
| `get_incident_details` | Retrieve incident metadata |
| `get_incident_telemetry` | Retrieve telemetry associated with an incident |
| `analyze_battery_discharge` | Analyze battery discharge behavior |
| `analyze_wheel_slip` | Analyze possible wheel slip |
| `analyze_unexpected_stop` | Analyze unexpected robot stops |

### Structured Investigation Report

Agent 1 generates a report containing nine sections:

1. Incident Information
2. Incident Summary
3. Observations
4. Measurements
5. Technical Assessment
6. Possible Causes
7. Root Cause
8. Recommended Actions
9. Investigation Status

The report is validated against a Pydantic schema to maintain a consistent output structure.

The agent is instructed to distinguish recorded measurements from technical interpretations and avoid presenting unconfirmed root causes as established facts.

## 5. Independent Report Verification

To improve reliability, RoboOps AI includes a verification layer implemented in `agent/verify_report.py`.

The verification process checks report metadata and numerical measurements against the original incident data.

```text
Agent 1
   |
   v
Structured Incident Report
   |
   v
Independent Verification
   |
   +---- Verification Failed
   |           |
   |           v
   |     Report Rejected
   |
   +---- Verification Passed
               |
               v
        Report Accepted
               |
               v
       Streamlit Display
               |
               v
         PDF / Agent 2
```

Only reports that pass verification are accepted for display and further investigation.

**Important:** Numerical verification does not automatically establish that a proposed root cause is correct. Possible causes remain technical interpretations unless supported by sufficient evidence.

## 6. Agent 2 — Conversational Investigation Assistant

Agent 2 provides an interactive engineering assistant for investigating an incident after Agent 1 has generated a verified report.

The agent is implemented using LangChain and LangGraph, with Google Gemini as the language model.

### Key Capabilities

- Receives the verified report as initial investigation context.
- Answers technical questions about the active incident.
- Maintains conversation history using a LangGraph checkpointer.
- Reuses the investigation tools when additional telemetry is required.
- Supports detailed analysis of specific measurements and operating conditions.
- Distinguishes confirmed findings from possible explanations.
- Remains focused on the active incident.

To investigate another incident, the engineer must select it through the Streamlit interface and generate a new report.

### Example Questions

```text
What was the robot velocity during the incident?

Did the motor current change during the unexpected stop?

What evidence supports the possible causes?

Can you retrieve more detailed telemetry?

What additional measurements are needed to confirm
the root cause?
```

The assistant uses the report when sufficient information is already available and can request additional analysis through Python tools.

## 7. Streamlit Application

The user interface is implemented in `ui/interface.py`.

It contains three main pages.

### Fleet Overview

Displays general fleet information, including:

- Number of monitored robots
- Total telemetry records
- Number of detected incidents
- Number of incident types
- Incident distribution
- Incidents per robot

### Incident Explorer

Allows engineers to:

- Browse detected incidents.
- Filter incidents by robot.
- Filter incidents by incident type.
- Review incident timestamps, duration, and anomaly counts.

### AI Investigation

Provides the complete AI investigation workflow:

1. Select an incident.
2. Execute Agent 1.
3. Verify the generated report.
4. Display the structured investigation report.
5. Download the report as a PDF.
6. Explore interactive telemetry graphs.
7. Ask follow-up questions through Agent 2.

### Interactive Telemetry Visualization

The application uses Plotly to visualize:

- Robot velocity
- Left and right wheel speeds
- Motor current
- Battery level

Engineers can adjust the time window before and after an incident to inspect the robot's behavior.

The incident interval is highlighted within the graphs.

## 8. PDF Report Export

RoboOps AI uses **ReportLab** to generate downloadable PDF investigation reports.

The PDF contains the verified engineering report, including incident information, observations, numerical measurements, technical assessment, possible causes, and recommended actions.

PDF generation is performed in memory, and engineers can download the document directly through the Streamlit interface.

Example filename:

```text
RoboOps_Incident_7_Report.pdf
```

## 9. Project Structure

```text
roboops-ai/
│
├── agent/
│   ├── agent.py
│   ├── chat_agent.py
│   ├── tools.py
│   └── verify_report.py
│
├── evaluation/
│
├── incidents/
│
├── ml/
│
├── simulation/
│
├── ui/
│   └── interface.py
│
├── validation/
│
├── data/
│   ├── robot_telemetry_test.csv
│   └── classified_incidents_test_filtered.csv
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

**Note:** The `data/` directory contains the input CSV files required by the Streamlit application. These files must be available when running the project.

## 10. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, SciPy |
| Large Language Model | Google Gemini |
| Agent Framework | LangChain, LangGraph |
| Structured Output | Pydantic |
| User Interface | Streamlit |
| Visualization | Plotly |
| PDF Generation | ReportLab |
| Containerization | Docker |
| Version Control | Git, GitHub |

## 11. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Google Gemini API key
- Docker (optional)

### Clone the Repository

```bash
git clone https://github.com/ElieAntoun-dev/roboops-ai.git
cd roboops-ai
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

The `.env` file must not be committed to GitHub.

### Run the Streamlit Application

```bash
streamlit run ui/interface.py
```

Open the application:

```text
http://localhost:8501
```

## 12. Running with Docker

RoboOps AI can also be executed in a Docker container.

### Build the Docker Image

From the project root:

```bash
docker build -t roboops-ai .
```

### Run the Container

```bash
docker run --rm -p 8501:8501 --env-file .env roboops-ai
```

The application will be accessible at:

```text
http://localhost:8501
```

The `--env-file` option provides the Gemini API key at runtime without embedding it in the Docker image.

### Stop the Container

Press `Ctrl + C` in the terminal running Docker.

## 13. Current Scope and Limitations

RoboOps AI is an engineering prototype intended to demonstrate the integration of machine learning, agentic AI, and telemetry-based incident investigation.

Current limitations include:

- The application operates on prepared telemetry and incident datasets rather than a live robot fleet.
- AI-generated technical explanations require engineering judgment.
- Verification checks report data consistency but does not guarantee causal correctness.
- Agent conversation memory is maintained in memory and is not designed for long-term persistence.
- The current PDF export contains the structured report without the interactive telemetry charts.
- Additional testing and security controls would be required before production deployment.

## 14. Future Improvements

Potential extensions include:

- Real-time telemetry ingestion from AMRs.
- Integration with robot fleet management systems.
- Persistent incident storage and investigation history.
- Automated notifications for critical incidents.
- Additional specialized diagnostic tools.
- Expanded agent evaluation and automated testing.
- Production deployment with authentication and monitoring.

## 15. Project Objective

RoboOps AI demonstrates how traditional machine learning and modern agentic AI can be combined to support engineering workflows.

Instead of relying exclusively on an LLM to interpret operational data, the platform integrates deterministic Python analysis tools, structured reporting, independent numerical verification, and conversational investigation.

The objective is to develop an AI-assisted engineering system that makes robot incident analysis more accessible, structured, and evidence-driven.

---

**Developed by Elie Antoun**

Mechatronics Engineer | Robotics | Python | Machine Learning | Agentic AI

[GitHub Profile](https://github.com/ElieAntoun-dev) | [LinkedIn](https://www.linkedin.com/in/elie-antoun01/)