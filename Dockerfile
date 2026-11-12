
# Use Python 3.11 as the base environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Streamlit runs on port 8501
EXPOSE 8501

# Start the RoboOps AI application
CMD ["streamlit", "run", "ui/interface.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
