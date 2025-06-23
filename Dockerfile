FROM python:3.13-slim-bookworm

WORKDIR /app

# Install system libraries
COPY requirements.txt /app/
RUN apt-get update \
 && apt-get install -y \
      libxml2-dev \
      libxslt-dev \
      libffi-dev \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY echopulse.py /app
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Launch the Streamlit app
CMD ["streamlit", "run", "echopulse.py", "--server.port", "8501", "--server.address", "0.0.0.0"]