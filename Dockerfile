# Stage 1: Base image with Java (PySpark) and Python
FROM eclipse-temurin:11-jdk-jammy as base

# Core configurations
ENV PYTHON_VERSION=3.9 \
    AIRFLOW_VERSION=2.8.1 \
    AIRFLOW_HOME=/opt/airflow \
    SPARK_HOME=/opt/spark \
    PATH="/opt/spark/bin:/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    libarrow-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependency installation
FROM base as builder

WORKDIR /install

# Airflow 2.8.1 constraint file
ARG AIRFLOW_CONSTRAINTS="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

COPY requirements.txt .

# Install with version pinning
RUN pip install --user \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "${AIRFLOW_CONSTRAINTS}" && \
    pip install --user -r requirements.txt

# Stage 3: Final image
FROM base

# Copy installed Python packages
COPY --from=builder /root/.local /root/.local

# Install Spark (PySpark 3.5.0 compatible)
RUN curl -s https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz | tar xz -C /opt && \
    ln -s /opt/spark-3.5.0-bin-hadoop3 /opt/spark

WORKDIR /app
COPY . .

# Initialize Airflow DB if missing
RUN if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then \
    airflow db init; \
    airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com; \
    fi

CMD ["airflow", "standalone"]