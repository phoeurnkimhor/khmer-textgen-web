FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y gcc curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies stage ---
FROM base AS deps
WORKDIR /app

# Configure pip for better reliability
RUN pip config set global.timeout 1000 && \
    pip config set global.retries 3 && \
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"

COPY requirements.txt .

# Install torch first (largest package) with CPU-only version for smaller size
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# --- Production stage ---
FROM base AS runner
WORKDIR /app

# Copy installed packages
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Non-root user
RUN addgroup --system --gid 1001 appuser && adduser --system --uid 1001 --gid 1001 appuser

# Copy backend source code
COPY . /app

# Change ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8800
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800", "--reload"]
