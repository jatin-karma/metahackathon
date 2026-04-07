# ===========================================================================
#  Dockerfile — Traffic Signal Control OpenEnv
# ===========================================================================
#  Build:  docker build -t traffic-env .
#  Run:    docker run -p 7860:7860 traffic-env
#  HF:     Deploy as Docker Space on Hugging Face Spaces (port 7860)
# ===========================================================================

FROM python:3.11-slim

# --- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ------------------------------------------------------
WORKDIR /app

# --- Install Python dependencies (cached layer) ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files -----------------------------------------------------
COPY env/       ./env/
COPY server/    ./server/
COPY openenv.yaml .
COPY inference.py .

# --- Expose HF Space port ---------------------------------------------------
EXPOSE 7860

# --- Environment defaults (override with -e flags or HF Spaces secrets) ----
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV TRAFFIC_ENV_URL="http://localhost:7860"

# --- Health check -----------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# --- Start FastAPI server ---------------------------------------------------
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--log-level", "info"]
