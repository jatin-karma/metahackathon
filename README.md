---
title: Traffic Signal Control OpenEnv
emoji: "🚦"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# Traffic Signal Control OpenEnv

FastAPI + Docker environment for adaptive traffic signal control at a 4-way intersection.

## What is here

- `server/app.py` exposes the OpenEnv API over HTTP.
- `env/traffic_env.py` contains the simulation.
- `env/models.py` defines the request/response models and tasks.
- `env/graders.py` computes the task score.
- `inference.py` is a sample LLM-driven agent for interacting with the environment.

## Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then check:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /grade`
- `POST /close`

## Deploy to Hugging Face Spaces

This repository is set up for a **Docker Space**.

1. Create a new Space on Hugging Face.
2. Choose **Docker** as the SDK.
3. Push this repository to the Space.
4. Make sure the app listens on port `7860`.
5. Add any required secrets in the Space settings.

Recommended secrets and environment variables:

- `API_BASE_URL` - defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME` - defaults to `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` - Hugging Face or API token
- `TRAFFIC_ENV_URL` - defaults to `http://localhost:7860`

## Validate submission

If you have Docker and `openenv` installed locally, run:

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space-name.hf.space
```

## Notes

- The Docker image starts the FastAPI app with `uvicorn` on port `7860`.
- The environment is deterministic for a fixed seed.
- Tasks available: `traffic_easy`, `traffic_medium`, `traffic_hard`.
