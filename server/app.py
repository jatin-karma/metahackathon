"""
FastAPI server — Traffic Signal Control OpenEnv
===============================================

Exposes the OpenEnv standard REST API:

  POST /reset         → StepResult   (initialise / restart episode)
  POST /step          → StepResult   (advance one step)
  GET  /state         → TrafficState (full internal snapshot)
  POST /close         → {"status": "closed"}
  GET  /health        → {"status": "ok", "task": ..., "done": ...}

The server maintains a single global environment instance (suitable for
one-session HF Spaces deployments). Concurrent multi-session use is not
required by the OpenEnv spec.

Running locally
---------------
  uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

Docker / HF Space
-----------------
  CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import sys
import os

# Make the project root importable when running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.traffic_env import TrafficSignalEnv
from env.models import TrafficAction, StepResult, TrafficState, VALID_TASKS
from env.graders import grade

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Traffic Signal Control — OpenEnv",
    description=(
        "Adaptive traffic signal control at a 4-way intersection. "
        "An AI agent controls NS/EW green phases to minimise wait time "
        "and maximise throughput."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------

_env = TrafficSignalEnv()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task: Optional[str] = "traffic_easy"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: str  # "switch_signal" | "keep_signal"


class GradeResponse(BaseModel):
    task: str
    score: float
    stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe — always returns 200 OK."""
    state = _env.state()
    return {
        "status": "ok",
        "task": state.task,
        "done": state.done,
        "time_elapsed": state.observation.time_elapsed,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    """Base route for Space landing checks."""
    return {
        "status": "ok",
        "message": "Traffic Signal Control OpenEnv is running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/grade", "/close"],
    }


@app.post("/reset", response_model=StepResult)
def reset(req: Optional[ResetRequest] = None) -> StepResult:
    """
    Reset the environment for a new episode.

    Body (optional JSON):
      { "task": "traffic_easy" | "traffic_medium" | "traffic_hard",
        "seed": <int> }

    Returns the initial StepResult (reward=0, done=false).
    """
    task = (req.task if req else None) or "traffic_easy"
    seed = (req.seed if req else None) or 42

    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task}'. Valid tasks: {VALID_TASKS}",
        )

    result = _env.reset(task=task, seed=seed)
    return result


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """
    Advance the environment by one step.

    Body:
      { "action": "switch_signal" | "keep_signal" }

    Returns (observation, reward, done, info).
    """
    if req.action not in ("switch_signal", "keep_signal"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{req.action}'. "
                   f"Must be 'switch_signal' or 'keep_signal'.",
        )

    state = _env.state()
    if state.done:
        raise HTTPException(
            status_code=409,
            detail="Episode is done. POST /reset to start a new episode.",
        )

    action = TrafficAction(action=req.action)
    result = _env.step(action)
    return result


@app.get("/state", response_model=TrafficState)
def get_state() -> TrafficState:
    """Return the full internal state snapshot (includes stats and config)."""
    return _env.state()


@app.post("/grade", response_model=GradeResponse)
def get_grade() -> GradeResponse:
    """
    Compute the task score for the current (or just-completed) episode.

    Can be called mid-episode for a partial score, or after done=true for
    the final score.
    """
    state = _env.state()
    score = grade(state.task, state.stats)
    return GradeResponse(
        task=state.task,
        score=score,
        stats=state.stats.model_dump(),
    )


@app.post("/close")
def close() -> Dict[str, str]:
    """
    Close the current episode (reset internal state without starting a new
    episode). Used by the inference script at the end of a run.
    """
    _env.reset(task=_env.state().task)
    return {"status": "closed"}


# ---------------------------------------------------------------------------
# Entry point (for direct python server/app.py execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
