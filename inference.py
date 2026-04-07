"""
Inference Script — Traffic Signal Control OpenEnv
==================================================

MANDATORY CONFIGURATION
-----------------------
Set these environment variables before running:

  export API_BASE_URL=<your_endpoint>      # e.g. https://router.huggingface.co/v1
  export MODEL_NAME=<your_model>           # e.g. Qwen/Qwen2.5-72B-Instruct
  export HF_TOKEN=<your_hf_or_api_key>

  Optional (for remote HF Space):
  export TRAFFIC_ENV_URL=http://localhost:7860   # defaults to localhost

STDOUT FORMAT (mandatory — do not modify)
-----------------------------------------
  [START] task=<task_name> env=traffic_env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

RUNNING
-------
  python inference.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_KEY: str        = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf-no-key"
API_BASE_URL: str   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str   = os.getenv("TRAFFIC_ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK: str      = "traffic_env"
MAX_STEPS: int      = 80          # hard upper bound across all tasks
SUCCESS_THRESHOLD: float = 0.4    # score ≥ this → success
TEMPERATURE: float  = 0.2         # low temp → more deterministic
MAX_TOKENS: int     = 50

TASKS: List[str] = ["traffic_easy", "traffic_medium", "traffic_hard"]

# ---------------------------------------------------------------------------
# Structured logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class TrafficEnvClient:
    """Thin synchronous HTTP wrapper around the traffic-env FastAPI server."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url
        self._http = httpx.Client(timeout=30.0)

    def reset(self, task: str, seed: int = 42) -> Dict[str, Any]:
        resp = self._http.post(
            f"{self._base}/reset",
            json={"task": task, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: str) -> Dict[str, Any]:
        resp = self._http.post(
            f"{self._base}/step",
            json={"action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self._http.get(f"{self._base}/state")
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> Dict[str, Any]:
        resp = self._http.post(f"{self._base}/grade")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        try:
            self._http.post(f"{self._base}/close")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LLM decision-making
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an intelligent traffic signal controller at a 4-way intersection.

OBSERVATION (given each step):
  north_queue   — vehicles waiting at the North approach
  south_queue   — vehicles waiting at the South approach
  east_queue    — vehicles waiting at the East approach
  west_queue    — vehicles waiting at the West approach
  current_signal — "NS" (North-South is green) or "EW" (East-West is green)
  phase_duration — how many steps the current signal has been active
  total_waiting  — total vehicles waiting

ACTIONS:
  switch_signal — toggle the green phase (NS ↔ EW)
  keep_signal   — leave the current phase unchanged

STRATEGY:
  - The green pair of approaches discharges 3 vehicles per step.
  - Switching penalises you slightly (-0.5) unless the other direction has a
    bigger queue.
  - You cannot switch before the signal has been active for at least 2 steps.
  - Minimise total waiting vehicles. Maximise throughput.

Reply with EXACTLY one word: either  switch_signal  or  keep_signal
""").strip()


def choose_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
) -> str:
    """Ask the LLM to choose an action given the current observation."""

    signal = obs["current_signal"]
    phase  = obs["phase_duration"]

    if signal == "NS":
        green_q = obs["north_queue"] + obs["south_queue"]
        red_q   = obs["east_queue"]  + obs["west_queue"]
        green_dir, red_dir = "NS", "EW"
    else:
        green_q = obs["east_queue"]  + obs["west_queue"]
        red_q   = obs["north_queue"] + obs["south_queue"]
        green_dir, red_dir = "EW", "NS"

    user_content = textwrap.dedent(f"""
        Step {step}
        Current green: {signal}  (active for {phase} steps)
        Green queue ({green_dir}): {green_q} vehicles
        Red queue   ({red_dir}): {red_q} vehicles
        Total waiting: {obs['total_waiting']}
        Time elapsed: {obs['time_elapsed']}

        Recent history:
        {chr(10).join(history[-4:]) if history else "None"}

        Choose: switch_signal or keep_signal?
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        # Parse — accept any response that contains the keyword
        if "switch" in text:
            return "switch_signal"
        return "keep_signal"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed (step {step}): {exc}", flush=True)
        # Fallback heuristic: switch if red queue is significantly larger
        return "switch_signal" if red_q > green_q + 2 else "keep_signal"


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(
    env_client: TrafficEnvClient,
    llm_client: OpenAI,
    task: str,
) -> None:
    """Run one full episode for a given task and emit the required log lines."""

    rewards: List[float] = []
    history: List[str]   = []
    steps_taken: int      = 0
    score: float          = 0.0
    success: bool         = False
    error_msg: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_client.reset(task=task, seed=42)
        obs    = result["observation"]
        done   = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action    = choose_action(llm_client, obs, step, history)
            step_result = env_client.step(action)

            obs    = step_result["observation"]
            reward = step_result.get("reward", 0.0)
            done   = step_result.get("done", False)
            info   = step_result.get("info", {})
            error  = None   # env HTTP errors are raised; None means ok

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action} → reward {reward:+.2f} "
                f"(cleared={info.get('vehicles_cleared', '?')})"
            )

        # Retrieve final grade from server
        grade_resp = env_client.grade()
        score   = float(grade_resp.get("score", 0.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Task {task} failed: {error_msg}", flush=True)
        success = False
        score   = 0.0

    finally:
        env_client.close()
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = TrafficEnvClient(base_url=ENV_BASE_URL)

    # Sanity-check: ping the server
    try:
        health = httpx.get(f"{ENV_BASE_URL}/health", timeout=10).json()
        print(f"[DEBUG] Server health: {health}", flush=True)
    except Exception as exc:
        print(
            f"[DEBUG] WARNING — could not reach environment at {ENV_BASE_URL}: {exc}",
            flush=True,
        )

    for task in TASKS:
        run_task(env_client, llm_client, task)
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    main()
