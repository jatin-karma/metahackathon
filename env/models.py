"""
Typed Pydantic models for the Traffic Signal Control OpenEnv environment.

These models define the full contract between the environment and any agent
or external consumer of the OpenEnv API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TrafficObservation(BaseModel):
    """
    The observation returned to the agent after every step (and on reset).

    Attributes
    ----------
    north_queue : int
        Number of vehicles waiting at the North approach.
    south_queue : int
        Number of vehicles waiting at the South approach.
    east_queue : int
        Number of vehicles waiting at the East approach.
    west_queue : int
        Number of vehicles waiting at the West approach.
    current_signal : str
        Which direction pair currently has the green light.
        "NS" means North-South is green, East-West is red.
        "EW" means East-West is green, North-South is red.
    time_elapsed : int
        Number of simulation steps elapsed in the current episode.
    phase_duration : int
        Number of consecutive steps the current signal phase has been active.
    total_waiting : int
        Sum of all queue lengths (quick summary for the agent).
    """

    north_queue: int = Field(ge=0, description="Vehicles queued at the North approach")
    south_queue: int = Field(ge=0, description="Vehicles queued at the South approach")
    east_queue: int = Field(ge=0, description="Vehicles queued at the East approach")
    west_queue: int = Field(ge=0, description="Vehicles queued at the West approach")
    current_signal: Literal["NS", "EW"] = Field(
        description="Active green signal phase: 'NS' or 'EW'"
    )
    time_elapsed: int = Field(ge=0, description="Steps elapsed in this episode")
    phase_duration: int = Field(ge=0, description="Steps the current phase has been active")
    total_waiting: int = Field(ge=0, description="Total vehicles waiting across all approaches")

    class Config:
        json_schema_extra = {
            "example": {
                "north_queue": 3,
                "south_queue": 2,
                "east_queue": 5,
                "west_queue": 1,
                "current_signal": "NS",
                "time_elapsed": 4,
                "phase_duration": 2,
                "total_waiting": 11,
            }
        }


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TrafficAction(BaseModel):
    """
    The action the agent submits each step.

    Attributes
    ----------
    action : str
        One of:
          - "switch_signal" — toggle the green phase from NS↔EW.
          - "keep_signal"   — leave the current phase unchanged.
    """

    action: Literal["switch_signal", "keep_signal"] = Field(
        description="Agent action: 'switch_signal' or 'keep_signal'"
    )

    class Config:
        json_schema_extra = {"example": {"action": "switch_signal"}}


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Itemised reward components for interpretability."""

    vehicles_cleared: float = Field(description="+1.0 per vehicle that passed through")
    waiting_penalty: float = Field(description="-0.1 per waiting vehicle per step")
    switch_penalty: float = Field(description="-0.5 for an unnecessary signal switch")
    congestion_penalty: float = Field(description="-1.0 per approach queue above threshold")
    total: float = Field(description="Sum of all components")


# ---------------------------------------------------------------------------
# Episode statistics (internal, also surfaced in info)
# ---------------------------------------------------------------------------

class EpisodeStats(BaseModel):
    """Cumulative statistics tracked over an episode."""

    total_arrived: int = Field(default=0, description="Vehicles that arrived at the intersection")
    total_cleared: int = Field(default=0, description="Vehicles that passed through")
    total_wait_steps: int = Field(default=0, description="Cumulative vehicle-steps spent waiting")
    total_switches: int = Field(default=0, description="Number of signal switches performed")
    total_unnecessary_switches: int = Field(default=0, description="Switches that were penalised")
    congestion_events: int = Field(default=0, description="Steps where any queue exceeded threshold")
    cumulative_reward: float = Field(default=0.0, description="Sum of all step rewards so far")


# ---------------------------------------------------------------------------
# Full internal state snapshot
# ---------------------------------------------------------------------------

class TrafficState(BaseModel):
    """
    Complete internal state snapshot — returned by the /state endpoint.

    This is a superset of the observation and is intended for debugging,
    evaluation, and grader access.
    """

    observation: TrafficObservation
    task: str = Field(description="Active task name")
    max_steps: int = Field(description="Maximum steps for this episode")
    done: bool = Field(description="Whether the episode has ended")
    stats: EpisodeStats
    config: Dict[str, Any] = Field(description="Task configuration parameters")


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    The complete result returned by step() and reset().

    This mirrors the OpenEnv standard: (observation, reward, done, info).
    """

    observation: TrafficObservation
    reward: float = Field(description="Scalar reward for this step")
    done: bool = Field(description="True when the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information (reward breakdown, stats, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "north_queue": 2,
                    "south_queue": 1,
                    "east_queue": 4,
                    "west_queue": 0,
                    "current_signal": "EW",
                    "time_elapsed": 5,
                    "phase_duration": 1,
                    "total_waiting": 7,
                },
                "reward": 0.7,
                "done": False,
                "info": {
                    "vehicles_cleared": 2,
                    "waiting_penalty": -0.7,
                    "switch_penalty": 0.0,
                    "congestion_penalty": 0.0,
                },
            }
        }


# ---------------------------------------------------------------------------
# Task definitions (used by openenv.yaml and graders)
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "traffic_easy": {
        "description": "Light traffic — low arrival rate, basic signal timing.",
        "arrival_rate_range": (1, 2),   # vehicles per step per direction
        "service_rate": 3,              # vehicles cleared per step per green direction
        "max_steps": 20,
        "congestion_threshold": 8,
        "surge": False,
        "variable_arrivals": False,
    },
    "traffic_medium": {
        "description": "Moderate traffic — dynamic inflow, requires queue balancing.",
        "arrival_rate_range": (2, 4),
        "service_rate": 3,
        "max_steps": 40,
        "congestion_threshold": 10,
        "surge": False,
        "variable_arrivals": True,
    },
    "traffic_hard": {
        "description": "Peak-hour congestion — heavy traffic surge, long-term planning.",
        "arrival_rate_range": (4, 8),
        "service_rate": 3,
        "max_steps": 60,
        "congestion_threshold": 12,
        "surge": True,
        "variable_arrivals": True,
    },
}

VALID_TASKS: List[str] = list(TASK_CONFIGS.keys())
