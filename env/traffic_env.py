"""
Traffic Signal Control — Core Simulation Engine
================================================

Simulates a 4-way intersection (North / South / East / West) with adaptive
traffic signal control. The agent's job is to decide whether to switch the
green phase (NS ↔ EW) or keep it unchanged at each discrete time step.

Simulation rules
----------------
* At every step, vehicles arrive randomly at each approach (Poisson process).
* Green-phase approaches discharge vehicles at a fixed service rate per step.
* Switching the signal incurs a minimum-green constraint: the new phase must
  be held for at least `MIN_GREEN_STEPS` before it can switch again.
* Episode ends when `max_steps` is reached.

Reward function (dense)
-----------------------
  +1.0   per vehicle cleared from a green approach
  -0.1   per vehicle waiting anywhere (applied each step)
  -0.5   for an unnecessary switch (switched when the green queue was not
          smaller than the red queue, i.e. no good reason to switch)
  -1.0   per approach whose queue exceeds `congestion_threshold`
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .models import (
    EpisodeStats,
    RewardBreakdown,
    StepResult,
    TASK_CONFIGS,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    VALID_TASKS,
)

MIN_GREEN_STEPS = 2   # minimum steps before a switch is allowed


class TrafficSignalEnv:
    """Adaptive traffic signal control environment (OpenEnv compatible)."""

    # ------------------------------------------------------------------
    # Construction / reset
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._task: str = "traffic_easy"
        self._cfg: Dict[str, Any] = deepcopy(TASK_CONFIGS["traffic_easy"])
        self._rng = random.Random(42)   # deterministic seed; reset() re-seeds
        self._reset_state()

    def _reset_state(self) -> None:
        """Zero-out all mutable episode state."""
        self._queues: Dict[str, int] = {
            "north": 0, "south": 0, "east": 0, "west": 0
        }
        self._signal: str = "NS"          # "NS" or "EW"
        self._time_elapsed: int = 0
        self._phase_duration: int = 0
        self._done: bool = False
        self._stats = EpisodeStats()
        # For surge tasks: track when surge begins
        self._surge_active: bool = False

    def reset(self, task: Optional[str] = None, seed: int = 42) -> StepResult:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        task : str, optional
            One of ``VALID_TASKS``. Defaults to ``"traffic_easy"``.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        StepResult
            Initial observation with reward=0, done=False.
        """
        if task is not None:
            if task not in VALID_TASKS:
                raise ValueError(
                    f"Unknown task {task!r}. Valid: {VALID_TASKS}"
                )
            self._task = task
            self._cfg = deepcopy(TASK_CONFIGS[task])

        self._rng = random.Random(seed)
        self._reset_state()

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"task": self._task, "seed": seed},
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: TrafficAction) -> StepResult:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        action : TrafficAction
            ``"switch_signal"`` or ``"keep_signal"``.

        Returns
        -------
        StepResult
            Next observation, reward, done flag, and diagnostic info.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )

        act = action.action

        # 1. Vehicles arrive (before signal acts so agent sees updated queues)
        self._arrive_vehicles()

        # 2. Apply action
        switched, unnecessary = self._apply_action(act)

        # 3. Discharge vehicles through green signal
        cleared = self._discharge_vehicles()

        # 4. Advance time
        self._time_elapsed += 1
        self._phase_duration += 1

        # 5. Compute reward
        reward_breakdown = self._compute_reward(cleared, switched, unnecessary)
        reward = reward_breakdown.total

        # 6. Update episode stats
        self._stats.total_cleared += cleared
        if switched:
            self._stats.total_switches += 1
        if unnecessary:
            self._stats.total_unnecessary_switches += 1
        for q in self._queues.values():
            if q > self._cfg["congestion_threshold"]:
                self._stats.congestion_events += 1
                break
        self._stats.cumulative_reward += reward

        # 7. Check termination
        self._done = self._time_elapsed >= self._cfg["max_steps"]

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info={
                "vehicles_cleared": cleared,
                "switched": switched,
                "unnecessary_switch": unnecessary,
                "reward_breakdown": reward_breakdown.model_dump(),
                "stats": self._stats.model_dump(),
            },
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state(self) -> TrafficState:
        """Return a full snapshot of the environment's internal state."""
        return TrafficState(
            observation=self._make_observation(),
            task=self._task,
            max_steps=self._cfg["max_steps"],
            done=self._done,
            stats=deepcopy(self._stats),
            config=deepcopy(self._cfg),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _arrive_vehicles(self) -> None:
        """Sample Poisson arrivals for each approach."""
        lo, hi = self._cfg["arrival_rate_range"]

        # Surge: in the second half of the hard episode, double arrivals
        if self._cfg.get("surge") and self._time_elapsed >= self._cfg["max_steps"] // 2:
            lo, hi = lo * 2, hi * 2
            self._surge_active = True

        # Variable arrivals: random per-step jitter
        if self._cfg.get("variable_arrivals"):
            for direction in ("north", "south", "east", "west"):
                rate = self._rng.randint(lo, hi)
                arrived = self._poisson(rate)
                self._queues[direction] += arrived
                self._stats.total_arrived += arrived
        else:
            rate = self._rng.randint(lo, hi)
            for direction in ("north", "south", "east", "west"):
                arrived = self._poisson(rate)
                self._queues[direction] += arrived
                self._stats.total_arrived += arrived

    def _poisson(self, lam: float) -> int:
        """Approximate Poisson sample via sum-of-Bernoulli (simple & deterministic)."""
        # Using the rng for reproducibility
        count = 0
        for _ in range(int(lam) + 3):          # generous upper bound
            if self._rng.random() < lam / (int(lam) + 3):
                count += 1
        return count

    def _apply_action(self, act: str) -> Tuple[bool, bool]:
        """
        Execute the agent's action.

        Returns
        -------
        switched : bool
            Whether the signal was actually toggled this step.
        unnecessary : bool
            Whether the switch was penalised as unnecessary.
        """
        switched = False
        unnecessary = False

        if act == "switch_signal":
            # Minimum-green constraint
            if self._phase_duration < MIN_GREEN_STEPS:
                # Forced keep — penalise as if they tried to switch too early
                unnecessary = True
                return switched, unnecessary

            # Evaluate whether switch is beneficial
            if self._signal == "NS":
                green_total = self._queues["north"] + self._queues["south"]
                red_total   = self._queues["east"]  + self._queues["west"]
            else:
                green_total = self._queues["east"]  + self._queues["west"]
                red_total   = self._queues["north"] + self._queues["south"]

            # Penalise if the agent switched away from the longer queue
            if green_total > red_total:
                unnecessary = True

            # Perform switch
            self._signal = "EW" if self._signal == "NS" else "NS"
            self._phase_duration = 0
            switched = True

        # "keep_signal" → do nothing

        return switched, unnecessary

    def _discharge_vehicles(self) -> int:
        """
        Clear vehicles from green-phase approaches.

        Returns
        -------
        int
            Total vehicles cleared this step.
        """
        service_rate: int = self._cfg["service_rate"]
        cleared = 0

        if self._signal == "NS":
            for direction in ("north", "south"):
                remove = min(self._queues[direction], service_rate)
                self._queues[direction] -= remove
                cleared += remove
        else:
            for direction in ("east", "west"):
                remove = min(self._queues[direction], service_rate)
                self._queues[direction] -= remove
                cleared += remove

        return cleared

    def _compute_reward(
        self,
        cleared: int,
        switched: bool,
        unnecessary: bool,
    ) -> RewardBreakdown:
        """Calculate the dense reward signal for this step."""
        threshold = self._cfg["congestion_threshold"]

        # +1.0 per vehicle cleared
        r_cleared = float(cleared) * 1.0

        # -0.1 per waiting vehicle
        total_waiting = sum(self._queues.values())
        r_waiting = -0.1 * total_waiting

        # -0.5 for unnecessary switch
        r_switch = -0.5 if unnecessary else 0.0

        # -1.0 per congested approach
        congested = sum(1 for q in self._queues.values() if q > threshold)
        r_congestion = -1.0 * congested

        total = r_cleared + r_waiting + r_switch + r_congestion

        return RewardBreakdown(
            vehicles_cleared=round(r_cleared, 4),
            waiting_penalty=round(r_waiting, 4),
            switch_penalty=round(r_switch, 4),
            congestion_penalty=round(r_congestion, 4),
            total=round(total, 4),
        )

    def _make_observation(self) -> TrafficObservation:
        """Build the observation dict from current state."""
        total_waiting = sum(self._queues.values())
        return TrafficObservation(
            north_queue=self._queues["north"],
            south_queue=self._queues["south"],
            east_queue=self._queues["east"],
            west_queue=self._queues["west"],
            current_signal=self._signal,
            time_elapsed=self._time_elapsed,
            phase_duration=self._phase_duration,
            total_waiting=total_waiting,
        )
