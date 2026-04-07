"""
Task Graders for Traffic Signal Control OpenEnv
================================================

Each grader accepts an ``EpisodeStats`` snapshot and returns a float in
[0.0, 1.0]. Graders are:
  - Deterministic (same inputs → same output, no randomness)
  - Reproducible  (no external state)
  - Normalised    (guaranteed to return values in [0.0, 1.0])

Task difficulty progression
---------------------------
  traffic_easy   → Basic throughput: can the agent clear most vehicles?
  traffic_medium → Queue balance: can the agent prevent lopsided queues?
  traffic_hard   → Peak-hour planning: multi-criteria performance under surge.
"""

from __future__ import annotations

import math
from typing import Dict

from .models import EpisodeStats, TASK_CONFIGS


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Task 1 — traffic_easy (Light Traffic)
# ---------------------------------------------------------------------------

def grade_easy(stats: EpisodeStats) -> float:
    """
    Score = vehicles_cleared / total_arrived.

    A perfect agent that always picks the right green phase and clears every
    vehicle that arrives will score 1.0. A random or idle agent will score
    well below 0.5.

    Edge-case: if no vehicles arrived at all, return 1.0 (trivially perfect).
    """
    if stats.total_arrived == 0:
        return 1.0

    raw = stats.total_cleared / stats.total_arrived
    return round(_clamp(raw), 4)


# ---------------------------------------------------------------------------
# Task 2 — traffic_medium (Moderate Traffic)
# ---------------------------------------------------------------------------

def grade_medium(stats: EpisodeStats) -> float:
    """
    Score balances throughput with a penalty for excessive waiting.

    Formula
    -------
      throughput_ratio = cleared / arrived
      wait_penalty     = total_wait_steps / (arrived × max_steps)
                         (normalised average wait per vehicle)
      score            = throughput_ratio × (1 - wait_penalty)

    A perfect agent scores close to 1.0.  An agent that clears many vehicles
    but lets queues build will be penalised.
    """
    if stats.total_arrived == 0:
        return 1.0

    cfg = TASK_CONFIGS["traffic_medium"]
    max_steps: int = cfg["max_steps"]

    throughput_ratio = _clamp(stats.total_cleared / stats.total_arrived)

    # Normalise wait steps: worst case every vehicle waits every step
    max_possible_wait = stats.total_arrived * max_steps
    wait_ratio = _clamp(
        stats.total_wait_steps / max_possible_wait if max_possible_wait > 0 else 0.0
    )

    score = throughput_ratio * (1.0 - wait_ratio)
    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Task 3 — traffic_hard (Peak-Hour Congestion)
# ---------------------------------------------------------------------------

def grade_hard(stats: EpisodeStats) -> float:
    """
    Weighted multi-criteria score for peak-hour performance.

    Components (weights sum to 1.0)
    --------------------------------
      throughput_score    (0.40)  — cleared / arrived
      wait_score          (0.35)  — inverse of normalised wait time
      congestion_score    (0.15)  — 1 − (congestion_events / max_steps)
      efficiency_score    (0.10)  — penalise excessive unnecessary switches

    Each component is in [0, 1] before weighting.
    """
    if stats.total_arrived == 0:
        return 1.0

    cfg = TASK_CONFIGS["traffic_hard"]
    max_steps: int = cfg["max_steps"]

    # 1. Throughput
    throughput_score = _clamp(stats.total_cleared / stats.total_arrived)

    # 2. Wait time (lower is better)
    max_possible_wait = stats.total_arrived * max_steps
    wait_ratio = _clamp(
        stats.total_wait_steps / max_possible_wait if max_possible_wait > 0 else 0.0
    )
    wait_score = 1.0 - wait_ratio

    # 3. Congestion avoidance (fewer congestion events is better)
    congestion_score = _clamp(1.0 - stats.congestion_events / max_steps)

    # 4. Switch efficiency (penalise unnecessary switches)
    if stats.total_switches == 0:
        efficiency_score = 1.0
    else:
        bad_ratio = stats.total_unnecessary_switches / stats.total_switches
        efficiency_score = _clamp(1.0 - bad_ratio)

    score = (
        0.40 * throughput_score
        + 0.35 * wait_score
        + 0.15 * congestion_score
        + 0.10 * efficiency_score
    )
    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS: Dict[str, callable] = {
    "traffic_easy":   grade_easy,
    "traffic_medium": grade_medium,
    "traffic_hard":   grade_hard,
}


def grade(task: str, stats: EpisodeStats) -> float:
    """
    Dispatch to the correct grader.

    Parameters
    ----------
    task : str
        Task name (one of ``GRADERS.keys()``).
    stats : EpisodeStats
        Cumulative stats collected during the episode.

    Returns
    -------
    float
        Normalised score in [0.0, 1.0].
    """
    if task not in GRADERS:
        raise ValueError(f"Unknown task {task!r}. Valid: {list(GRADERS)}")
    return GRADERS[task](stats)
