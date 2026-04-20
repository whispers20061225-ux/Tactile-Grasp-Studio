from __future__ import annotations

from typing import Any

PLANNER_DB_ONLY = "planner_db_only"
PLANNER_RETRIEVAL_AUG = "planner_retrieval_aug"
DEFAULT_PLANNER_STRATEGY = PLANNER_DB_ONLY

PLANNER_STRATEGIES = {
    PLANNER_DB_ONLY,
    PLANNER_RETRIEVAL_AUG,
}

PLANNER_NUMERIC_FIELDS = (
    "target_force",
    "contact_threshold",
    "safety_max",
    "kp",
    "ki",
    "kd",
    "deadband",
    "max_step_per_tick",
    "move_time_ms",
    "poll_period_ms",
)


def normalize_planner_strategy(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in PLANNER_STRATEGIES:
        return normalized
    return DEFAULT_PLANNER_STRATEGY


def planner_promotes_rollup(value: Any) -> bool:
    return normalize_planner_strategy(value) == PLANNER_DB_ONLY
