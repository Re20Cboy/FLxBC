from __future__ import annotations

from typing import Any


def short_hash(value: str | None, *, prefix: int = 10, suffix: int = 6) -> str:
    if not value:
        return "-"
    if len(value) <= prefix + suffix + 3:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


def audit_status_label(status: dict[str, Any]) -> str:
    if status.get("valid"):
        blocks = int(status.get("blocks", 0))
        return f"有效 / {blocks} blocks"
    broken_at = status.get("broken_at")
    if broken_at is None:
        return "异常"
    return f"异常 / block {broken_at}"


def related_experiment_run_ids(run_id: str, runs: list[dict[str, Any]]) -> list[str]:
    prefix = f"{run_id}-"
    return sorted(
        run["run_id"]
        for run in runs
        if isinstance(run.get("run_id"), str) and run["run_id"].startswith(prefix)
    )
