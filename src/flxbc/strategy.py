from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

ParameterDict = dict[str, np.ndarray]


def aggregate_parameters(updates: list[ParameterDict], weights: list[float]) -> ParameterDict:
    if not updates:
        raise ValueError("at least one update is required")
    normalized = _normalize(weights)
    result: ParameterDict = {}
    for name in updates[0]:
        if not np.issubdtype(updates[0][name].dtype, np.floating):
            result[name] = updates[0][name].copy()
            continue
        result[name] = sum(
            update[name] * weight for update, weight in zip(updates, normalized, strict=True)
        )
    return result


def compute_bc_weights(records: list[Mapping[str, float | int | str]]) -> dict[str, float]:
    raw: dict[str, float] = {}
    for record in records:
        node_id = str(record["node_id"])
        samples = float(record.get("samples", 1.0))
        contribution = max(float(record.get("contribution", 0.0)), 0.02)
        reputation = min(max(float(record.get("reputation", 0.5)), 0.05), 1.0)
        raw[node_id] = samples * contribution * reputation
    total = sum(raw.values())
    if total <= 0 or not math.isfinite(total):
        even = 1.0 / max(len(raw), 1)
        return {node_id: even for node_id in raw}
    return {node_id: value / total for node_id, value in raw.items()}


def update_reputation(
    previous: float,
    *,
    contribution: float,
    misbehaved: bool,
    ema: float = 0.8,
) -> float:
    contribution = min(max(contribution, 0.0), 1.0)
    target = contribution
    if misbehaved:
        target *= 0.35
    updated = ema * previous + (1.0 - ema) * target
    if misbehaved:
        updated -= 0.08
    return min(max(updated, 0.05), 1.0)


def contribution_from_metrics(
    *,
    previous_accuracy: float,
    local_accuracy: float,
    update_norm: float,
    misbehaved: bool,
) -> float:
    if misbehaved:
        return 0.02
    improvement = local_accuracy - previous_accuracy
    stability = 1.0 / (1.0 + max(update_norm, 0.0))
    score = 0.45 + improvement + 0.25 * stability
    return min(max(score, 0.02), 1.0)


def _normalize(weights: list[float]) -> list[float]:
    if len(weights) == 0:
        raise ValueError("weights cannot be empty")
    total = float(sum(weights))
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(weights)] * len(weights)
    return [float(weight) / total for weight in weights]
