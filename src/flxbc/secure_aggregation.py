from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flxbc.strategy import ParameterDict, aggregate_parameters


@dataclass(frozen=True, slots=True)
class SecureAggregationResult:
    parameters: ParameterDict
    status: str
    fallback_reason: str | None


def secure_aggregate_parameters(
    updates: list[ParameterDict],
    weights: list[float],
    *,
    seed: int,
    expected_clients: int | None = None,
) -> SecureAggregationResult:
    if expected_clients is not None and len(updates) != expected_clients:
        return SecureAggregationResult(
            parameters=aggregate_parameters(updates, weights),
            status="fallback",
            fallback_reason="missing-clients",
        )
    if len(updates) < 2:
        return SecureAggregationResult(
            parameters=aggregate_parameters(updates, weights),
            status="fallback",
            fallback_reason="not-enough-clients",
        )

    normalized = _normalize(weights)
    scaled_updates = _scale_updates(updates, normalized)
    masked_updates = _apply_zero_sum_masks(scaled_updates, seed=seed)
    aggregated: ParameterDict = {}
    for name, first_value in masked_updates[0].items():
        if not np.issubdtype(first_value.dtype, np.floating):
            aggregated[name] = first_value.copy()
            continue
        aggregated[name] = sum(update[name] for update in masked_updates)
    return SecureAggregationResult(
        parameters=aggregated,
        status="applied",
        fallback_reason=None,
    )


def _scale_updates(updates: list[ParameterDict], weights: list[float]) -> list[ParameterDict]:
    scaled: list[ParameterDict] = []
    for update, weight in zip(updates, weights, strict=True):
        scaled_update: ParameterDict = {}
        for name, value in update.items():
            if np.issubdtype(value.dtype, np.floating):
                scaled_update[name] = value * weight
            else:
                scaled_update[name] = value.copy()
        scaled.append(scaled_update)
    return scaled


def _apply_zero_sum_masks(updates: list[ParameterDict], *, seed: int) -> list[ParameterDict]:
    rng = np.random.default_rng(seed)
    masked = [{name: value.copy() for name, value in update.items()} for update in updates]
    for name, first_value in updates[0].items():
        if not np.issubdtype(first_value.dtype, np.floating):
            continue
        masks = [
            rng.normal(0.0, 1.0, size=first_value.shape).astype(first_value.dtype)
            for _ in range(len(updates) - 1)
        ]
        final_mask = -sum(masks)
        for masked_update, mask in zip(masked[:-1], masks, strict=True):
            masked_update[name] = masked_update[name] + mask
        masked[-1][name] = masked[-1][name] + final_mask
    return masked


def _normalize(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [float(weight) / total for weight in weights]
