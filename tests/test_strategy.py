import numpy as np

from flxbc.strategy import aggregate_parameters, compute_bc_weights, update_reputation


def test_aggregate_parameters_uses_normalized_weights():
    updates = [
        {"layer": np.array([1.0, 3.0])},
        {"layer": np.array([5.0, 7.0])},
    ]

    aggregated = aggregate_parameters(updates, [1.0, 3.0])

    np.testing.assert_allclose(aggregated["layer"], np.array([4.0, 6.0]))


def test_bc_weights_reward_reliable_high_contribution_nodes():
    records = [
        {"node_id": "hospital_1", "samples": 100, "contribution": 0.9, "reputation": 0.9},
        {"node_id": "hospital_2", "samples": 100, "contribution": 0.2, "reputation": 0.4},
    ]

    weights = compute_bc_weights(records)

    assert weights["hospital_1"] > weights["hospital_2"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_reputation_penalizes_misbehavior_and_caps_value():
    assert update_reputation(0.9, contribution=1.0, misbehaved=False) <= 1.0
    assert update_reputation(0.9, contribution=0.5, misbehaved=True) < 0.9
    assert update_reputation(0.05, contribution=0.0, misbehaved=True) >= 0.05
