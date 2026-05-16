import numpy as np

from flxbc.secure_aggregation import secure_aggregate_parameters
from flxbc.strategy import aggregate_parameters


def test_secure_aggregate_matches_plain_weighted_aggregate():
    updates = [
        {"layer": np.array([1.0, 3.0], dtype=np.float32)},
        {"layer": np.array([5.0, 7.0], dtype=np.float32)},
        {"layer": np.array([9.0, 11.0], dtype=np.float32)},
    ]
    weights = [1.0, 2.0, 3.0]

    secured = secure_aggregate_parameters(updates, weights, seed=123)
    plain = aggregate_parameters(updates, weights)

    assert secured.status == "applied"
    assert secured.fallback_reason is None
    np.testing.assert_allclose(secured.parameters["layer"], plain["layer"], rtol=1e-6)


def test_secure_aggregate_falls_back_when_client_is_missing():
    updates = [
        {"layer": np.array([1.0, 3.0], dtype=np.float32)},
        {"layer": np.array([5.0, 7.0], dtype=np.float32)},
    ]

    secured = secure_aggregate_parameters(
        updates,
        [1.0, 1.0],
        seed=123,
        expected_clients=3,
    )
    plain = aggregate_parameters(updates, [1.0, 1.0])

    assert secured.status == "fallback"
    assert secured.fallback_reason == "missing-clients"
    np.testing.assert_allclose(secured.parameters["layer"], plain["layer"], rtol=1e-6)
