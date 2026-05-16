from dataclasses import replace

import numpy as np

from flxbc.privacy import (
    apply_update_privacy,
    create_parameter_envelope,
    deserialize_parameters,
    hash_parameters,
    validate_parameter_envelope,
)


def test_parameter_hash_is_stable_and_content_based():
    first = {
        "b": np.array([3.0, 4.0], dtype=np.float32),
        "a": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    second = {
        "a": np.array([[1.0, 2.0]], dtype=np.float32),
        "b": np.array([3.0, 4.0], dtype=np.float32),
    }
    changed = {
        "a": np.array([[1.0, 2.0]], dtype=np.float32),
        "b": np.array([3.0, 4.1], dtype=np.float32),
    }

    assert hash_parameters(first) == hash_parameters(second)
    assert hash_parameters(first) != hash_parameters(changed)


def test_apply_update_privacy_clips_update_norm():
    global_parameters = {"layer": np.array([0.0, 0.0], dtype=np.float32)}
    local_parameters = {"layer": np.array([3.0, 4.0], dtype=np.float32)}

    result = apply_update_privacy(
        local_parameters,
        global_parameters,
        clipping_norm=2.0,
        dp_noise_multiplier=0.0,
        rng=np.random.default_rng(1),
    )

    assert result.before_clip_norm == 5.0
    assert result.after_clip_norm == 2.0
    np.testing.assert_allclose(result.parameters["layer"], np.array([1.2, 1.6]))


def test_apply_update_privacy_adds_reproducible_noise():
    global_parameters = {"layer": np.array([0.0, 0.0], dtype=np.float32)}
    local_parameters = {"layer": np.array([1.0, 0.0], dtype=np.float32)}

    first = apply_update_privacy(
        local_parameters,
        global_parameters,
        clipping_norm=1.0,
        dp_noise_multiplier=0.5,
        rng=np.random.default_rng(9),
    )
    second = apply_update_privacy(
        local_parameters,
        global_parameters,
        clipping_norm=1.0,
        dp_noise_multiplier=0.5,
        rng=np.random.default_rng(9),
    )

    np.testing.assert_allclose(first.parameters["layer"], second.parameters["layer"])
    assert first.noise_std == 0.5
    assert not np.allclose(first.parameters["layer"], local_parameters["layer"])


def test_parameter_envelope_round_trips_payload_and_rejects_tampering():
    parameters = {"layer": np.array([1.0, 2.0], dtype=np.float32)}
    envelope = create_parameter_envelope(
        parameters,
        run_id="run-1",
        round_id=1,
        node_id="hospital_1",
        privacy_mode="full-demo",
        timestamp=100.0,
        nonce="nonce-1",
    )

    decoded = deserialize_parameters(envelope.payload_bytes)
    np.testing.assert_allclose(decoded["layer"], parameters["layer"])

    seen_nonces: set[tuple[str, int, str, str]] = set()
    valid, reason = validate_parameter_envelope(
        envelope,
        expected_run_id="run-1",
        expected_round_id=1,
        expected_node_id="hospital_1",
        seen_nonces=seen_nonces,
        now=100.0,
        replay_window_seconds=300,
    )
    assert valid is True
    assert reason is None

    replay_valid, replay_reason = validate_parameter_envelope(
        envelope,
        expected_run_id="run-1",
        expected_round_id=1,
        expected_node_id="hospital_1",
        seen_nonces=seen_nonces,
        now=100.0,
        replay_window_seconds=300,
    )
    assert replay_valid is False
    assert replay_reason == "replay"

    tampered = replace(envelope, payload_bytes=envelope.payload_bytes + b"x")
    tamper_valid, tamper_reason = validate_parameter_envelope(
        tampered,
        expected_run_id="run-1",
        expected_round_id=1,
        expected_node_id="hospital_1",
        seen_nonces=set(),
        now=100.0,
        replay_window_seconds=300,
    )
    assert tamper_valid is False
    assert tamper_reason == "hash-mismatch"
