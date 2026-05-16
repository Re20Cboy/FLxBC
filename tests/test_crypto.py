from dataclasses import replace

import numpy as np

from flxbc.crypto import demo_client_secret, sign_hmac_payload, verify_hmac_payload
from flxbc.privacy import (
    create_parameter_envelope,
    envelope_signature_fields,
    validate_parameter_envelope,
)


def test_hmac_payload_signature_rejects_changed_fields_and_wrong_secret():
    fields = {"run_id": "run-1", "round_id": 1, "node_id": "hospital_1"}
    secret = demo_client_secret("run-1", "hospital_1", master_secret="master")
    signature = sign_hmac_payload(secret, fields)

    assert verify_hmac_payload(secret, fields, signature) is True
    assert verify_hmac_payload(secret, {**fields, "round_id": 2}, signature) is False

    wrong_secret = demo_client_secret("run-1", "hospital_2", master_secret="master")
    assert verify_hmac_payload(wrong_secret, fields, signature) is False


def test_envelope_validation_requires_valid_hmac_signature():
    parameters = {"layer": np.array([1.0, 2.0], dtype=np.float32)}
    unsigned = create_parameter_envelope(
        parameters,
        run_id="run-1",
        round_id=1,
        node_id="hospital_1",
        privacy_mode="full-demo",
        timestamp=100.0,
        nonce="nonce-1",
    )
    secret = demo_client_secret("run-1", "hospital_1", master_secret="master")
    signed = replace(
        unsigned,
        signature=sign_hmac_payload(secret, envelope_signature_fields(unsigned)),
    )

    valid, reason = validate_parameter_envelope(
        signed,
        expected_run_id="run-1",
        expected_round_id=1,
        expected_node_id="hospital_1",
        seen_nonces=set(),
        now=100.0,
        replay_window_seconds=300,
        signature_secret=secret,
    )
    assert valid is True
    assert reason is None

    invalid, invalid_reason = validate_parameter_envelope(
        replace(signed, signature="bad-signature"),
        expected_run_id="run-1",
        expected_round_id=1,
        expected_node_id="hospital_1",
        seen_nonces=set(),
        now=100.0,
        replay_window_seconds=300,
        signature_secret=secret,
    )
    assert invalid is False
    assert invalid_reason == "invalid-signature"
