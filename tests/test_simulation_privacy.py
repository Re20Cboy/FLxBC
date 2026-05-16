from pathlib import Path

import flxbc.simulation as simulation
from flxbc.config import RunConfig
from flxbc.privacy import hash_file


def test_privacy_demo_records_envelope_counts_and_content_model_hash(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("FLXBC_ARTIFACT_KEY", "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE=")
    monkeypatch.setenv("FLXBC_CLIENT_AUTH_SECRET", "test-demo-secret")
    config = RunConfig(
        run_id="privacy-smoke",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=1,
        seed=5,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        device="numpy",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        privacy_mode="full-demo",
        client_auth="hmac-demo",
        artifact_encryption=True,
        secure_aggregation=True,
        replay_window_seconds=120,
        dp_noise_multiplier=0.1,
        clipping_norm=0.5,
    )

    result = simulation.run_federated_demo(config)

    round_row = result.ledger.get_rounds("privacy-smoke")[0]
    artifact_path = Path(round_row["artifact_uri"])
    assert round_row["model_hash"] == hash_file(artifact_path)
    assert round_row["metrics"]["privacy_mode"] == "full-demo"
    assert round_row["metrics"]["accepted_clients"] == 2.0
    assert round_row["metrics"]["rejected_clients"] == 0.0
    assert round_row["metrics"]["dp_enabled"] is True
    assert round_row["metrics"]["dp_noise_multiplier"] == 0.1
    assert round_row["metrics"]["clipping_norm"] == 0.5
    assert round_row["metrics"]["mean_update_norm_before_clip"] >= 0.0
    assert round_row["metrics"]["mean_update_norm_after_clip"] <= 0.5

    audit_payload = result.ledger.get_audit_blocks("privacy-smoke")[0]["payload"]
    assert audit_payload["model_hash"] == round_row["model_hash"]
    assert "privacy_hash" in audit_payload


def test_hmac_demo_rejects_invalid_update_signatures(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLXBC_CLIENT_AUTH_SECRET", "test-demo-secret")
    monkeypatch.setattr(simulation, "sign_hmac_payload", lambda secret, payload: "bad")
    config = RunConfig(
        run_id="invalid-hmac",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=1,
        seed=6,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        device="numpy",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        client_auth="hmac-demo",
    )

    result = simulation.run_federated_demo(config)

    assert result.rounds == []
    misbehavior = result.ledger.get_misbehavior("invalid-hmac")
    assert [row["kind"] for row in misbehavior] == ["invalid-signature", "invalid-signature"]


def test_secure_aggregation_records_fallback_on_dropout(tmp_path: Path, monkeypatch):
    behaviors = iter(["dropout", None, None])
    monkeypatch.setattr(simulation, "_simulate_node_behavior", lambda config, rng: next(behaviors))
    config = RunConfig(
        run_id="secure-agg-fallback",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=3,
        rounds=1,
        seed=10,
        max_train_samples=36,
        max_test_samples=12,
        use_synthetic=True,
        device="numpy",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=True,
        secure_aggregation=True,
    )

    result = simulation.run_federated_demo(config)

    metrics = result.ledger.get_rounds("secure-agg-fallback")[0]["metrics"]
    assert metrics["secure_aggregation"] is True
    assert metrics["secure_aggregation_status"] == "fallback"
    assert metrics["secure_aggregation_fallback_reason"] == "missing-clients"
