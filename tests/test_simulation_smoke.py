import json
from pathlib import Path

from flxbc.config import RunConfig
from flxbc.simulation import run_federated_demo


def test_synthetic_federated_demo_records_rounds_and_artifacts(tmp_path: Path):
    config = RunConfig(
        run_id="smoke",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=3,
        rounds=2,
        local_epochs=1,
        batch_size=8,
        seed=3,
        max_train_samples=72,
        max_test_samples=24,
        use_synthetic=True,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=True,
    )

    result = run_federated_demo(config)

    assert result.run_id == "smoke"
    assert len(result.rounds) == 2
    final_metrics = result.rounds[-1]
    for split in ("train", "val", "test"):
        assert f"{split}_loss" in final_metrics
        assert f"{split}_accuracy" in final_metrics
        assert f"{split}_macro_f1" in final_metrics
    assert final_metrics["loss"] == final_metrics["test_loss"]
    assert final_metrics["accuracy"] == final_metrics["test_accuracy"]
    assert final_metrics["download_bytes"] > 0
    assert final_metrics["upload_bytes"] > 0
    assert final_metrics["communication_bytes"] > 0
    assert final_metrics["cumulative_communication_bytes"] >= final_metrics["communication_bytes"]
    assert final_metrics["round_duration_seconds"] > 0
    assert final_metrics["local_train_duration_seconds"] > 0
    assert final_metrics["aggregation_duration_seconds"] >= 0
    assert final_metrics["evaluation_duration_seconds"] > 0
    assert final_metrics["client_accuracy_mean"] >= 0
    assert final_metrics["client_accuracy_std"] >= 0
    assert final_metrics["client_loss_mean"] >= 0
    assert final_metrics["client_loss_std"] >= 0
    assert (tmp_path / "artifacts" / "runs" / "smoke" / "round_2_metrics.json").exists()
    assert len(result.ledger.get_rounds("smoke")) == 2
    assert result.ledger.get_contributions("smoke")
    assert len(result.ledger.get_audit_blocks("smoke")) == 2
    assert result.ledger.verify_audit_chain("smoke")["valid"] is True


def test_early_stopping_stops_after_validation_metric_stalls(tmp_path: Path):
    config = RunConfig(
        run_id="early-stop",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=8,
        local_epochs=1,
        batch_size=8,
        seed=3,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        device="numpy",
        early_stopping=True,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_patience=1,
        early_stopping_min_delta=10.0,
        min_rounds=2,
    )

    result = run_federated_demo(config)

    assert len(result.rounds) < 8
    final_metrics = result.rounds[-1]
    assert final_metrics["early_stopped"] is True
    assert final_metrics["stop_reason"] == "patience-exhausted"
    assert final_metrics["best_round"] >= 1
    summary = json.loads(
        (tmp_path / "artifacts" / "runs" / "early-stop" / "summary.json").read_text()
    )
    assert summary["best_round"] == int(final_metrics["best_round"])
    assert summary["best_metrics"]["round"] == final_metrics["best_round"]


def test_time_to_target_records_threshold_round_seconds_and_communication(tmp_path: Path):
    config = RunConfig(
        run_id="target-stop",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=5,
        local_epochs=1,
        batch_size=8,
        seed=5,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        device="numpy",
        early_stopping=True,
        target_accuracy=0.0,
    )

    result = run_federated_demo(config)

    assert len(result.rounds) == 1
    final_metrics = result.rounds[-1]
    assert final_metrics["early_stopped"] is True
    assert final_metrics["stop_reason"] == "target-accuracy"
    assert final_metrics["target_reached"] is True
    assert final_metrics["target_reason"] == "target-accuracy"
    assert final_metrics["time_to_target_round"] == 1
    assert final_metrics["time_to_target_seconds"] > 0
    assert (
        final_metrics["communication_bytes_at_target"]
        == final_metrics["cumulative_communication_bytes"]
    )


def test_adaptive_rounds_extend_budget_before_max_stop(tmp_path: Path):
    config = RunConfig(
        run_id="adaptive-rounds",
        mode="demo",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=2,
        rounds=2,
        local_epochs=1,
        batch_size=8,
        seed=7,
        max_train_samples=24,
        max_test_samples=12,
        use_synthetic=True,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "flxbc.db",
        simulate_failures=False,
        device="numpy",
        early_stopping=True,
        early_stopping_patience=99,
        adaptive_rounds=True,
        round_extension=2,
        max_rounds_cap=4,
    )

    result = run_federated_demo(config)

    assert len(result.rounds) == 4
    assert result.rounds[1]["round_limit_extended"] is True
    assert result.rounds[-1]["round_limit"] == 4
    assert result.rounds[-1]["stop_reason"] == "max-rounds-reached"
