from pathlib import Path

from flxbc.ledger import Ledger, stable_hash


def test_stable_hash_is_order_independent_for_dicts():
    left = {"b": 2, "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": 2}

    assert stable_hash(left) == stable_hash(right)


def test_ledger_records_round_contribution_and_misbehavior(tmp_path: Path):
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")
    ledger.upsert_node("hospital_1", display_name="Hospital 1")

    ledger.record_round(
        run_id="run_1",
        round_id=1,
        metrics={"accuracy": 0.75},
        artifact_uri="artifacts/runs/run_1/round_1.pt",
        participants=["hospital_1"],
        tx_hash="0xabc",
    )
    ledger.record_contribution(
        run_id="run_1",
        round_id=1,
        node_id="hospital_1",
        samples=12,
        contribution=0.8,
        reputation=0.9,
        points=7.2,
    )
    ledger.record_misbehavior(
        run_id="run_1",
        round_id=1,
        node_id="hospital_1",
        kind="timeout",
        penalty=0.1,
        detail="simulated timeout",
    )

    assert ledger.list_runs()[0]["run_id"] == "run_1"
    assert ledger.get_rounds("run_1")[0]["metrics"]["accuracy"] == 0.75
    assert ledger.get_contributions("run_1")[0]["points"] == 7.2
    assert ledger.get_misbehavior("run_1")[0]["kind"] == "timeout"


def test_record_round_can_store_content_model_hash(tmp_path: Path):
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")

    hashes = ledger.record_round(
        run_id="run_1",
        round_id=1,
        metrics={"accuracy": 0.75},
        artifact_uri="artifacts/runs/run_1/round_1_parameters.npz",
        participants=["hospital_1"],
        tx_hash="mockchain:content",
        model_hash="content-model-hash",
    )

    row = ledger.get_rounds("run_1")[0]
    assert hashes["model_hash"] == "content-model-hash"
    assert row["model_hash"] == "content-model-hash"


def test_audit_chain_records_round_blocks_and_verifies_integrity(tmp_path: Path):
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")

    first = ledger.record_audit_block(
        run_id="run_1",
        round_id=1,
        payload={
            "artifact_uri": "artifacts/runs/run_1/round_1_parameters.npz",
            "metrics_hash": "metrics-1",
            "model_hash": "model-1",
            "participants_hash": "participants-1",
            "strategy_hash": "strategy-1",
        },
    )
    second = ledger.record_audit_block(
        run_id="run_1",
        round_id=2,
        payload={
            "artifact_uri": "artifacts/runs/run_1/round_2_parameters.npz",
            "metrics_hash": "metrics-2",
            "model_hash": "model-2",
            "participants_hash": "participants-2",
            "strategy_hash": "strategy-2",
        },
    )

    blocks = ledger.get_audit_blocks("run_1")
    status = ledger.verify_audit_chain("run_1")

    assert first["tx_hash"].startswith("mockchain:")
    assert second["previous_block_hash"] == first["block_hash"]
    assert [block["block_height"] for block in blocks] == [1, 2]
    assert status == {
        "valid": True,
        "blocks": 2,
        "head_block_hash": second["block_hash"],
        "broken_at": None,
    }


def test_audit_chain_verification_detects_tampering(tmp_path: Path):
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")
    ledger.record_audit_block(
        run_id="run_1",
        round_id=1,
        payload={
            "artifact_uri": "artifact",
            "metrics_hash": "metrics",
            "model_hash": "model",
            "participants_hash": "participants",
            "strategy_hash": "strategy",
        },
    )
    with ledger.connect() as conn:
        conn.execute(
            "UPDATE audit_blocks SET payload_json = ? WHERE run_id = ? AND block_height = ?",
            ('{"tampered": true}', "run_1", 1),
        )

    assert ledger.verify_audit_chain("run_1") == {
        "valid": False,
        "blocks": 1,
        "head_block_hash": None,
        "broken_at": 1,
    }


def test_create_run_resets_previous_run_records(tmp_path: Path):
    ledger = Ledger(tmp_path / "flxbc.db")
    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="bc-ca-fedprox")
    ledger.upsert_node("hospital_1", display_name="Hospital 1")
    ledger.record_round(
        run_id="run_1",
        round_id=1,
        metrics={"accuracy": 0.75},
        artifact_uri="artifact",
        participants=["hospital_1"],
        tx_hash="mockchain:old",
    )
    ledger.record_contribution(
        run_id="run_1",
        round_id=1,
        node_id="hospital_1",
        samples=12,
        contribution=0.8,
        reputation=0.9,
        points=7.2,
    )
    ledger.record_audit_block(
        run_id="run_1",
        round_id=1,
        payload={
            "artifact_uri": "artifact",
            "metrics_hash": "metrics",
            "model_hash": "model",
            "participants_hash": "participants",
            "strategy_hash": "strategy",
        },
    )

    ledger.create_run("run_1", mode="demo", dataset="synthetic", strategy="fedavg")

    assert ledger.get_rounds("run_1") == []
    assert ledger.get_contributions("run_1") == []
    assert ledger.get_audit_blocks("run_1") == []
