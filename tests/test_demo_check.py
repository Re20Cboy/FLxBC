from scripts.demo_check import run_demo_check


def test_demo_check_runs_synthetic_fl_and_verifies_audit_chain(tmp_path):
    summary = run_demo_check(
        db_path=tmp_path / "flxbc.db",
        artifact_dir=tmp_path / "artifacts",
        run_id="demo-check-test",
    )

    assert summary["run_id"] == "demo-check-test"
    assert summary["rounds"] == 2
    assert summary["audit_blocks"] == 2
    assert summary["audit_valid"] is True
    assert summary["backend"] == "numpy-prototype"
