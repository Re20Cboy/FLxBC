from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flxbc.config import RunConfig
from flxbc.simulation import run_federated_demo


def run_demo_check(
    *,
    db_path: Path = Path("data/flxbc.db"),
    artifact_dir: Path = Path("artifacts"),
    run_id: str = "demo-check",
) -> dict[str, Any]:
    config = RunConfig(
        run_id=run_id,
        mode="demo-check",
        dataset="synthetic3d",
        strategy="bc-ca-fedprox",
        num_clients=3,
        rounds=2,
        local_epochs=1,
        batch_size=8,
        seed=7,
        max_train_samples=72,
        max_test_samples=24,
        use_synthetic=True,
        simulate_failures=False,
        device="numpy",
        db_path=db_path,
        artifact_dir=artifact_dir,
    )
    result = run_federated_demo(config)
    audit_blocks = result.ledger.get_audit_blocks(run_id)
    audit_status = result.ledger.verify_audit_chain(run_id)
    final_metrics = result.rounds[-1] if result.rounds else {}
    return {
        "run_id": run_id,
        "backend": result.backend,
        "rounds": len(result.rounds),
        "audit_blocks": len(audit_blocks),
        "audit_valid": audit_status["valid"],
        "head_block_hash": audit_status["head_block_hash"],
        "accuracy": final_metrics.get("accuracy", 0.0),
        "macro_f1": final_metrics.get("macro_f1", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FLxBC reproducible demo check")
    parser.add_argument("--db", type=Path, default=Path("data/flxbc.db"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", default="demo-check")
    args = parser.parse_args()
    summary = run_demo_check(db_path=args.db, artifact_dir=args.artifact_dir, run_id=args.run_id)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
