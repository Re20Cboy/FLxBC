from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from flxbc.config import RunConfig
from flxbc.ledger import Ledger
from flxbc.simulation import run_centralized_baseline, run_comparison_suite, run_federated_demo


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "init-db":
        Ledger(args.db)
        print(f"Initialized SQLite ledger at {args.db}")
        return

    run_id = args.run_id or f"{args.command}-{int(time.time())}"
    if args.command == "demo":
        config = _config_from_args(args, run_id=run_id, mode=args.command)
        result = run_federated_demo(config)
        final = result.rounds[-1] if result.rounds else {}
        print(f"Run {result.run_id} finished using {result.backend}: {final}")
    elif args.command == "experiment":
        config = _config_from_args(args, run_id=run_id, mode=args.command)
        results = run_comparison_suite(config)
        print(f"Comparison run {config.run_id} finished: {results}")
    elif args.command == "run":
        config = _config_from_args(
            args,
            run_id=run_id,
            mode="demo",
            db_path=Path(args.db),
        )
        result = run_federated_demo(config)
        final = result.rounds[-1] if result.rounds else {}
        print(f"Run {result.run_id} finished using {result.backend}: {final}")

        print("\n[Comparison] Running centralized baseline ...")
        centralized = run_centralized_baseline(config)
        ledger = Ledger(config.db_path)
        ledger.record_experiment_result(
            run_id=config.run_id, label=f"federated ({config.strategy})", metrics=final
        )
        print(
            f"[Comparison] centralized:   acc={centralized.get('accuracy', 0):.3f}  "
            f"f1={centralized.get('macro_f1', 0):.3f}\n"
            f"[Comparison] {config.strategy}:  acc={final.get('accuracy', 0):.3f}  "
            f"f1={final.get('macro_f1', 0):.3f}"
        )

        if not args.no_dashboard:
            db_path = str(config.db_path)
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ]
            env = os.environ.copy()
            env["FLXBC_DB"] = db_path
            proc = subprocess.Popen(cmd, env=env)
            print(f"Dashboard launching (PID {proc.pid}) ...")
            time.sleep(3)
            webbrowser.open(f"http://127.0.0.1:8501/?run_id={result.run_id}")
            print("Press Ctrl+C to stop the dashboard.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLxBC local Mac federated learning demo")
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_db = subcommands.add_parser("init-db", help="Initialize SQLite ledger")
    init_db.add_argument("--db", default="data/flxbc.db")

    demo = subcommands.add_parser("demo", help="Run 5-client quick local demo")
    _add_run_args(demo)
    demo.add_argument("--rounds", type=int, default=10)
    demo.add_argument("--clients", type=int, default=5)

    experiment = subcommands.add_parser("experiment", help="Run 8-client comparison suite")
    _add_run_args(experiment)
    experiment.add_argument("--rounds", type=int, default=30)
    experiment.add_argument("--clients", type=int, default=8)

    run_cmd = subcommands.add_parser("run", help="Run demo then launch dashboard")
    _add_run_args(run_cmd)
    run_cmd.add_argument("--rounds", type=int, default=10)
    run_cmd.add_argument("--clients", type=int, default=5)
    run_cmd.add_argument("--db", default="data/flxbc.db")
    run_cmd.add_argument("--no-dashboard", action="store_true", help="Skip dashboard launch")

    return parser


def _config_from_args(
    args: argparse.Namespace,
    *,
    run_id: str,
    mode: str,
    db_path: Path | None = None,
) -> RunConfig:
    kwargs = dict(
        run_id=run_id,
        mode=mode,
        dataset=args.dataset,
        strategy=args.strategy,
        num_clients=args.clients,
        rounds=args.rounds,
        seed=args.seed,
        iid=args.iid,
        use_synthetic=args.synthetic,
        simulate_failures=not args.no_failures,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        device=args.device,
        privacy_mode=args.privacy_mode,
        client_auth=args.client_auth,
        transport_security=args.transport_security,
        artifact_encryption=args.artifact_encryption,
        secure_aggregation=args.secure_aggregation,
        replay_window_seconds=args.replay_window_seconds,
        dp_noise_multiplier=args.dp_noise_multiplier,
        clipping_norm=args.clipping_norm,
        early_stopping=args.early_stopping,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        min_rounds=args.min_rounds,
        target_accuracy=args.target_accuracy,
        target_macro_f1=args.target_macro_f1,
        target_loss=args.target_loss,
        adaptive_rounds=args.adaptive_rounds,
        round_extension=args.round_extension,
        max_rounds_cap=args.max_rounds_cap,
    )
    if db_path is not None:
        kwargs["db_path"] = db_path
    return RunConfig(**kwargs)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--dataset", default="nodulemnist3d")
    parser.add_argument(
        "--strategy", default="bc-ca-fedprox", choices=["fedavg", "fedprox", "bc-ca-fedprox"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=800)
    parser.add_argument("--max-test-samples", type=int, default=240)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "numpy"])
    parser.add_argument(
        "--privacy-mode",
        default="none",
        choices=["none", "dp", "encrypted", "secure-aggregation", "full-demo"],
        help="Privacy demo mode to record in run configuration",
    )
    parser.add_argument(
        "--client-auth",
        default="none",
        choices=["none", "hmac-demo"],
        help="Client update authentication mode",
    )
    parser.add_argument(
        "--transport-security",
        default="local-only",
        choices=["local-only", "tls", "mtls"],
        help="Transport-security boundary expected by this run",
    )
    parser.add_argument(
        "--artifact-encryption",
        action="store_true",
        help="Require encrypted model artifacts for privacy-enabled runs",
    )
    parser.add_argument(
        "--secure-aggregation",
        action="store_true",
        help="Enable the secure aggregation demo path when implemented",
    )
    parser.add_argument("--replay-window-seconds", type=int, default=300)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0)
    parser.add_argument("--clipping-norm", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop training when validation progress stalls or target metrics are reached",
    )
    parser.add_argument("--early-stopping-monitor", default="val_loss")
    parser.add_argument("--early-stopping-mode", default="min", choices=["min", "max"])
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument("--target-macro-f1", type=float)
    parser.add_argument("--target-loss", type=float)
    parser.add_argument(
        "--adaptive-rounds",
        action="store_true",
        help="Extend the training budget when the configured max round is reached before stopping",
    )
    parser.add_argument("--round-extension", type=int, default=20)
    parser.add_argument("--max-rounds-cap", type=int, default=200)


if __name__ == "__main__":
    main()
