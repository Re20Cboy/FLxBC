from pathlib import Path

from flxbc.cli import _build_parser, _config_from_args
from flxbc.config import RunConfig


def test_run_config_privacy_defaults_are_disabled():
    config = RunConfig(run_id="privacy-defaults")

    assert config.privacy_mode == "none"
    assert config.client_auth == "none"
    assert config.artifact_encryption is False
    assert config.secure_aggregation is False
    assert config.replay_window_seconds == 300
    assert config.dp_noise_multiplier == 0.0
    assert config.clipping_norm == 1.0
    assert config.early_stopping is False
    assert config.early_stopping_monitor == "val_loss"
    assert config.early_stopping_patience == 5
    assert config.min_rounds == 1
    assert config.target_accuracy is None
    assert config.target_macro_f1 is None
    assert config.target_loss is None
    assert config.adaptive_rounds is False
    assert config.round_extension == 20
    assert config.max_rounds_cap == 200


def test_cli_run_config_includes_privacy_options():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--run-id",
            "privacy-demo",
            "--synthetic",
            "--rounds",
            "2",
            "--clients",
            "3",
            "--privacy-mode",
            "full-demo",
            "--client-auth",
            "hmac-demo",
            "--dp-noise-multiplier",
            "0.25",
            "--clipping-norm",
            "0.75",
            "--artifact-encryption",
            "--secure-aggregation",
            "--replay-window-seconds",
            "120",
            "--early-stopping",
            "--early-stopping-monitor",
            "val_macro_f1",
            "--early-stopping-mode",
            "max",
            "--early-stopping-patience",
            "2",
            "--early-stopping-min-delta",
            "0.01",
            "--min-rounds",
            "3",
            "--target-accuracy",
            "0.8",
            "--target-macro-f1",
            "0.7",
            "--target-loss",
            "0.4",
            "--adaptive-rounds",
            "--round-extension",
            "7",
            "--max-rounds-cap",
            "25",
            "--db",
            "data/custom.db",
        ]
    )

    config = _config_from_args(
        args,
        run_id="privacy-demo",
        mode="demo",
        db_path=Path(args.db),
    )

    assert config.privacy_mode == "full-demo"
    assert config.client_auth == "hmac-demo"
    assert config.dp_noise_multiplier == 0.25
    assert config.clipping_norm == 0.75
    assert config.artifact_encryption is True
    assert config.secure_aggregation is True
    assert config.replay_window_seconds == 120
    assert config.early_stopping is True
    assert config.early_stopping_monitor == "val_macro_f1"
    assert config.early_stopping_mode == "max"
    assert config.early_stopping_patience == 2
    assert config.early_stopping_min_delta == 0.01
    assert config.min_rounds == 3
    assert config.target_accuracy == 0.8
    assert config.target_macro_f1 == 0.7
    assert config.target_loss == 0.4
    assert config.adaptive_rounds is True
    assert config.round_extension == 7
    assert config.max_rounds_cap == 25
    assert config.db_path == Path("data/custom.db")
