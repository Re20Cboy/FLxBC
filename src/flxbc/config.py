from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

StrategyName = Literal["fedavg", "fedprox", "bc-ca-fedprox", "centralized", "local-only"]
PrivacyMode = Literal["none", "dp", "encrypted", "secure-aggregation", "full-demo"]
ClientAuthMode = Literal["none", "hmac-demo"]
TransportSecurityMode = Literal["local-only", "tls", "mtls"]
EarlyStoppingMode = Literal["min", "max"]


@dataclass(slots=True)
class RunConfig:
    run_id: str
    mode: str = "demo"
    dataset: str = "nodulemnist3d"
    strategy: StrategyName = "bc-ca-fedprox"
    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 42
    alpha: float = 0.5
    iid: bool = False
    max_train_samples: int | None = 1200
    max_test_samples: int | None = 400
    use_synthetic: bool = False
    device: str = "auto"
    max_parallel_clients: int = 1
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts"))
    db_path: Path = field(default_factory=lambda: Path("data/flxbc.db"))
    simulate_failures: bool = False
    timeout_rate: float = 0.08
    dropout_rate: float = 0.05
    malicious_rate: float = 0.02
    proximal_mu: float = 0.01
    privacy_mode: PrivacyMode = "none"
    client_auth: ClientAuthMode = "none"
    transport_security: TransportSecurityMode = "local-only"
    artifact_encryption: bool = False
    secure_aggregation: bool = False
    replay_window_seconds: int = 300
    dp_noise_multiplier: float = 0.0
    clipping_norm: float = 1.0
    early_stopping: bool = False
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: EarlyStoppingMode = "min"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    min_rounds: int = 1
    target_accuracy: float | None = None
    target_macro_f1: float | None = None
    target_loss: float | None = None
    adaptive_rounds: bool = False
    round_extension: int = 20
    max_rounds_cap: int = 200

    @classmethod
    def demo(cls, *, run_id: str, use_synthetic: bool = False) -> RunConfig:
        return cls(
            run_id=run_id,
            mode="demo",
            num_clients=5,
            rounds=10,
            max_train_samples=800,
            max_test_samples=240,
            use_synthetic=use_synthetic,
            simulate_failures=True,
        )

    @classmethod
    def experiment(cls, *, run_id: str, use_synthetic: bool = False) -> RunConfig:
        return cls(
            run_id=run_id,
            mode="experiment",
            num_clients=8,
            rounds=30,
            max_train_samples=1600,
            max_test_samples=400,
            use_synthetic=use_synthetic,
            simulate_failures=True,
        )
