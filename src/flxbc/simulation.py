from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from flxbc.chain import maybe_commit_round
from flxbc.config import RunConfig
from flxbc.crypto import (
    demo_client_secret,
    encrypt_artifact_file,
    load_artifact_key_from_env,
    sign_hmac_payload,
)
from flxbc.data import ArrayDataset, load_medical_dataset, partition_indices
from flxbc.ledger import Ledger, stable_hash
from flxbc.privacy import (
    apply_update_privacy,
    create_parameter_envelope,
    deserialize_parameters,
    envelope_signature_fields,
    hash_file,
    serialize_parameters,
    validate_parameter_envelope,
)
from flxbc.secure_aggregation import secure_aggregate_parameters
from flxbc.strategy import (
    aggregate_parameters,
    compute_bc_weights,
    contribution_from_metrics,
    update_reputation,
)
from flxbc.training import LocalTrainResult, select_backend


@dataclass(slots=True)
class SimulationResult:
    run_id: str
    backend: str
    rounds: list[dict[str, Any]]
    ledger: Ledger


def run_federated_demo(config: RunConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    ledger = Ledger(config.db_path)
    ledger.create_run(
        config.run_id,
        mode=config.mode,
        dataset=config.dataset,
        strategy=config.strategy,
        config=asdict(config),
    )

    train, val, test = load_medical_dataset(
        dataset=config.dataset,
        seed=config.seed,
        max_train_samples=config.max_train_samples,
        max_test_samples=config.max_test_samples,
        use_synthetic=config.use_synthetic,
    )
    partitions = partition_indices(
        train.labels,
        num_clients=config.num_clients,
        iid=config.iid,
        alpha=config.alpha,
        seed=config.seed,
    )
    hospital_data = {node_id: train.subset(indices) for node_id, indices in partitions.items()}
    reputations = {node_id: 0.8 for node_id in partitions}
    for node_id in partitions:
        ledger.upsert_node(node_id, display_name=node_id.replace("_", " ").title())

    backend = select_backend(requested_device=config.device)
    global_parameters = backend.initial_parameters(train)
    previous_metrics = backend.evaluate(global_parameters, val)
    run_dir = Path(config.artifact_dir) / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))

    data_label = "synthetic" if config.use_synthetic else config.dataset
    current_round_limit = config.rounds
    max_rounds_cap = max(config.rounds, config.max_rounds_cap)
    round_label = (
        f"Initial/Cap: {config.rounds}/{max_rounds_cap}"
        if config.adaptive_rounds
        else str(config.rounds)
    )
    print(
        f"\n{'=' * 60}\n"
        f"  FLxBC Federated Learning\n"
        f"  Run:         {config.run_id}\n"
        f"  Dataset:     {data_label}  "
        f"({len(train.images)} train / {len(test.images)} test)\n"
        f"  Strategy:    {config.strategy}\n"
        f"  Clients:     {config.num_clients}   |   Rounds: {round_label}\n"
        f"  Backend:     {backend.name}\n"
        f"{'=' * 60}"
    )

    round_metrics: list[dict[str, Any]] = []
    seen_envelope_nonces: set[tuple[str, int, str, str]] = set()
    dp_enabled = config.privacy_mode in {"dp", "full-demo"} or config.dp_noise_multiplier > 0
    early_stopper = _EarlyStopper(config)
    cumulative_communication_bytes = 0
    run_start = perf_counter()
    target_progress = _TargetProgress()
    round_id = 1
    while round_id <= current_round_limit:
        round_start = perf_counter()
        local_results: list[LocalTrainResult] = []
        contribution_records: list[dict[str, float | int | str]] = []
        participants: list[str] = []
        accepted_clients = 0
        rejected_clients = 0
        update_norms_before_clip: list[float] = []
        update_norms_after_clip: list[float] = []
        attempted_upload_bytes = 0
        accepted_upload_bytes = 0
        global_download_bytes_per_client = len(serialize_parameters(global_parameters))
        local_train_duration_seconds = 0.0

        for offset, (node_id, local_train) in enumerate(hospital_data.items()):
            misbehavior = _simulate_node_behavior(config, rng)
            if misbehavior == "dropout":
                new_rep = update_reputation(reputations[node_id], contribution=0.0, misbehaved=True)
                reputations[node_id] = new_rep
                ledger.record_misbehavior(
                    run_id=config.run_id,
                    round_id=round_id,
                    node_id=node_id,
                    kind="dropout",
                    penalty=0.15,
                    detail="client skipped this round in simulated local run",
                )
                continue

            local_train_start = perf_counter()
            result = backend.train_local(
                node_id=node_id,
                global_parameters=global_parameters,
                train=local_train,
                val=val,
                epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                proximal_mu=config.proximal_mu,
                strategy=config.strategy,
                seed=config.seed + round_id * 100 + offset,
            )
            local_train_duration_seconds += perf_counter() - local_train_start
            if misbehavior == "malicious":
                result = _poison_result(result)
            if dp_enabled:
                private_update = apply_update_privacy(
                    result.parameters,
                    global_parameters,
                    clipping_norm=config.clipping_norm,
                    dp_noise_multiplier=config.dp_noise_multiplier,
                    rng=np.random.default_rng(config.seed + round_id * 10_000 + offset),
                )
                update_norms_before_clip.append(private_update.before_clip_norm)
                update_norms_after_clip.append(private_update.after_clip_norm)
                result = LocalTrainResult(
                    node_id=result.node_id,
                    parameters=private_update.parameters,
                    samples=result.samples,
                    metrics=result.metrics,
                    update_norm=private_update.after_clip_norm,
                )

            envelope = create_parameter_envelope(
                result.parameters,
                run_id=config.run_id,
                round_id=round_id,
                node_id=node_id,
                privacy_mode=config.privacy_mode,
                timestamp=float(config.seed + round_id * 1000 + offset),
                nonce=stable_hash(
                    {
                        "run_id": config.run_id,
                        "round_id": round_id,
                        "node_id": node_id,
                        "seed": config.seed,
                    }
                )[:32],
            )
            attempted_upload_bytes += len(envelope.payload_bytes)
            signature_secret = None
            if config.client_auth == "hmac-demo":
                signature_secret = demo_client_secret(config.run_id, node_id)
                envelope = replace(
                    envelope,
                    signature=sign_hmac_payload(
                        signature_secret,
                        envelope_signature_fields(envelope),
                    ),
                )
            valid, invalid_reason = validate_parameter_envelope(
                envelope,
                expected_run_id=config.run_id,
                expected_round_id=round_id,
                expected_node_id=node_id,
                seen_nonces=seen_envelope_nonces,
                now=envelope.timestamp,
                replay_window_seconds=config.replay_window_seconds,
                signature_secret=signature_secret,
            )
            if not valid:
                rejected_clients += 1
                ledger.record_misbehavior(
                    run_id=config.run_id,
                    round_id=round_id,
                    node_id=node_id,
                    kind=invalid_reason or "invalid-envelope",
                    penalty=0.1,
                    detail="client update envelope failed validation",
                )
                continue
            accepted_clients += 1
            accepted_upload_bytes += len(envelope.payload_bytes)
            result = LocalTrainResult(
                node_id=result.node_id,
                parameters=deserialize_parameters(envelope.payload_bytes),
                samples=result.samples,
                metrics=result.metrics,
                update_norm=result.update_norm,
            )
            local_results.append(result)
            participants.append(node_id)

            contribution = contribution_from_metrics(
                previous_accuracy=previous_metrics["accuracy"],
                local_accuracy=result.metrics["accuracy"],
                update_norm=result.update_norm,
                misbehaved=misbehavior is not None,
            )
            reputations[node_id] = update_reputation(
                reputations[node_id],
                contribution=contribution,
                misbehaved=misbehavior is not None,
            )
            if misbehavior is not None:
                ledger.record_misbehavior(
                    run_id=config.run_id,
                    round_id=round_id,
                    node_id=node_id,
                    kind=misbehavior,
                    penalty=0.1,
                    detail=f"simulated {misbehavior} during local training",
                )

            contribution_records.append(
                {
                    "node_id": node_id,
                    "samples": result.samples,
                    "contribution": contribution,
                    "reputation": reputations[node_id],
                }
            )

        if not local_results:
            round_id += 1
            continue

        if config.strategy == "bc-ca-fedprox":
            weight_map = compute_bc_weights(contribution_records)
            weights = [weight_map[result.node_id] for result in local_results]
        else:
            weights = [float(result.samples) for result in local_results]

        secure_aggregation_status = "disabled"
        secure_aggregation_fallback_reason = ""
        aggregation_start = perf_counter()
        if config.secure_aggregation:
            secure_result = secure_aggregate_parameters(
                [result.parameters for result in local_results],
                weights,
                seed=config.seed + round_id,
                expected_clients=config.num_clients,
            )
            global_parameters = secure_result.parameters
            secure_aggregation_status = secure_result.status
            secure_aggregation_fallback_reason = secure_result.fallback_reason or ""
        else:
            global_parameters = aggregate_parameters(
                [result.parameters for result in local_results],
                weights,
            )
        aggregation_duration_seconds = perf_counter() - aggregation_start
        evaluation_start = perf_counter()
        train_metrics = backend.evaluate(global_parameters, train)
        val_metrics = backend.evaluate(global_parameters, val)
        test_metrics = backend.evaluate(global_parameters, test)
        client_metrics = _client_evaluation_metrics(backend, global_parameters, hospital_data)
        evaluation_duration_seconds = perf_counter() - evaluation_start
        metrics = _round_metrics(train=train_metrics, val=val_metrics, test=test_metrics)
        metrics.update(client_metrics)
        download_bytes = global_download_bytes_per_client * len(participants)
        communication_bytes = download_bytes + attempted_upload_bytes
        cumulative_communication_bytes += communication_bytes
        target_reason = _target_reason(config, metrics)
        if target_reason and not target_progress.reached:
            target_progress = _TargetProgress(
                reached=True,
                reason=target_reason,
                round_id=round_id,
                seconds=perf_counter() - run_start,
                communication_bytes=float(cumulative_communication_bytes),
            )
        metrics.update(
            {
                "round": float(round_id),
                "participants": float(len(participants)),
                "backend": backend.name,
                "privacy_mode": config.privacy_mode,
                "accepted_clients": float(accepted_clients),
                "rejected_clients": float(rejected_clients),
                "client_auth": config.client_auth,
                "artifact_encryption": config.artifact_encryption,
                "secure_aggregation": config.secure_aggregation,
                "secure_aggregation_status": secure_aggregation_status,
                "secure_aggregation_fallback_reason": secure_aggregation_fallback_reason,
                "dp_enabled": dp_enabled,
                "clipping_norm": config.clipping_norm,
                "dp_noise_multiplier": config.dp_noise_multiplier,
                "mean_update_norm_before_clip": _mean(update_norms_before_clip),
                "mean_update_norm_after_clip": _mean(update_norms_after_clip),
                "download_bytes": float(download_bytes),
                "upload_bytes": float(accepted_upload_bytes),
                "attempted_upload_bytes": float(attempted_upload_bytes),
                "communication_bytes": float(communication_bytes),
                "cumulative_communication_bytes": float(cumulative_communication_bytes),
                "round_duration_seconds": 0.0,
                "local_train_duration_seconds": float(local_train_duration_seconds),
                "aggregation_duration_seconds": float(aggregation_duration_seconds),
                "evaluation_duration_seconds": float(evaluation_duration_seconds),
                "target_reached": target_progress.reached,
                "target_reason": target_progress.reason,
                "time_to_target_round": float(target_progress.round_id),
                "time_to_target_seconds": float(target_progress.seconds),
                "communication_bytes_at_target": float(target_progress.communication_bytes),
            }
        )
        stop_decision = early_stopper.update(round_id, metrics)
        round_limit_extended = False
        next_round_limit = current_round_limit
        should_stop = stop_decision.should_stop
        stop_reason = stop_decision.reason
        if (
            not should_stop
            and config.adaptive_rounds
            and round_id >= current_round_limit
            and current_round_limit < max_rounds_cap
        ):
            next_round_limit = min(current_round_limit + config.round_extension, max_rounds_cap)
            round_limit_extended = next_round_limit > current_round_limit
        if not should_stop and round_id >= current_round_limit and not round_limit_extended:
            should_stop = True
            stop_reason = "max-rounds-reached"
        metrics.update(
            {
                "early_stopping_enabled": config.early_stopping,
                "early_stopped": stop_decision.should_stop,
                "stop_reason": stop_reason,
                "best_round": float(stop_decision.best_round),
                "best_monitor_value": float(stop_decision.best_value),
                "early_stopping_monitor": config.early_stopping_monitor,
                "adaptive_rounds_enabled": config.adaptive_rounds,
                "round_limit": float(current_round_limit),
                "round_limit_extended": round_limit_extended,
                "max_rounds_cap": float(max_rounds_cap),
            }
        )
        previous_metrics = val_metrics
        round_metrics.append(metrics)

        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("macro_f1", 0)
        bar_len = current_round_limit
        filled = round_id
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"  Round {round_id:>2}/{current_round_limit}  |{bar}|  "
            f"acc={acc:.3f}  f1={f1:.3f}  clients={len(participants)}"
        )
        if round_limit_extended:
            print(
                f"  Extending round budget to {next_round_limit} "
                "because stopping criteria are not met"
            )

        metrics_path = run_dir / f"round_{round_id}_metrics.json"
        params_path = run_dir / f"round_{round_id}_parameters.npz"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        np.savez_compressed(params_path, **global_parameters)
        artifact_path = params_path
        artifact_encryption_mode = "none"
        artifact_ciphertext_hash = ""
        artifact_nonce_hash = ""
        artifact_plaintext_hash = hash_file(params_path)
        if config.artifact_encryption:
            encrypted = encrypt_artifact_file(
                params_path,
                params_path.with_suffix(params_path.suffix + ".enc"),
                key=load_artifact_key_from_env(),
                associated_data=f"{config.run_id}:{round_id}".encode(),
            )
            artifact_path = encrypted.path
            artifact_encryption_mode = encrypted.mode
            artifact_ciphertext_hash = encrypted.ciphertext_hash
            artifact_nonce_hash = encrypted.nonce_hash
            artifact_plaintext_hash = encrypted.plaintext_hash
        artifact_uri = str(artifact_path)
        model_hash = hash_file(artifact_path)
        metrics.update(
            {
                "round_duration_seconds": float(perf_counter() - round_start),
                "artifact_uri": artifact_uri,
                "artifact_encryption_mode": artifact_encryption_mode,
                "artifact_ciphertext_hash": artifact_ciphertext_hash,
                "artifact_nonce_hash": artifact_nonce_hash,
                "artifact_plaintext_hash": artifact_plaintext_hash,
            }
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        metrics_hash = stable_hash(metrics)
        participants_hash = stable_hash(participants)
        strategy_hash = stable_hash({"strategy": config.strategy})
        privacy_hash = stable_hash(
            {
                "privacy_mode": config.privacy_mode,
                "client_auth": config.client_auth,
                "transport_security": config.transport_security,
                "artifact_encryption": config.artifact_encryption,
                "secure_aggregation": config.secure_aggregation,
                "secure_aggregation_status": secure_aggregation_status,
                "secure_aggregation_fallback_reason": secure_aggregation_fallback_reason,
                "dp_enabled": dp_enabled,
                "clipping_norm": config.clipping_norm,
                "dp_noise_multiplier": config.dp_noise_multiplier,
                "accepted_clients": accepted_clients,
                "rejected_clients": rejected_clients,
                "artifact_encryption_mode": artifact_encryption_mode,
                "artifact_ciphertext_hash": artifact_ciphertext_hash,
                "artifact_nonce_hash": artifact_nonce_hash,
                "artifact_plaintext_hash": artifact_plaintext_hash,
            }
        )
        audit_block = ledger.record_audit_block(
            run_id=config.run_id,
            round_id=round_id,
            payload={
                "artifact_uri": artifact_uri,
                "metrics_hash": metrics_hash,
                "model_hash": model_hash,
                "participants_hash": participants_hash,
                "strategy_hash": strategy_hash,
                "privacy_hash": privacy_hash,
                "artifact_encryption_mode": artifact_encryption_mode,
                "artifact_ciphertext_hash": artifact_ciphertext_hash,
                "artifact_nonce_hash": artifact_nonce_hash,
                "artifact_plaintext_hash": artifact_plaintext_hash,
            },
        )
        tx_hash = (
            maybe_commit_round(
                run_id=config.run_id,
                round_id=round_id,
                model_hash=model_hash,
                metrics_hash=metrics_hash,
                participants_hash=participants_hash,
                strategy_hash=strategy_hash,
                artifact_uri=artifact_uri,
            )
            or audit_block["tx_hash"]
        )
        ledger.record_round(
            run_id=config.run_id,
            round_id=round_id,
            metrics=metrics,
            artifact_uri=artifact_uri,
            participants=participants,
            tx_hash=tx_hash,
            model_hash=model_hash,
            metrics_hash=metrics_hash,
            participants_hash=participants_hash,
        )
        for record in contribution_records:
            points = float(record["contribution"]) * float(record["reputation"]) * 10.0
            ledger.record_contribution(
                run_id=config.run_id,
                round_id=round_id,
                node_id=str(record["node_id"]),
                samples=int(record["samples"]),
                contribution=float(record["contribution"]),
                reputation=float(record["reputation"]),
                points=points,
            )
        if round_limit_extended:
            current_round_limit = next_round_limit
        if should_stop:
            break
        round_id += 1

    ledger.finish_run(config.run_id)
    _write_summary(run_dir, config, backend.name, round_metrics)

    if round_metrics:
        final = round_metrics[-1]
        best_round = int(final.get("best_round", final.get("round", len(round_metrics))))
        print(
            f"{'=' * 60}\n"
            f"  Done.  accuracy={final.get('accuracy', 0):.3f}  "
            f"macro_f1={final.get('macro_f1', 0):.3f}  "
            f"rounds={len(round_metrics)}  best_round={best_round}\n"
            f"{'=' * 60}\n"
        )

    return SimulationResult(
        run_id=config.run_id,
        backend=backend.name,
        rounds=round_metrics,
        ledger=ledger,
    )


def run_centralized_baseline(config: RunConfig) -> dict[str, float]:
    train, val, test = load_medical_dataset(
        dataset=config.dataset,
        seed=config.seed,
        max_train_samples=config.max_train_samples,
        max_test_samples=config.max_test_samples,
        use_synthetic=config.use_synthetic,
    )
    backend = select_backend(requested_device=config.device)
    parameters = backend.initial_parameters(train)
    result = backend.train_local(
        node_id="centralized",
        global_parameters=parameters,
        train=train,
        val=val,
        epochs=max(1, config.rounds // 2),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        proximal_mu=0.0,
        strategy="centralized",
        seed=config.seed,
    )
    metrics = backend.evaluate(result.parameters, test)
    ledger = Ledger(config.db_path)
    ledger.record_experiment_result(run_id=config.run_id, label="centralized", metrics=metrics)
    return metrics


def run_comparison_suite(config: RunConfig) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    ledger = Ledger(config.db_path)
    ledger.create_run(
        config.run_id,
        mode=config.mode,
        dataset=config.dataset,
        strategy="comparison",
        config=asdict(config),
    )

    print("\n[1/4] Centralized baseline ...")
    centralized = run_centralized_baseline(config)
    results["centralized"] = centralized
    print(
        f"        → acc={centralized.get('accuracy', 0):.3f}  "
        f"f1={centralized.get('macro_f1', 0):.3f}"
    )

    for i, strategy in enumerate(("fedavg", "fedprox", "bc-ca-fedprox"), start=2):
        print(f"\n[{i}/4] {strategy} ...")
        strategy_config = RunConfig(
            **{**asdict(config), "strategy": strategy, "run_id": f"{config.run_id}-{strategy}"}
        )
        sim = run_federated_demo(strategy_config)
        results[strategy] = sim.rounds[-1] if sim.rounds else {}
        final = results[strategy]
        print(f"        → acc={final.get('accuracy', 0):.3f}  f1={final.get('macro_f1', 0):.3f}")
        ledger.record_experiment_result(
            run_id=config.run_id,
            label=strategy,
            metrics=results[strategy],
        )
    ledger.finish_run(config.run_id)
    return results


def _simulate_node_behavior(config: RunConfig, rng: np.random.Generator) -> str | None:
    if not config.simulate_failures:
        return None
    draw = float(rng.random())
    if draw < config.dropout_rate:
        return "dropout"
    if draw < config.dropout_rate + config.timeout_rate:
        return "timeout"
    if draw < config.dropout_rate + config.timeout_rate + config.malicious_rate:
        return "malicious"
    return None


def _poison_result(result: LocalTrainResult) -> LocalTrainResult:
    poisoned = {name: value * -1.0 for name, value in result.parameters.items()}
    return LocalTrainResult(
        node_id=result.node_id,
        parameters=poisoned,
        samples=result.samples,
        metrics=result.metrics,
        update_norm=result.update_norm * 2.0,
    )


def _write_summary(
    run_dir: Path,
    config: RunConfig,
    backend_name: str,
    round_metrics: list[dict[str, Any]],
) -> None:
    best_metrics = _best_round_metrics(round_metrics)
    summary = {
        "run_id": config.run_id,
        "mode": config.mode,
        "dataset": config.dataset,
        "strategy": config.strategy,
        "backend": backend_name,
        "rounds": len(round_metrics),
        "final_metrics": round_metrics[-1] if round_metrics else {},
        "best_round": int(best_metrics.get("round", 0)) if best_metrics else 0,
        "best_metrics": best_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _best_round_metrics(round_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not round_metrics:
        return {}
    final = round_metrics[-1]
    best_round = int(final.get("best_round", final.get("round", len(round_metrics))))
    for metrics in round_metrics:
        if int(metrics.get("round", 0)) == best_round:
            return metrics
    return final


@dataclass(slots=True)
class _TargetProgress:
    reached: bool = False
    reason: str = ""
    round_id: int = 0
    seconds: float = 0.0
    communication_bytes: float = 0.0


@dataclass(slots=True)
class _StopDecision:
    should_stop: bool
    reason: str
    best_round: int
    best_value: float


class _EarlyStopper:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.best_round = 0
        self.best_value = float("inf") if config.early_stopping_mode == "min" else -float("inf")
        self.bad_rounds = 0

    def update(self, round_id: int, metrics: dict[str, Any]) -> _StopDecision:
        monitor = self.config.early_stopping_monitor
        if monitor not in metrics:
            if self.config.early_stopping:
                numeric_keys = ", ".join(
                    sorted(
                        key
                        for key, value in metrics.items()
                        if isinstance(value, int | float)
                    )
                )
                raise ValueError(
                    f"Early stopping monitor '{monitor}' is not available. "
                    f"Available numeric metrics: {numeric_keys}"
                )
            current = self.best_value
        else:
            current = float(metrics[monitor])
        if self._improved(current):
            self.best_value = current
            self.best_round = round_id
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1

        best_round = self.best_round or round_id
        if not self.config.early_stopping or round_id < self.config.min_rounds:
            return _StopDecision(False, "", best_round, self.best_value)
        target_reason = _target_reason(self.config, metrics)
        if target_reason:
            return _StopDecision(True, target_reason, best_round, self.best_value)
        if self.bad_rounds > self.config.early_stopping_patience:
            return _StopDecision(True, "patience-exhausted", best_round, self.best_value)
        return _StopDecision(False, "", best_round, self.best_value)

    def _improved(self, current: float) -> bool:
        if self.config.early_stopping_mode == "min":
            return current < self.best_value - self.config.early_stopping_min_delta
        return current > self.best_value + self.config.early_stopping_min_delta


def _target_reason(config: RunConfig, metrics: dict[str, Any]) -> str:
    if config.target_accuracy is not None and metrics["val_accuracy"] >= config.target_accuracy:
        return "target-accuracy"
    if config.target_macro_f1 is not None and metrics["val_macro_f1"] >= config.target_macro_f1:
        return "target-macro-f1"
    if config.target_loss is not None and metrics["val_loss"] <= config.target_loss:
        return "target-loss"
    return ""


def _client_evaluation_metrics(
    backend: Any,
    parameters: dict[str, np.ndarray],
    hospital_data: dict[str, ArrayDataset],
) -> dict[str, float]:
    evaluations = [
        backend.evaluate(parameters, dataset)
        for dataset in hospital_data.values()
        if len(dataset.labels)
    ]
    if not evaluations:
        return {}

    metrics: dict[str, float] = {}
    for name in ("loss", "accuracy", "macro_f1", "auc"):
        values = np.asarray([evaluation[name] for evaluation in evaluations], dtype=np.float64)
        metrics[f"client_{name}_mean"] = float(np.mean(values))
        metrics[f"client_{name}_std"] = float(np.std(values))
        metrics[f"client_{name}_min"] = float(np.min(values))
        metrics[f"client_{name}_max"] = float(np.max(values))
    return metrics


def _round_metrics(
    *,
    train: dict[str, float],
    val: dict[str, float],
    test: dict[str, float],
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "loss": test["loss"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "auc": test["auc"],
    }
    for prefix, values in (("train", train), ("val", val), ("test", test)):
        for name, value in values.items():
            metrics[f"{prefix}_{name}"] = value
    metrics["generalization_loss_gap"] = val["loss"] - train["loss"]
    metrics["generalization_accuracy_gap"] = train["accuracy"] - val["accuracy"]
    return metrics
