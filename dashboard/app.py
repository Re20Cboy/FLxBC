from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from flxbc.ledger import Ledger
from flxbc.presentation import audit_status_label, related_experiment_run_ids, short_hash

st.set_page_config(page_title="FLxBC Dashboard", layout="wide")

LINE_STYLES: dict[str, dict[str, Any]] = {
    "test_accuracy": {"color": "#1f5fbf", "width": 4, "dash": "solid", "opacity": 1.0},
    "test_macro_f1": {"color": "#d33f32", "width": 4, "dash": "solid", "opacity": 1.0},
    "val_accuracy": {"color": "#8bbff2", "width": 2, "dash": "dash", "opacity": 0.75},
    "val_macro_f1": {"color": "#f3a6a1", "width": 2, "dash": "dash", "opacity": 0.75},
    "val_loss": {"color": "#1f5fbf", "width": 4, "dash": "solid", "opacity": 1.0},
    "test_loss": {"color": "#d33f32", "width": 3, "dash": "solid", "opacity": 0.9},
    "train_loss": {"color": "#8bbff2", "width": 2, "dash": "dot", "opacity": 0.7},
}

DEFAULT_LINE_STYLE = {"color": "#6f7d95", "width": 2, "dash": "solid", "opacity": 0.8}


def metric_value(
    metrics: dict[str, Any],
    name: str,
    *,
    fallback: str | None = None,
    default: Any = None,
) -> Any:
    if name in metrics:
        return metrics[name]
    if fallback is not None and fallback in metrics:
        return metrics[fallback]
    return default


def plot_metric_lines(
    df: pd.DataFrame,
    *,
    title: str,
    columns: list[str],
    empty_message: str,
) -> None:
    available = [column for column in columns if column in df and df[column].notna().any()]
    if not available:
        st.info(empty_message)
        return
    st.subheader(title)
    figure = go.Figure()
    for column in available:
        style = LINE_STYLES.get(column, DEFAULT_LINE_STYLE)
        figure.add_trace(
            go.Scatter(
                x=df["round"],
                y=df[column],
                mode="lines+markers",
                name=column,
                opacity=style["opacity"],
                line={
                    "color": style["color"],
                    "width": style["width"],
                    "dash": style["dash"],
                },
                marker={"size": 7 if style["width"] >= 4 else 5},
            )
        )
    figure.update_layout(
        xaxis_title="round",
        yaxis_title="value",
        legend_title_text="variable",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(figure, width="stretch")


@st.cache_data(ttl=3)
def load_data(db_path: str) -> dict[str, list[dict]]:
    ledger = Ledger(Path(db_path))
    return {
        "runs": ledger.list_runs(),
        "nodes": ledger.list_nodes(),
        "rounds": ledger.get_rounds(),
        "contributions": ledger.get_contributions(),
        "misbehavior": ledger.get_misbehavior(),
        "experiments": ledger.get_experiments(),
        "audit_blocks": ledger.get_audit_blocks(),
    }


def main() -> None:
    db_path = st.sidebar.text_input("SQLite DB", os.getenv("FLXBC_DB", "data/flxbc.db"))
    data = load_data(db_path)
    runs = data["runs"]

    st.title("FLxBC 医疗联邦学习审计面板")
    if not runs:
        st.info("暂无训练记录。先运行 `make demo` 或 `uv run flxbc demo --synthetic`。")
        return

    run_ids = [run["run_id"] for run in runs]
    query_run_id = st.query_params.get("run_id")
    default_index = run_ids.index(query_run_id) if query_run_id in run_ids else 0
    selected_run = st.sidebar.selectbox("Run", run_ids, index=default_index)
    selected_meta = next(run for run in runs if run["run_id"] == selected_run)
    rounds = [row for row in data["rounds"] if row["run_id"] == selected_run]
    contributions = [row for row in data["contributions"] if row["run_id"] == selected_run]
    misbehavior = [row for row in data["misbehavior"] if row["run_id"] == selected_run]
    experiments = [row for row in data["experiments"] if row["run_id"] == selected_run]
    audit_blocks = [row for row in data["audit_blocks"] if row["run_id"] == selected_run]
    audit_status = Ledger(Path(db_path)).verify_audit_chain(selected_run)
    related_runs = related_experiment_run_ids(selected_run, runs)

    final_metrics = rounds[-1]["metrics"] if rounds else {}
    metric_cols = st.columns(8)
    metric_cols[0].metric("状态", selected_meta["status"])
    metric_cols[1].metric("策略", selected_meta["strategy"])
    metric_cols[2].metric("轮次", len(rounds))
    metric_cols[3].metric("Accuracy", f"{final_metrics.get('accuracy', 0):.3f}")
    metric_cols[4].metric("Macro-F1", f"{final_metrics.get('macro_f1', 0):.3f}")
    metric_cols[5].metric("审计链", audit_status_label(audit_status))
    metric_cols[6].metric("隐私模式", str(final_metrics.get("privacy_mode", "none")))
    metric_cols[7].metric("DP", "on" if final_metrics.get("dp_enabled") else "off")
    st.caption(
        f"Run: {selected_run} | Dataset: {selected_meta['dataset']} | "
        f"Mode: {selected_meta['mode']} | Started: {selected_meta['started_at']}"
    )

    overview, nodes_tab, audit_tab, compare_tab = st.tabs(
        ["训练过程", "节点贡献", "链上审计", "实验对比"]
    )

    with overview:
        if rounds:
            round_df = pd.DataFrame(
                {
                    "round": row["round_id"],
                    "accuracy": metric_value(row["metrics"], "accuracy"),
                    "macro_f1": metric_value(row["metrics"], "macro_f1"),
                    "auc": metric_value(row["metrics"], "auc"),
                    "loss": metric_value(row["metrics"], "loss"),
                    "train_accuracy": metric_value(row["metrics"], "train_accuracy"),
                    "val_accuracy": metric_value(row["metrics"], "val_accuracy"),
                    "test_accuracy": metric_value(
                        row["metrics"], "test_accuracy", fallback="accuracy"
                    ),
                    "train_macro_f1": metric_value(row["metrics"], "train_macro_f1"),
                    "val_macro_f1": metric_value(row["metrics"], "val_macro_f1"),
                    "test_macro_f1": metric_value(
                        row["metrics"], "test_macro_f1", fallback="macro_f1"
                    ),
                    "train_auc": metric_value(row["metrics"], "train_auc"),
                    "val_auc": metric_value(row["metrics"], "val_auc"),
                    "test_auc": metric_value(
                        row["metrics"], "test_auc", fallback="auc"
                    ),
                    "train_loss": metric_value(row["metrics"], "train_loss"),
                    "val_loss": metric_value(row["metrics"], "val_loss"),
                    "test_loss": metric_value(row["metrics"], "test_loss", fallback="loss"),
                    "generalization_loss_gap": row["metrics"].get(
                        "generalization_loss_gap"
                    ),
                    "generalization_accuracy_gap": row["metrics"].get(
                        "generalization_accuracy_gap"
                    ),
                    "participants": row["metrics"].get("participants", 0),
                    "accepted_clients": row["metrics"].get("accepted_clients", 0),
                    "rejected_clients": row["metrics"].get("rejected_clients", 0),
                    "mean_update_norm_before_clip": row["metrics"].get(
                        "mean_update_norm_before_clip", 0
                    ),
                    "mean_update_norm_after_clip": row["metrics"].get(
                        "mean_update_norm_after_clip", 0
                    ),
                    "download_bytes": row["metrics"].get("download_bytes"),
                    "upload_bytes": row["metrics"].get("upload_bytes"),
                    "attempted_upload_bytes": row["metrics"].get("attempted_upload_bytes"),
                    "communication_bytes": row["metrics"].get("communication_bytes"),
                    "cumulative_communication_bytes": row["metrics"].get(
                        "cumulative_communication_bytes"
                    ),
                    "round_duration_seconds": row["metrics"].get("round_duration_seconds"),
                    "local_train_duration_seconds": row["metrics"].get(
                        "local_train_duration_seconds"
                    ),
                    "aggregation_duration_seconds": row["metrics"].get(
                        "aggregation_duration_seconds"
                    ),
                    "evaluation_duration_seconds": row["metrics"].get(
                        "evaluation_duration_seconds"
                    ),
                    "client_accuracy_mean": row["metrics"].get("client_accuracy_mean"),
                    "client_accuracy_std": row["metrics"].get("client_accuracy_std"),
                    "client_loss_mean": row["metrics"].get("client_loss_mean"),
                    "client_loss_std": row["metrics"].get("client_loss_std"),
                    "target_reached": row["metrics"].get("target_reached", False),
                    "target_reason": row["metrics"].get("target_reason", ""),
                    "time_to_target_round": row["metrics"].get("time_to_target_round", 0),
                    "time_to_target_seconds": row["metrics"].get("time_to_target_seconds", 0),
                    "communication_bytes_at_target": row["metrics"].get(
                        "communication_bytes_at_target", 0
                    ),
                    "early_stopped": row["metrics"].get("early_stopped", False),
                    "stop_reason": row["metrics"].get("stop_reason", ""),
                    "best_round": row["metrics"].get("best_round", 0),
                    "is_best_round": int(row["round_id"])
                    == int(final_metrics.get("best_round", 0)),
                    "artifact_uri": row["artifact_uri"],
                    "model_hash": short_hash(row["model_hash"]),
                    "tx_hash": short_hash(row["tx_hash"]),
                }
                for row in rounds
            )
            stop_reason = final_metrics.get("stop_reason")
            if stop_reason:
                st.info(
                    f"Early stopping: {stop_reason}, "
                    f"best round={int(final_metrics.get('best_round', 0))}"
                )
            best_round = int(final_metrics.get("best_round", 0))
            best_rows = round_df[round_df["round"] == best_round]
            if best_round and not best_rows.empty:
                best_row = best_rows.iloc[0]
                st.caption(
                    f"Best model: round={best_round} | "
                    f"model_hash={best_row['model_hash']} | artifact={best_row['artifact_uri']}"
                )
            plot_metric_lines(
                round_df,
                title="质量指标",
                columns=[
                    "test_accuracy",
                    "val_accuracy",
                    "test_macro_f1",
                    "val_macro_f1",
                ],
                empty_message="该 run 尚未记录 accuracy / macro-F1 指标。",
            )
            plot_metric_lines(
                round_df,
                title="Loss",
                columns=["train_loss", "val_loss", "test_loss"],
                empty_message="该 run 尚未记录 loss；请重新运行包含新指标的训练。",
            )
            plot_metric_lines(
                round_df,
                title="泛化差距",
                columns=["generalization_loss_gap", "generalization_accuracy_gap"],
                empty_message="该 run 尚未记录 train/val 泛化差距。",
            )
            plot_metric_lines(
                round_df,
                title="系统与隐私指标",
                columns=[
                    "participants",
                    "accepted_clients",
                    "rejected_clients",
                    "mean_update_norm_before_clip",
                    "mean_update_norm_after_clip",
                ],
                empty_message="该 run 尚未记录系统或隐私指标。",
            )
            plot_metric_lines(
                round_df,
                title="通信成本",
                columns=[
                    "download_bytes",
                    "upload_bytes",
                    "communication_bytes",
                    "cumulative_communication_bytes",
                ],
                empty_message="该 run 尚未记录参数传输字节数。",
            )
            plot_metric_lines(
                round_df,
                title="耗时",
                columns=[
                    "round_duration_seconds",
                    "local_train_duration_seconds",
                    "aggregation_duration_seconds",
                    "evaluation_duration_seconds",
                ],
                empty_message="该 run 尚未记录每轮耗时。",
            )
            plot_metric_lines(
                round_df,
                title="客户端差异",
                columns=[
                    "client_accuracy_mean",
                    "client_accuracy_std",
                    "client_loss_mean",
                    "client_loss_std",
                ],
                empty_message="该 run 尚未记录 per-client 评估指标。",
            )
            st.dataframe(round_df, width="stretch", hide_index=True)
        else:
            st.warning("该 run 尚未记录训练轮次。")

    with nodes_tab:
        node_df = pd.DataFrame(data["nodes"])
        if not node_df.empty:
            st.dataframe(node_df, width="stretch", hide_index=True)
        if contributions:
            contrib_df = pd.DataFrame(contributions)
            st.plotly_chart(
                px.bar(
                    contrib_df,
                    x="node_id",
                    y="points",
                    color="round_id",
                    title="每轮积分结算",
                ),
                width="stretch",
            )
            st.dataframe(contrib_df, width="stretch", hide_index=True)

    with audit_tab:
        if audit_status["valid"]:
            st.success(
                f"Mock audit chain verified: {audit_status['blocks']} blocks, "
                f"head={short_hash(audit_status['head_block_hash'])}"
            )
        else:
            st.error(f"Mock audit chain failed at block {audit_status['broken_at']}")
        if rounds:
            audit_df = pd.DataFrame(
                {
                    "round": row["round_id"],
                    "model_hash": short_hash(row["model_hash"]),
                    "metrics_hash": short_hash(row["metrics_hash"]),
                    "participants_hash": short_hash(row["participants_hash"]),
                    "tx_hash": short_hash(row["tx_hash"]),
                    "artifact_uri": row["artifact_uri"],
                }
                for row in rounds
            )
            st.dataframe(audit_df, width="stretch", hide_index=True)
        if audit_blocks:
            block_df = pd.DataFrame(
                {
                    "height": row["block_height"],
                    "round": row["round_id"],
                    "tx_hash": short_hash(row["tx_hash"]),
                    "block_hash": short_hash(row["block_hash"]),
                    "previous_block_hash": short_hash(row["previous_block_hash"]),
                }
                for row in audit_blocks
            )
            st.subheader("Mock 区块")
            st.dataframe(block_df, width="stretch", hide_index=True)
        if final_metrics:
            st.subheader("隐私状态")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "privacy_mode": final_metrics.get("privacy_mode", "none"),
                            "client_auth": final_metrics.get("client_auth", "none"),
                            "artifact_encryption": final_metrics.get(
                                "artifact_encryption", False
                            ),
                            "artifact_encryption_mode": final_metrics.get(
                                "artifact_encryption_mode", "none"
                            ),
                            "secure_aggregation": final_metrics.get("secure_aggregation", False),
                            "secure_aggregation_status": final_metrics.get(
                                "secure_aggregation_status", "disabled"
                            ),
                            "secure_aggregation_fallback_reason": final_metrics.get(
                                "secure_aggregation_fallback_reason", ""
                            ),
                            "dp_enabled": final_metrics.get("dp_enabled", False),
                            "clipping_norm": final_metrics.get("clipping_norm", 0.0),
                            "dp_noise_multiplier": final_metrics.get(
                                "dp_noise_multiplier", 0.0
                            ),
                            "mean_update_norm_before_clip": final_metrics.get(
                                "mean_update_norm_before_clip", 0.0
                            ),
                            "mean_update_norm_after_clip": final_metrics.get(
                                "mean_update_norm_after_clip", 0.0
                            ),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if misbehavior:
            st.subheader("失信事件")
            st.dataframe(pd.DataFrame(misbehavior), width="stretch", hide_index=True)
        else:
            st.info("该 run 暂无失信事件。")

    with compare_tab:
        if experiments:
            exp_df = pd.DataFrame(
                {
                    "label": row["label"],
                    "accuracy": row["metrics"].get("accuracy", 0),
                    "macro_f1": row["metrics"].get("macro_f1", 0),
                    "auc": row["metrics"].get("auc", 0),
                }
                for row in experiments
            )
            st.plotly_chart(
                px.bar(exp_df, x="label", y=["accuracy", "macro_f1"], barmode="group"),
                width="stretch",
            )
            st.dataframe(exp_df, width="stretch", hide_index=True)
        if related_runs:
            child_rounds = [
                row for row in data["rounds"] if row["run_id"] in related_runs
            ]
            if child_rounds:
                child_df = pd.DataFrame(
                    {
                        "run_id": row["run_id"],
                        "round": row["round_id"],
                        "accuracy": row["metrics"].get("accuracy", 0),
                        "macro_f1": row["metrics"].get("macro_f1", 0),
                        "backend": row["metrics"].get("backend", "-"),
                        "tx_hash": short_hash(row["tx_hash"]),
                    }
                    for row in child_rounds
                )
                st.subheader("子实验轮次")
                st.dataframe(child_df, width="stretch", hide_index=True)
        else:
            st.info("运行 `make run` / `make run-real` 后这里会显示中心化与联邦训练对比。")


if __name__ == "__main__":
    main()
