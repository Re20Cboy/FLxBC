# FLxBC 训练指标与停止机制推进计划 5-13

## 背景判断

当前 dashboard 训练过程页展示的是 `accuracy` 和 `macro_f1`，不是 loss 曲线。截图中的曲线对应测试集准确率与 Macro-F1。后续应把关键训练指标拆开展示，并加入停止训练机制，避免纯手动设置轮次导致训练不足、过拟合或资源浪费。

## 参考实践

- FedAvg 论文重点展示 test accuracy vs communication rounds，并比较达到目标准确率所需通信轮数。
- FedProx 论文和补充材料关注 training loss、testing accuracy、系统/统计异质性、straggler/dropout 影响。
- Flower 等工程框架将 federated train metrics、evaluate metrics、global evaluation metrics 作为每轮结果记录。
- 通用深度学习 early stopping 使用 hold-out validation set 监控 loss 或 accuracy，并通过 patience/min_delta 判断停止，以缓解过拟合。

## 本项目第一版设计

### 指标体系

每轮记录：

- Test: `test_loss`, `test_accuracy`, `test_macro_f1`, `test_auc`
- Validation: `val_loss`, `val_accuracy`, `val_macro_f1`, `val_auc`
- Train diagnostic: `train_loss`, `train_accuracy`, `train_macro_f1`, `train_auc`
- Backward-compatible aliases: `loss`, `accuracy`, `macro_f1`, `auc` 继续指向 test 指标
- FL system: participants, accepted/rejected clients, update norm before/after clip
- Communication: `download_bytes`, `upload_bytes`, `communication_bytes`, `cumulative_communication_bytes`
- Timing: `round_duration_seconds`, `local_train_duration_seconds`, `aggregation_duration_seconds`, `evaluation_duration_seconds`
- Target progress: `time_to_target_round`, `time_to_target_seconds`, `communication_bytes_at_target`
- Per-client: `client_loss_*`, `client_accuracy_*`, `client_macro_f1_*`, `client_auc_*`
- Privacy/security: privacy mode, DP, artifact encryption, secure aggregation status

### Dashboard 展示

训练过程页拆成：

- 质量指标曲线：test/val accuracy、macro-F1；test accuracy 和 test Macro-F1 加粗突出
- AUC 保留在 metrics/table 中作为二分类排序能力诊断，但不进入主质量曲线
- Loss 曲线：train/val/test loss
- 泛化差距：`val_loss - train_loss`, `train_accuracy - val_accuracy`
- 训练控制状态：early stopping status、best round、stop reason
- 系统效率曲线：participants, accepted/rejected clients, update norm
- 通信成本曲线：download/upload bytes、round/cumulative communication bytes
- 耗时曲线：round/local-train/aggregation/evaluation seconds
- 客户端差异曲线：per-client accuracy/loss mean/std

### 停止机制

第一版采用验证集驱动的 early stopping：

- 默认不改变旧行为，需显式 `--early-stopping` 启用。
- 默认 monitor 为 `val_loss`，mode 为 `min`。
- 支持 `--early-stopping-patience`、`--early-stopping-min-delta`、`--min-rounds`。
- 支持目标达标停止：`--target-accuracy`、`--target-macro-f1`、`--target-loss`。
- 达标或 patience 用尽后停止后续轮次，并在 summary/metrics 中记录 stop reason。

## 当前已推进

- `training.py` 已补齐 loss、accuracy、Macro-F1、AUC 计算。
- `simulation.py` 已每轮记录 train/validation/test 指标，并继续保留旧字段别名。
- `simulation.py` 已支持验证集驱动 early stopping，记录 `best_round`、`best_monitor_value`、`stop_reason`。
- `summary.json` 已同时写入 `final_metrics` 与 `best_metrics`，防止早停后误用非最佳轮次。
- `dashboard/app.py` 已拆分展示质量指标、loss、泛化差距、系统/隐私指标，并突出主指标线。
- 每轮已记录 download/upload 参数字节数、round communication bytes、累计通信量。
- 每轮已记录 round/local-train/aggregation/evaluation 耗时。
- 已记录 target-reaching round/seconds/communication bytes。
- 已记录 per-client loss/accuracy/Macro-F1/AUC 的 mean/std/min/max。
- Makefile 常用入口已切换为“最大轮次 + early stopping”。
- `make run-real-2d` 已改为真实数据更保守的默认策略：不默认设置单一 `target_accuracy`，至少 10 轮 warm-up，使用 `val_loss` patience 停止，并增大默认真实样本量。
- 已新增 adaptive rounds：到达初始轮次预算时若还未满足停止条件，可扩展训练预算直到 cap。
- 已新增 `make run-real-2d` 和 `make run-real-3d`，分别覆盖 PneumoniaMNIST 与 NoduleMNIST3D。
- `README.md` 与 `PROJECT_STATUS.md` 已更新指标与早停使用方式。

## 下一步推进计划

1. 补 dashboard/report export：自动导出 final vs best、目标达标轮次、隐私配置、链上审计摘要。
2. 补更细通信拆分：metadata/signature bytes 与真实网络传输开销估计。
3. 补 per-client 可视化详情：按节点展示局部评估与贡献积分。
4. 补实验矩阵聚合表：不同策略/隐私模式/早停配置的横向对比。

### 后续可扩展

- Time-to-target: 达到目标精度所需轮次、耗时和通信量。
- Communication cost: 进一步区分模型下发、client update、metadata/signature bytes。
- Fairness: 节点贡献分布、per-client accuracy/loss 方差。
- Heterogeneity: client drift/update norm variance。
- Robustness: malicious/dropout 情况下的收敛曲线。

## 参考来源

- FedAvg: https://arxiv.org/abs/1602.05629
- FedProx: https://arxiv.org/abs/1812.06127
- Flower federated evaluation: https://flower.ai/docs/framework/explanation-federated-evaluation.html
- TensorFlow/Keras EarlyStopping: https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/EarlyStopping
