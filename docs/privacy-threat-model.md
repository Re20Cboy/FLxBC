# FLxBC 隐私威胁模型与边界

本文档用于指导 FLxBC 后续隐私增强开发。当前项目定位仍是本地研究/demo 系统，不是生产级临床联邦学习平台，也不处理真实患者数据。

## 1. 保护对象

后续隐私功能应围绕以下资产设计：

- 原始医疗影像数据：每个模拟医院的训练样本必须停留在本地客户端边界内。
- 客户端模型参数或梯度更新：上传给聚合端的更新可能泄露局部数据分布，应作为敏感数据处理。
- 医院真实身份：公开展示层应使用 pseudonym，真实机构映射不应写入公开 ledger。
- 训练 artifact：每轮全局参数文件、更新 payload、实验 summary 都应有内容哈希；开启隐私模式时应支持加密落盘。
- 审计链元数据：审计记录需要证明完整性，但不应泄露 secret、明文参数或真实机构身份。

## 2. 信任边界

FLxBC 的本地 demo 可分为四个边界：

- 客户端边界：模拟医院持有本地数据，执行本地训练，生成参数更新。
- 聚合端边界：接收客户端更新，验证提交，执行聚合，写出全局模型。
- 审计边界：SQLite mock chain 或 optional Hardhat adapter 只记录哈希、状态和证明材料。
- 展示边界：FastAPI 和 Streamlit 只展示必要结果，并应过滤敏感字段。

当前实现仍是单进程内存模拟，客户端边界和聚合端边界没有物理隔离。后续隐私模块需要先用 envelope、签名、nonce、DP 和加密 artifact 模拟真实边界，再考虑多进程或多机器部署。

## 3. 攻击者模型

后续实现至少要覆盖以下攻击者：

- 半诚实聚合端：按流程运行聚合，但会尝试查看单个客户端更新。
- 恶意客户端：提交 poisoned update、重复提交旧 update、伪造其他客户端身份。
- 被动网络窃听者：在多进程或多机器部署中观察上传 payload。
- 本地 artifact 读取者：能访问 `artifacts/runs/<run_id>/` 目录并尝试读取参数文件。
- 未授权 dashboard/API 访问者：能访问本地或暴露出的 HTTP 服务并读取 ledger 记录。

## 4. 隐私目标

后续 P0 开发需要达成以下目标：

- 原始数据不离开客户端训练边界。
- 聚合端不能依赖裸参数对象作为唯一输入，必须先通过提交 envelope。
- 每个客户端提交应具备 node id、timestamp、nonce、payload hash 和认证字段。
- 重放提交、payload 篡改、身份不匹配应被拒绝并写入 misbehavior。
- DP 模式必须实际执行 update clipping 和 Gaussian noise，而不只是记录配置。
- artifact 内容哈希必须基于参数内容或文件字节，而不是仅基于 URI。
- 开启 artifact 加密时，磁盘上的参数文件不能被 `np.load` 直接读取。
- dashboard/API 默认不暴露 secret、key、完整 artifact 明文路径或真实机构身份。

## 5. 非目标

当前阶段不承诺以下能力：

- 不承诺符合 HIPAA、GDPR、等保或医疗器械软件法规。
- 不承诺生产级链上治理、跨机构密钥管理或合规审计。
- 不承诺抵抗具备主机 root 权限的本地攻击者。
- 不承诺在 demo 阶段实现完整密码学安全多方计算协议。
- 不接入真实患者数据。

这些非目标不意味着可以忽视隐私，而是避免把本地研究系统误描述为生产级医疗平台。

## 6. 最小隐私闭环

下一阶段应优先形成以下闭环：

1. 客户端训练生成本地 update。
2. update 经 clipping 和可选 DP noise 处理。
3. update 被序列化为 payload，并计算内容哈希。
4. 客户端生成包含 node id、timestamp、nonce、payload hash 的 envelope。
5. envelope 使用 demo HMAC 或后续签名机制认证。
6. 聚合端验证 envelope，拒绝重放、篡改和身份不匹配提交。
7. 聚合端执行普通聚合或安全聚合 demo。
8. 全局参数 artifact 写入内容哈希；开启加密时加密落盘。
9. ledger 记录隐私模式、DP 参数、artifact 内容哈希、认证状态和安全聚合状态。
10. dashboard/API 只展示脱敏后的隐私状态和审计状态。

## 7. 后续设计约束

- 默认 demo 必须继续快速可运行。
- 隐私增强应通过配置开关启用，不能破坏现有 smoke test。
- 所有安全降级必须显式记录，例如 secure aggregation 因 dropout 回退。
- 密钥和 secret 不得写入 ledger、artifact summary 或 dashboard。
- 任何新增隐私机制都必须配套测试，覆盖正常路径和失败路径。

