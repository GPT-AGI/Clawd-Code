# Clawd-Code：Peer-Native Coding Agent Collaboration 实现任务

你正在修改 Clawd-Code。请直接完成下面的代码、测试和文档工作，不要只给设计建议。

## 0. 研究目标与硬约束

我们要研究的问题是：

> 今天的 coding agents 究竟具不具备 peer collaboration intelligence，还是只能充当被 workflow 调用的独立执行器？

这里评测的是 Claude Code、Codex 这一类能够读写仓库、执行命令、持续行动的完整 coding agent，而不是一个由 Planner/Coder/Reviewer 节点组成的预编排 workflow。

硬约束：

1. **不训练任何模型**。不要加入训练、微调、强化学习、学习型 router 或 learned policy。
2. Peer 模式中不存在有特权的 LLM manager/lead。运行基础设施可以有非智能 supervisor，但它只能负责启动、资源限制、日志、超时和停止，不能替 agents 做任务分解、分配、冲突解决或答案选择。
3. 所有 peers 接收同一个顶层任务、同一套非通信工具、同等级权限。除稳定的 peer ID/name 外，不给任何 peer 预设 Planner、Coder、Reviewer 等角色。
4. 不预先创建 task DAG，不给 peer 分配 owned task，不通过 prompt 暗示固定分工，不硬编码协作策略。
5. 保留现有 lead-controlled teammate/team workflow 及其兼容性。新增 peer-native mode，不要把原有模式强行改成另一种语义。
6. 不要通过简单解除现有 teammate 的 forbidden tools 来伪造 peer mode。Peer mode 应有清楚、独立的控制面和权限语义。
7. 不得破坏或覆盖当前 dirty worktree 中与本任务无关的用户修改；禁止 destructive git 操作。

## 1. 开始修改前

先完成以下检查，再给出一个简短实施计划并开始实现：

- 阅读现有 teammate/team runtime、store、message tools、task tools、CLI、trace/event、workspace/worktree 和相关测试。
- 检查 `git status --short`，区分现有修改和本任务修改。
- 运行当前 teammate 相关测试作为 baseline：

```bash
.venv/bin/python -m pytest -q \
  tests/test_teammate_runtime.py \
  tests/test_teammate_store.py \
  tests/test_teammate_resilience.py
```

- 优先复用现有可靠的 session、message persistence、workspace、provider、tool registry 和 trace 能力，但不要把 `lead_agent_id` 偷换成一个“名义上是 peer、实际上有特权”的 agent。

## 2. 需要实现的核心语义

### 2.1 新增固定 N 的 peer-native run

实现一个独立、明确命名的 peer collaboration mode。具体类名和模块位置可根据现有架构选择，但代码中的概念应能区分：

- 旧的 lead-controlled `TeamRun`；
- 新的 peer-native run；
- 非 LLM 的运行 supervisor；
- 地位完全相同的 peer participants。

第一版使用固定 N 个 peers 即可。动态招募、动态扩缩容和 peer 创建 peer 暂不作为必需功能。

Peer run 启动时：

1. supervisor 创建 N 个独立、持久的 agent sessions；
2. 所有 peers 并发启动，而不是逐个串行执行；
3. 所有 peers 获得同一个顶层 mission；
4. prompt 不包含预设角色、子任务、owner、依赖关系或建议分工；
5. 每个 peer 只额外知道自己的稳定 ID/name、团队 roster 的获取方式以及可用通信协议；
6. peer 在一次局部工作完成后不能像 task worker 一样立即永久退出，而应保持可唤醒状态，直到 run 被提交、取消、超时或耗尽预算。

建议将 peer system context 保持中性，例如只说明：你是平等的 coding peer；你可以自主检查仓库、决定工作、与其他 peers 协调；任何 peer 都可以发起最终提交。不要告诉它应该如何分工。

### 2.2 最小 P2P 接口

Peer mode 至少提供以下能力：

#### `PeerList`

- 返回当前 run 中可通信的 peers、稳定 ID/name 和粗粒度 lifecycle status；
- 不返回 supervisor 规划出的任务、角色或“你应该联系谁”的建议；
- roster 对所有 peers 一致，不存在隐藏的 lead 权限。

#### `SendMessage`

- 允许任意 peer 直接给任意其他 peer 发消息；
- 不需要经过 lead 转发；
- 验证 sender/recipient 都属于当前 run；
- 未知 recipient、越权通信和非法 payload 应明确报错；
- 每条消息有稳定 ID、sender、recipient、创建/投递时间和消费状态；
- 保留现有 team mode 的兼容行为。

#### `ReadMessages`

- 读取当前 peer 的 inbox；
- 明确定义 unread/consumed 语义，重复读取不能造成消息无意丢失或重复执行；
- 支持非 busy-polling 的等待方式；
- peer 从 idle 被消息唤醒后，下一次 model boundary 必须能看到这条消息。

#### `Broadcast`

- 向当前 run 中除自己外的所有可通信 peers 发送同一消息；
- 对每个 recipient 有可审计的 delivery/consumption 记录；
- 不给 sender 自己投递；
- 重试时要有清楚的幂等规则，不能静默重复广播。

#### `TeamSubmit`（或语义等价的 `PeerSubmit`）

- 任意 peer 都可以提交最终结果；
- 参数至少包含最终 commit hash 或可验证的 workspace revision，以及简短说明；
- supervisor 验证 revision 属于本次允许的仓库/工作空间且真实存在；
- 第一份原子接受的有效提交结束 run，后续并发提交返回同一个已接受结果或清楚的 already-submitted 状态；
- 记录 submitting peer、revision、时间和验证结果；
- 不要求预设 lead 才能提交，也不要由隐藏 judge 在多个候选中替 agents 做智能选择。

工具命名可以与现有风格协调，但用户可见语义必须完整。

### 2.3 Persistent、event-driven peer loop

当前 task-bounded worker “完成当前 task 后退出/idle” 的语义不够。Peer mode 需要：

- peer session 在 run 生命周期内持续存在；
- 没有可做工作时进入可观测 idle 状态；
- 收到新消息时可被事件驱动地唤醒，而不是只能依靠 agent 碰巧轮询 inbox；
- 不用高频 polling 或 busy loop；
- run submit/cancel/timeout/budget exhausted 时能够干净停止所有 peers；
- 停止后不能继续执行 tool call；
- 异常 peer 不应导致其他 peer 或 supervisor 永久死锁；
- 并发发送、读取、广播、提交必须线程安全；若当前 backend 只支持线程，要明确隔离边界并为未来独立进程 backend 留出接口。

第一版允许复用现有线程池，但不要在文档中把同进程线程描述成强进程隔离。真实 Claude Code/Codex CLI 进程适配器可放在后续阶段。

### 2.4 Workspace 与集成语义

同时保留两种实验能力：

- `shared`：所有 peers 直接操作同一工作区，用于研究 contention/race；
- `worktree`：每个 peer 有独立 worktree，用于更可控的实验。

Peer benchmark 的科学默认值建议使用独立 worktree，并遵循：

- 不启用旧 team workflow 的隐藏 `auto_integrate`；
- peers 通过消息交换接口、文件、分支或 commit 信息；
- peers 自己使用正常 git 操作整合彼此工作；
- 最终由任意 peer 使用 `TeamSubmit(revision=...)` 提交一个可验证 revision；
- trace 能把 commit/worktree 与 peer 对应起来；
- shared 模式和 worktree 模式都必须有清楚的 teardown，不残留失控 worker。

### 2.5 Agent backend 抽象

不要让 peer benchmark 永久耦合到单一 provider 或当前内部 agent loop。

请定义尽量小的 peer runner/session adapter 边界：

- 当前 Clawd agent loop 是第一个可用 backend；
- 测试中可注入 deterministic/scripted fake backend；
- 后续可以接 Claude Code CLI、Codex CLI 等完整 coding agent 进程；
- 本次不要求真的实现所有外部 CLI adapter，但接口和生命周期不能阻止它们接入。

## 3. 实验条件必须由协议控制，不靠角色 prompt 模拟

为 benchmark 增加 communication policy/condition。至少支持并测试：

1. `solo`：1 个 agent；
2. `independent` / `none`：N 个 agent，同一个顶层任务，但无 peer 消息；
3. `artifact-only`：N 个 agent，无消息工具，仅通过实验允许的仓库/artifact 可见性协作；
4. `star`：N 个 agent，只有指定 coordinator peer 可以与其他 peers 通信，普通 peers 不能直接互发；
5. `p2p`：N 个 agents 可任意 direct message 和 broadcast。

要求：

- policy 在 tool registry/transport ACL 层执行，不是只在 prompt 里写“请不要通信”；
- 对同一 backend 和同一实验，除 communication policy 所必需的工具差异外，非通信工具、顶层任务、模型设置、预算口径应一致；
- `star` 中 coordinator 是实验通信拓扑中的普通 agent 节点，不获得额外代码权限、预算或 supervisor 控制权；
- 非法边必须被拒绝并留下 trace；
- 设计上允许以后加入局部图、带宽限制、延迟和消息成本，但本次不必全部实现。

如果一次改动无法安全完成全部条件，优先级为：`p2p`、`artifact-only`、`independent`、`solo`、`star`。不过交付时必须明确未完成项，不能用 prompt 约束假装协议已经实现。

## 4. CLI / API

为 peer run 增加一个最小、可复现的入口。命令形式可以服从现有 CLI 规范，功能上至少能表达：

```text
run peer collaboration
  --repo /path/to/repo
  --prompt-file TASK.md
  --peers N
  --communication solo|independent|artifact-only|star|p2p
  --workspace-mode shared|worktree
  --model ...
  --timeout-seconds ...
  --max-turns ...
  --token-budget ...
  --output-dir ...
```

要求：

- 所有会影响结果的配置写入 run manifest；
- 支持非交互运行和明确 exit code；
- API/CLI 参数校验覆盖 N、condition、workspace、预算和路径；
- 不影响现有 team CLI；
- 文档给出一个无需真实 API 的 scripted smoke 示例，以及一个真实模型运行示例。

## 5. Trace、结果格式与可计算指标

Peer benchmark 首先要保证原始事实可审计，不要在 runtime 中强行判断“这次协作好不好”。

每次 run 至少记录：

- run ID、condition、N、model/provider、任务/仓库 revision、workspace mode；
- 每个 peer 的 session ID、start/idle/wake/stop/error 时间；
- model call、tool call、输入/输出/cache tokens、预算消耗；
- message created/delivered/consumed，sender、recipient、message ID、时间戳、payload 大小；
- broadcast ID 与实际 recipients；
- policy rejection；
- worktree/branch/commit 与 peer 的关联；
- submit attempt、accepted submit、revision、submitting peer；
- timeout、cancel、budget exhaustion 和异常；
- 最终 wall-clock time、aggregate tokens/calls、验收测试结果。

事件同时记录 wall-clock 和适合计算延迟的 monotonic 时间（或等价可靠设计）。结果应能离线计算：

- direct P2P edge 和通信图；
- delivery/consumption/response latency；
- message volume 与成本；
- accepted solution quality；
- wall time、总 token、每 peer token；
- commit/work attribution；
- 重复工作、stale work、冲突和 rework 的代理指标；
- P2P 相对 `solo`、`independent`、`artifact-only`、`star` 的差异。

不要把单次运行的差异直接宣称为 scaling law 或 causal result。统计聚合由 benchmark analysis 层完成。

## 6. Benchmark 目录

在现有 eval 结构中新增一个名称清楚的目录，例如：

```text
teammate-evals/peer-collaboration/
```

至少包含：

- `README.md`：研究问题、非训练设定、五种 condition、运行方式、输出字段、限制；
- run/config schema 或等价配置；
- 可重复的 scripted smoke fixture；
- 调用 peer runtime 的 runner；
- 对结果 JSON/JSONL 的 schema 验证；
- 一个 coupled task 示例，必须需要至少一次接口对齐、信息传播或对 peer 工作的适应，不能只是两个完全独立文件的机械拼接；
- real-model pilot 入口默认不在普通单元测试中运行，必须显式 opt-in。

可以复用 `teammate-evals/nl2repo-pilot` 的仓库任务和 acceptance-test 方式，但不要复用其中 lead 预先分配 task 的控制逻辑。

## 7. 必须添加的测试

测试默认不得依赖网络、真实 API key 或非确定模型。使用 scripted/fake backend 测 runtime 语义；真实模型只做 opt-in pilot。

### 7.1 单元测试

覆盖：

- 所有 peers 看到同一 roster，且没有隐藏 lead 权限；
- peer A 可直接给 peer B 发消息，不经过第三方；
- unknown/self/跨 run recipient 的明确行为；
- unread、consumed、重复 read 的语义；
- broadcast 恰好投递给每个其他 peer 一次，不给自己；
- broadcast 重试/幂等；
- message persistence 在 store reload 后仍正确；
- `independent`、`artifact-only`、`star`、`p2p` 的 ACL；
- 非法通信产生 rejection event；
- 任意 peer 均可 submit；
- invalid revision 被拒绝；
- 两个并发有效 submit 只有一个原子胜出，结果可重复读取；
- prompt/context 不含预设 Planner/Coder/Reviewer、owned task 或 DAG；
- peer 间除 ID 外的工具和权限对称；
- 现有 team mode 的 `SendMessage` / `ReadMessages` 行为保持兼容。

### 7.2 并发与生命周期测试

覆盖：

- N 个 peers 实际重叠执行，且 session ID 独立；
- recipient 已 idle 后收到消息会被唤醒，并在下一 model boundary 看到消息；
- 多个 peers 同时 send/read 不丢失、不重复、不死锁；
- 同时 broadcast 的 delivery 数正确；
- 一个 peer submit 后，其余 peers 被干净停止；
- accepted submit 之后不再出现新的业务 tool call；
- timeout、cancel、peer crash、budget exhaustion 不留下 orphan worker；
- worktree 模式下 peer A 的 commit 可被 peer B 正常获取/整合，最终 revision 可验证；
- shared 模式的并发行为至少有一个 race-safe smoke test；
- store/event 写入在并发下不产生损坏 JSON 或丢事件。

### 7.3 Benchmark protocol 测试

覆盖：

- 五种 conditions 能从同一任务配置生成，模型/预算/非通信工具保持可比较；
- `artifact-only` 和 `independent` 不应意外暴露 message tools；
- `star` 的 worker→worker 被拒绝，worker↔coordinator 被允许；
- `p2p` trace 中可以出现非 coordinator 的 peer↔peer edge；
- manifest 完整记录所有实验参数；
- result schema 可验证；
- acceptance test 的 stdout/stderr/exit code 被保存；
- scripted coupled-task smoke 中，两个 peers 可通过消息完成一次接口协商并提交有效结果；
- 不要在单元测试中断言真实模型一定会表现出“聪明协作”，只断言机制、可观测性和协议正确。

### 7.4 回归测试

至少运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_teammate_runtime.py \
  tests/test_teammate_store.py \
  tests/test_teammate_resilience.py

.venv/bin/python -m pytest -q <新增的 peer runtime/tool/protocol 测试>

.venv/bin/python -m pytest -q
```

如果项目配置了 Ruff/type checker，也运行与改动文件对应的 lint/type check。不要通过放宽全局规则、删除断言或跳过测试来获得绿色结果。

## 8. 建议的实现阶段

### P0：保护兼容性

- 固定现有 teammate tests 的 baseline；
- 将 peer mode 与 lead-controlled mode 的状态和权限分开；
- 复用组件时保持旧 API/serialization 兼容。

### P1：Peer runtime MVP（本任务必需）

- 固定 N、对等 session、统一 mission；
- `PeerList`、direct send/read、broadcast、submit；
- persistent idle/wake loop；
- `p2p`、`artifact-only`、`independent`、`solo`；
- shared/worktree；
- manifest、trace、scripted tests。

### P2：完整 benchmark protocol（尽量在本任务完成）

- `star` ACL；
- benchmark 目录、coupled fixture、result schema；
- CLI、文档、全量回归。

### 暂不要求

- 训练或 learned policy；
- 动态创建/销毁 peers；
- 自动选择最优拓扑；
- 真正的网络分布式执行；
- 完整 Claude Code/Codex CLI adapters；
- 大规模真实模型实验或论文结论。

## 9. 验收标准（Definition of Done）

只有同时满足以下条件才算完成：

- [ ] 新 peer mode 中不存在有特权的 LLM lead/manager；
- [ ] peers 的任务、非通信工具、权限和预算口径对称，除 peer identity 外没有预设角色；
- [ ] runtime 不创建 owned tasks 或预定义 task DAG；
- [ ] peer A 能直接联系 peer B，消息可持久化、审计并唤醒 idle peer；
- [ ] broadcast、communication ACL 和并发 submit 行为确定且有测试；
- [ ] 任意 peer 能提交最终 revision，提交后所有 peers 干净停止；
- [ ] `solo`、`independent`、`artifact-only`、`p2p` 可运行；`star` 完成或明确列为唯一剩余协议项；
- [ ] shared/worktree 语义清楚，peer mode 不使用隐藏 auto-integrate；
- [ ] 默认测试不依赖真实模型或网络；
- [ ] 新测试通过，原有 teammate 测试和全量测试无回归；
- [ ] README 能让另一位研究者复现实验并理解其局限；
- [ ] 没有训练代码、固定角色 workflow 或通过 prompt 假装实现 ACL；
- [ ] 没有覆盖当前 worktree 中无关的用户修改。

## 10. 最终交付回复格式

实现完成后，请给出：

1. 实际实现了哪些语义；
2. 关键文件列表；
3. 运行过的测试命令和结果；
4. 一个 scripted smoke 命令；
5. 一个真实模型 pilot 命令（不要实际消耗 API，除非明确授权）；
6. 仍存在的限制，尤其是线程/进程隔离、外部 agent adapter 和未完成的 condition；
7. 说明如何确认 peer mode 中没有隐藏 lead、固定角色或 task DAG。

如果某个要求与现有架构冲突，不要静默弱化要求。先用代码证据说明冲突，再选择最小且兼容的设计；只有在会显著改变研究语义时才停下来询问。
