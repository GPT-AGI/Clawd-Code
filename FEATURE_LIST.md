# Claude Code Python Feature List & PR Roadmap

> 面向社区贡献者的能力清单、路线图与 PR 指南。
>
> 项目定位：**基于真实 Claude Code 源码结构的 Python 重构版**。当前已经具备可用的多 Provider 聊天 CLI、完整工具系统框架与 Agent Loop，正在分阶段完善原生 Claude Code 的关键能力。

---

## 状态说明

| 状态 | 含义 |
|------|------|
| ✅ 已实现 | 当前仓库中已有可验证实现 |
| 🟡 部分完成 | 已有骨架、镜像层或部分能力，但未形成完整闭环 |
| ⏳ 规划中 | 已明确方向，欢迎提交 PR |
| 🚫 未开始 | 当前尚无实现 |

---

## 项目亮点

- **Python 重构版**：不是单纯 UI 模仿，而是按 Claude Code 的架构思路重建。
- **多模型先行**：当前已支持 Anthropic、OpenAI、GLM 三类 Provider。
- **CLI / REPL 可用**：已经具备基础交互能力，适合持续迭代。
- **工具系统框架完整**：已实现 30+ 工具模块、Agent Loop、权限系统框架。
- **更适合社区共建**：Python 生态更易二开，适合工具、自动化、数据工程场景扩展。
- **强调真实性**：优先补齐真正可运行的核心链路，而不是只扩大命令/工具名录。

---

## 核心系统

| 能力 | 状态 | 当前情况 |
|------|------|----------|
| CLI 启动入口 | ✅ | 已支持 `clawd`、`login`、`config`、`--version` |
| 交互式 REPL | ✅ | 支持交互式输出、历史记录、Tab 补全、多行输入 |
| Slash Commands | ✅ | 已支持 `/help`、`/clear`、`/save`、`/load`、`/multiline`、`/exit` |
| 多 Provider 抽象 | ✅ | 已支持 Anthropic / OpenAI / GLM |
| Provider 配置管理 | ✅ | 支持默认 Provider、Base URL、默认模型配置 |
| 会话持久化 | ✅ | 支持保存/加载本地会话 |
| 会话消息管理 | ✅ | 支持会话历史维护与序列化 |
| 错误恢复 / 重新登录 | 🟡 | 已有基础认证错误处理与重新配置流程 |
| Token / Cost 跟踪 | 🚫 | 当前聊天 CLI 尚未形成完整统计视图 |
| 上下文构建 | 🟡 | 已有 `context_system` 基础版，支持 workspace / git / `CLAUDE.md` 注入，仍缺 README 摘要、memory、compact |
| Claude Code Agent Loop | ✅ | 已实现 agent_loop.py，支持工具调用循环 |
| 非交互任务执行 | ✅ | `clawd run` 支持 prompt 文件、stdin、workspace、模型与 turn 限制 |
| Teammate Runtime | ✅ | 持久化团队、独立会话、lease 恢复、重试、并行调度、预算、取消与消息交接 |
| Teammate Trace Viewer | ✅ | `clawd trace` 提供实时事件、工具调用、消息与任务依赖可视化 |
| Teammate Worktree | ✅ | 可选 Git worktree 隔离，支持自动或 lead 手动整合 |
| `/resume` 会话恢复体验 | 🚫 | 暂无独立恢复流程与 UI |
| `/compact` 对话压缩 | 🚫 | 暂无自动/手动压缩能力 |
| `/doctor` 诊断系统 | 🚫 | 暂无环境、配置、权限、依赖诊断命令 |
| Hook 系统 | 🚫 | 暂无 pre/post tool use hooks |
| 权限系统 | 🟡 | 已有 permissions.py 框架，尚未完全集成 |

---

## 工具系统

> **重大进展**：当前仓库已实现完整的工具系统框架，包括 30+ 工具模块、Agent Loop、Schema 验证、权限框架等。

### 工具框架

| 能力 | 状态 | 当前情况 |
|------|------|----------|
| Tool Registry | ✅ | 已实现工具注册与发现机制 |
| Tool Protocol | ✅ | 已定义工具协议与基类 |
| Schema Validation | ✅ | 已实现参数校验系统 |
| Agent Loop | ✅ | 已实现完整的工具调用循环 |
| Tool Context | ✅ | 已实现工具上下文管理 |
| Permission Framework | 🟡 | 已有权限检查框架，待完善集成 |
| Error Handling | ✅ | 已定义工具错误类型与处理 |
| Task Manager | ✅ | 已实现任务管理器 |

### 已实现工具模块

| 工具类别 | 工具名称 | 文件 | 状态 |
|---------|---------|------|------|
| 文件操作 | FileReadTool | `read.py` | ✅ 已实现 |
| 文件操作 | FileWriteTool | `write.py` | ✅ 已实现 |
| 文件操作 | FileEditTool | `edit.py` | ✅ 已实现 |
| 文件操作 | GlobTool | `glob.py` | ✅ 已实现 |
| 文件操作 | GrepTool | `grep.py` | ✅ 已实现 |
| 系统操作 | BashTool | `bash.py` | ✅ 已实现 |
| 网络工具 | WebFetchTool | `web_fetch.py` | ✅ 已实现 |
| 网络工具 | WebSearchTool | `web_search.py` | ✅ 已实现 |
| 交互工具 | AskUserQuestionTool | `ask_user_question.py` | ✅ 已实现 |
| 交互工具 | SendUserMessageTool | `send_user_message.py` | ✅ 已实现 |
| 任务管理 | TodoWriteTool | `todo_write.py` | ✅ 已实现 |
| 任务管理 | TaskStopTool | `task_stop.py` | ✅ 已实现 |
| 任务管理 | TasksV2Tool | `tasks_v2.py` | ✅ 已实现 |
| 任务管理 | TaskManager | `task_manager.py` | ✅ 已实现 |
| Agent 工具 | AgentTool | `agent.py` | ✅ 已实现 |
| Agent 工具 | BriefTool | `brief.py` | ✅ 已实现 |
| Agent 工具 | TeamTool | `team.py` | ✅ 已实现 |
| 配置工具 | ConfigTool | `config.py` | ✅ 已实现 |
| 计划模式 | PlanModeTool | `plan_mode.py` | ✅ 已实现 |
| 定时任务 | CronTool | `cron.py` | ✅ 已实现 |
| MCP 工具 | MCPTool | `mcp.py` | ✅ 已实现 |
| MCP 工具 | MCPResourcesTool | `mcp_resources.py` | ✅ 已实现 |
| 技能系统 | SkillTool | `skill.py` | ✅ 已实现 |
| 工具搜索 | ToolSearchTool | `tool_search.py` | ✅ 已实现 |
| LSP 集成 | LSPTool | `lsp.py` | ✅ 已实现 |
| Worktree | WorktreeTool | `worktree.py` | ✅ 已实现 |
| 杂项工具 | SleepTool | `sleep.py` | ✅ 已实现 |
| 杂项工具 | StructuredOutputTool | `structured_output.py` | ✅ 已实现 |
| 杂项工具 | MiscTools | `misc.py` | ✅ 已实现 |

---

## 服务与运行时

| 模块 | 状态 | 当前情况 |
|------|------|----------|
| Provider Runtime | ✅ | 已能完成基础聊天请求，Provider 层提供流式接口 |
| REPL Runtime | ✅ | 已支持基础交互、命令分流、消息记录 |
| Agent Loop Runtime | ✅ | 已实现完整的工具调用循环与结果处理 |
| Tool Execution Engine | ✅ | 已实现工具加载、执行、结果回填闭环 |
| Output Styles | ✅ | 已实现输出样式加载系统 |
| Session Persistence | ✅ | 已有会话保存/加载能力 |
| Context Engine | 🟡 | 已接入基础上下文构建链路，支持 workspace、git、`CLAUDE.md` prompt 注入 |
| Permission Engine | 🟡 | 已有框架，未完全集成到工具执行流程 |
| Compaction Engine | 🚫 | 未形成对话压缩与 token 管理能力 |
| Hook Runtime | 🚫 | 未接入设置驱动的 hook 执行机制 |
| MCP Runtime | 🟡 | 已有 MCP 工具，未形成完整 MCP 协议层 |

---

## 测试覆盖

| 测试类型 | 状态 | 文件 |
|---------|------|------|
| 工具系统测试 | ✅ | `test_tool_system_tools.py` (427 行) |
| Agent Loop 测试 | ✅ | `test_agent_loop.py` (134 行) |
| Claude Code 工具对等性测试 | ✅ | `test_claude_code_tool_parity.py` (137 行) |
| Provider 测试 | ✅ | `test_providers.py` (113 行) |
| 输出样式测试 | ✅ | `test_output_styles.py` (64 行) |
| 配置测试 | ✅ | `test_config.py` |
| Teammate runtime / trace 测试 | ✅ | `test_teammate_runtime.py`、`test_teammate_viewer.py` |
| Teammate resilience 评测 | ✅ | crash-resume、retry、review-reject、cancel、budget、parallel、worktree |
| 非交互 runner 测试 | ✅ | `test_runner.py` |


## 路线图

## Phase 0：可启动、可安装、可体验 ✅

目标：先保证项目对新用户和贡献者足够顺滑。

- [x] 解耦 CLI 启动路径，`--help`、`--version`、`config` 不应依赖 Provider SDK
- [x] Provider 改为延迟导入，缺少 SDK 时仍可浏览本地功能
- [x] 固定并验证 Python 3.11+ 开发环境
- [x] 完善安装说明与最小可运行示例
- [x] 清理 README 中与当前实现不一致的表述

## Phase 1：Claude Code 核心体验 MVP ✅

目标：先复现原生 Claude Code 最重要的第一层体验。

- [x] 统一聊天 REPL、slash commands、session store
- [x] 完成工具系统框架
- [x] 实现 Agent Loop
- [x] 统一错误处理、重试、重登流程
- [x] 完成 transcript 持久化与恢复基础设施
- [x] 整理一套稳定的用户命令集合

## Phase 2：真实工具调用闭环 ✅

目标：从"镜像工具清单"走向"真正可执行的 Python Agent"。

- [x] FileReadTool
- [x] FileWriteTool
- [x] FileEditTool
- [x] BashTool
- [x] AskUserQuestionTool
- [x] TodoWriteTool
- [x] WebFetchTool / WebSearchTool
- [x] 工具 schema、参数校验、异常处理、调用日志
- [x] 工具执行结果回填闭环

## Phase 3：上下文、权限、恢复能力 (进行中)

目标：补齐 Claude Code 的工程化能力。

- [ ] 工作区上下文构建完善
- [x] git status / 文件树 / `CLAUDE.md` 注入基础版
- [ ] README / 入口文件摘要注入
- [ ] memory 与历史上下文管理
- [ ] 权限系统完全集成
- [ ] `/resume`
- [ ] `/compact`
- [ ] `/doctor`
- [ ] pre/post tool use hooks

## Phase 4：MCP、插件、扩展生态

目标：把项目从单体 CLI 升级为可扩展平台。

- [ ] MCP client/runtime 完善
- [ ] Python 插件系统
- [ ] 自定义 commands / tools / hooks
- [ ] 本地模型与第三方 provider 扩展
- [ ] 更完善的 observability 与调试工具

## Phase 5：Python 版本的差异化亮点

目标：做出属于 Python 重构版的特色。

- [ ] Notebook 友好工具链
- [ ] 数据工程 / ETL 场景增强
- [ ] 中国模型生态一等公民支持
- [ ] pytest / ruff / mypy / uv 集成体验
- [ ] 面向企业内自动化与工作流的扩展接口

---

## 我们期待的 PR

### P0：最欢迎、最容易合并

- 测试覆盖增强
- 文档完善
- 错误处理改进
- 性能优化

### P1：高价值基础能力

- 上下文自动构建
- 权限系统完全集成
- `/resume` 实现
- `/compact` 实现
- `/doctor` 实现

### P2：Claude Code 关键体验补齐

- Hook 系统
- MCP 完善支持
- Token/Cost 统计
- 性能监控与调优

### P3：Python 版亮点方向

- Notebook 编辑与读取增强
- 数据文件工具增强
- pytest / ruff / mypy / uv 集成
- 更多国内外模型 provider
- 可插拔工具系统

---

## 建议的 PR 认领模块

| 方向 | 适合贡献内容 |
|------|--------------|
| CLI / UX | 命令设计、帮助信息、交互体验、错误提示 |
| Tools | 工具增强、新工具开发、工具测试 |
| Context | repo map、git status、项目文档注入、memory |
| Permissions | 权限集成、安全策略、命令限制 |
| Providers | 新 provider、模型选型、流式兼容 |
| MCP / Plugins | MCP runtime 完善、插件装载、自定义工具扩展 |
| Quality | 测试、基准、文档、安装流程、CI |
| Performance | 性能优化、内存管理、并发处理 |

---

## PR 提交建议

- 优先做真实能力，不优先堆命令名录
- 每个 PR 尽量聚焦单一模块
- 新功能请附带最小测试或运行示例
- 修改 README 时请同步更新能力状态
- 对"已完成"表述保持谨慎，优先写成可验证结果

---

## 推荐的对外表述

可以这样介绍项目：

> Claude Code Python 是一个基于真实 Claude Code 源码结构的 Python 重构版。当前已经具备多 Provider 聊天 CLI、完整工具系统框架（30+ 工具）与 Agent Loop，已实现工具调用闭环，正在完善上下文构建、权限系统、会话恢复、压缩、MCP 与插件体系。欢迎围绕工具增强、runtime、permissions、context 与 Python 原生扩展能力提交 PR。

---

## 一句话总结

**当前我们已经有一个具备完整工具系统框架和 Agent Loop 的 Python Agent Runtime；接下来要做的是，完善上下文构建、权限集成、恢复能力，把它真正推进成具备 Claude Code 完整体验的 Python Agent 平台。**
