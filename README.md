<div align="center">

**English** | [中文](#中文版) | [Français](docs/i18n/README_FR.md) | [Русский](docs/i18n/README_RU.md) | [हिन्दी](docs/i18n/README_HI.md) | [العربية](docs/i18n/README_AR.md) | [Português](docs/i18n/README_PT.md)

# 🚀 Claude Code Python

**A Complete Python Reimplementation Based on Real Claude Code Source**

*From TypeScript Source → Rebuilt in Python with ❤️*

***

[![GitHub stars](https://img.shields.io/github/stars/GPT-AGI/Clawd-Code?style=for-the-badge&logo=github&color=yellow)](https://github.com/GPT-AGI/Clawd-Code/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GPT-AGI/Clawd-Code?style=for-the-badge&logo=github&color=blue)](https://github.com/GPT-AGI/Clawd-Code/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)


**🔥 Active Development • New Features Weekly 🔥**

</div>

***

## 🎯 Why Clawd Code?

**Clawd Code** is a **production-oriented Python rebuild of Claude Code**, ported from the **real TypeScript architecture** and shipped as a **working CLI agent**, not just a source dump.

- **Real Agent Runtime** — tool-calling loop, streaming REPL, session history, and multi-turn execution
- **High-Fidelity Port** — keeps the original Claude Code architecture while adapting it to idiomatic Python
- **Built to Hack On** — readable Python codebase, rich tests, and markdown-driven skill extensibility

<div align="center">

**Token Streaming + Tool-Aware Agent Loop**

![Streaming Agent Experience](assets/clawd-stream.gif)

**Programmable Skill Runtime with Tool Sandboxing**

![Skills (Slash Commands)](assets/clawd-code-skill.png)

**Instant Web Fetch for External Context**

![Web Fetch](assets/claude-code-webfetch.png)

**Real CLI • Real Usage • Real Community**

</div>

**A real Claude Code-style terminal workflow in Python: stream replies, call tools, fetch context, and extend behavior with skills.**

**🚀 Try it now! Fork it, modify it, make it yours! Pull requests welcome!**

***

## ⭐ Star History

<a href="https://www.star-history.com/?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left" />
 </picture>
</a>

## ✨ Features

### Streaming Agent Experience

```text
>>> /stream on
>>> Explain tests/test_agent_loop.py
[streaming answer...]
• Read (tests/test_agent_loop.py) running...
  ↳ lines 1-180
>>> /render-last
```

- True API streaming for direct replies plus richer streaming during tool-driven agent loops
- Built-in `/stream` toggle for live output and `/render-last` for clean Markdown re-rendering on demand
- Designed for real terminal demos: streaming text, visible tool activity, and stable fallback behavior

### Programmable Skill Runtime

```md
---
description: Explain code with diagrams and analogies
allowed-tools:
  - Read
  - Grep
  - Glob
arguments: [path]
---

Explain the code in $path. Start with an analogy, then draw a diagram.
```

- Markdown-based `SKILL.md` slash commands
- Supports project skills, user skills, named arguments, and tool limits

### Multi-Provider Support

```python
providers = ["Anthropic Claude", "OpenAI GPT", "Zhipu GLM"]  # + easy to extend
```

### Interactive REPL

```text
>>> Hello!
Assistant: Hi! I'm Clawd Codex, a Python reimplementation...

>>> /help         # Show commands
>>> /             # Show all commands & skills
>>> /save         # Save session
>>> /multiline    # Multi-paragraph input
>>> Tab           # Auto-complete
>>> /explain-code qsort.py   # Run a skill
```

### Complete CLI

```bash
clawd              # Start REPL
clawd login        # Configure API
clawd --version    # Check version
clawd config       # View settings
clawd config --use glm5      # Switch to Z.ai GLM-5.2 profile
clawd config --use qwen3.5   # Switch to Tencent TI-ONE Qwen 3.5 profile
```

***

## 📊 Status

| Component     | Status     | Count     |
| ------------- | ---------- | --------- |
| REPL Commands | ✅ Complete | 6+ built-ins |
| Tool System   | ✅ Complete | 30+ tools |
| Automated Tests | ✅ Present | Core suites for skills, providers, REPL, tools, context |
| Documentation | ✅ Complete | 10+ docs  |

### Core Systems

| System | Status | Description |
|--------|--------|-------------|
| CLI Entry | ✅ | `clawd`, `login`, `config`, `--version` |
| Interactive REPL | ✅ | Rich interactive output, history, tab completion, multiline |
| Multi-Provider | ✅ | Anthropic, OpenAI, GLM, Qwen/TI-ONE support |
| Session Persistence | ✅ | Save/load sessions locally |
| Agent Loop | ✅ | Tool calling loop implementation |
| Skill System | ✅ | SKILL.md-based slash-command skills with args + tool limits |
| Context Building | 🟡 | Initial prompt injection for workspace, git, and CLAUDE.md; deeper project understanding still needed |
| Permission System | 🟡 | Framework exists, needs integration |

### Tool System (30+ Tools Implemented)

| Category | Tools | Status |
|----------|-------|--------|
| File Operations | Read, Write, Edit, Glob, Grep | ✅ Complete |
| System | Bash execution | ✅ Complete |
| Web | WebFetch, WebSearch | ✅ Complete |
| Interaction | AskUserQuestion, SendMessage | ✅ Complete |
| Task Management | TodoWrite, TaskManager, TaskStop | ✅ Complete |
| Agent Tools | Agent, Brief, Team | ✅ Complete |
| Configuration | Config, PlanMode, Cron | ✅ Complete |
| MCP | MCP tools and resources | ✅ Complete |
| Others | LSP, Worktree, Skill, ToolSearch | ✅ Complete |

### Roadmap Progress

- ✅ **Phase 0**: Installable, runnable CLI
- ✅ **Phase 1**: Core Claude Code MVP experience
- ✅ **Phase 2**: Real tool calling loop
- 🟡 **Phase 3**: Context, permissions, recovery (in progress)
- ⏳ **Phase 4**: MCP, plugins, extensibility
- ⏳ **Phase 5**: Python-native differentiators

**See [FEATURE_LIST.md](FEATURE_LIST.md) for detailed feature status and PR guidelines.**

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/GPT-AGI/Clawd-Code.git
cd Clawd-Code

# Create venv (uv recommended)
uv venv --python 3.11
source .venv/bin/activate

# Install
uv pip install -r requirements.txt
```

### Configure

#### Option 1: Interactive (Recommended)

```bash
python -m src.cli login
```

This flow will:

1. ask you to choose a provider: anthropic / openai / glm
2. ask for that provider's API key
3. optionally save a custom base URL
4. optionally save a default model
5. set the selected provider as default

For the preconfigured benchmark model profiles, switching does not require
re-entering keys:

```bash
clawd config --use glm5
clawd config --use qwen3.5
```

The Qwen profile uses the TI-ONE service group ID `ms-mnhdj86z` as its model
and the service URL ending in `/ms-mnhdj86z/v1`. Store the service AuthToken
through `clawd login` or `QWEN_API_KEY`; it is sent as the Authorization header
value and is never committed to the repository.

The configuration file is saved in in `~/.clawd/config.json`. Example structure:

```json
{
  "default_provider": "glm",
  "providers": {
    "anthropic": {
      "api_key": "base64-encoded-key",
      "base_url": "https://api.anthropic.com",
      "default_model": "claude-sonnet-4-20250514"
    },
    "openai": {
      "api_key": "base64-encoded-key",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4"
    },
    "glm": {
      "api_key": "base64-encoded-key",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "default_model": "glm-4.5"
    }
  }
}
```

### Run

```bash
python -m src.cli          # Start REPL
python -m src.cli --help   # Show help
clawd run -C ./project --prompt-file TASK.md  # Run one task non-interactively
clawd trace ./project      # Inspect teammate traces locally
clawd peer run --repo ./project --prompt-file TASK.md --peers 2 --communication p2p --workspace-mode worktree
```

`clawd run` also accepts a quoted prompt or piped stdin. Local file operations
are scoped to the selected workspace, progress is written to stderr, and the
final answer is written to stdout for scripting and CI use. Provider credentials
can come from `clawd login` or standard environment variables such as
`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and `OPENAI_API_KEY`.
Peer-native runs create equal persistent sessions without a lead or owned task
graph. See `teammate-evals/peer-collaboration/README.md` for conditions,
scripted smoke tests, output schemas, and real-model pilot commands.

**That's it!** Start chatting with AI in 3 steps.

***

## 💡 Usage

### REPL Commands

| Command      | Description           |
| ------------ | --------------------- |
| `/`          | Show commands & skills |
| `/help`      | Show all commands     |
| `/save`      | Save session          |
| `/load <id>` | Load session          |
| `/multiline` | Toggle multiline mode |
| `/clear`     | Clear history         |
| `/exit`      | Exit REPL             |

### Teammate Workflows

The lead can create persistent teammates, assign dependency-aware tasks, and
run them with bounded concurrency.

The lead first decides whether collaboration is worthwhile; using no team is a
valid outcome. When delegation helps, the lead chooses any task-specific roles,
models, tool allowlists, workspace modes, dependencies, and concurrency. There
is no required planner/coder/reviewer pipeline. Teammates can communicate
directly with one another using `SendMessage` and poll peer replies with
`ReadMessages`, so the communication topology emerges from the work.

```text
TeamCreate -> TeammateCreate -> TaskCreate -> TeamRun
```

For repository-generation work, enable `quality_gates` on `TeamCreate` and use:

```text
TeamCreate -> TeamConfigure -> 2+ TeammateCreate/TaskCreate -> TeamRun -> TeamVerify
```

Strict plans require two independently runnable owners, concrete non-overlapping
`ownedFiles`, per-task `acceptanceChecks`, and explicit provided/consumed interfaces.
An interface dependency must be represented in `blockedBy` and acknowledged by a
direct peer message. Finished workers leave the team in `verification_required`;
`TeamVerify` creates a fresh virtual environment and runs configured install, import,
and integration checks before the team can transition to `completed`.

Creation tools return structured `next_required_actions`; creating a teammate
does not start it. A worker runs only after it owns a task and the lead calls
`TeamRun`. If the lead tries to finish with an active team that has not settled,
the agent loop returns a lifecycle warning and requires the lead to run, resume,
or explicitly delete the team first.

`TeamRun` accepts `max_workers`, `max_batches`, `max_retries`,
`lease_timeout_s`, `timeout_s`, `token_budget`, and `turn_budget`. Use
`TeamResume` to recover expired leases or
resume a failed/cancelled team, `TaskRetry` for an explicit task retry, and
`TeamCancel` for cooperative cancellation. Every state transition, model call,
tool call, and message handoff is persisted for `clawd trace`.
Set `max_batches` to return control after a bounded number of scheduling batches
so the lead can inspect progress, add or reassign tasks, adjust dependencies,
stop or resume workers, and then continue the team.

Only the lead may call `TeammateStop` or `TeammateResume`. `TeammateStop` stops
one worker without cancelling the team and applies `task_policy: "requeue"`
(the default) or `task_policy: "cancel"` to unfinished work. Active model and
tool calls stop cooperatively at the next safe boundary; they are not force-killed.

Human operators can inspect and control the same persisted state without a lead
model round trip. The worker lifecycle and process-based force-stop migration
are documented in `docs/guide/TEAMMATE_WORKER_LIFECYCLE.md`:

```bash
clawd team list -C ./project
clawd team status -C ./project
clawd team stop coder --task-policy requeue -C ./project
clawd team resume-worker coder -C ./project
clawd team reassign implementation replacement-coder -C ./project
clawd team cancel --reason "operator request" -C ./project
clawd team resume --provider anthropic --model glm-5.2 -C ./project
```

Set `workspace_mode: "worktree"` on `TeammateCreate` for git isolation. With
`auto_integrate: true`, successful changes are committed in the isolated
worktree and cherry-picked into the lead repository. Downstream reviewers that
must inspect newly integrated changes should use the shared workspace. The
resilience evaluator is in `teammate-evals/runtime-resilience/`.

The five-scenario real-model benchmark in `teammate-evals/solo-vs-team/`
compares solo and adaptive lead-controlled execution on identical business
tasks. It records acceptance quality, elapsed time, token use, model/tool calls,
and collaboration evidence in isolated workspaces.

### Skills (Slash Commands)

Skills are markdown-based slash commands stored under `.clawd/skills`. Each skill lives in its own directory and must be named `SKILL.md`.

**1) Create a project skill**

Create:

```text
<project-root>/.clawd/skills/<skill-name>/SKILL.md
```

Example:

```md
---
description: Explains code with diagrams and analogies
when_to_use: Use when explaining how code works
allowed-tools:
  - Read
  - Grep
  - Glob
arguments: [path]
---

Explain the code in $path. Start with an analogy, then draw a diagram.
```

**2) Use it in the REPL**

```text
❯ /
❯ /<skill-name> <args>
```

Example:

```text
❯ /explain-code qsort.py
```

**Notes**

- User-level skills: `~/.clawd/skills/<skill-name>/SKILL.md`
- Tool limits: `allowed-tools` controls which tools the skill can use.
- Arguments: use `$ARGUMENTS`, `$0`, `$1`, or named args like `$path` (from `arguments`).
- Placeholder syntax: use `$path`, not `${path}`.



***

## 🎓 Why Clawd Codex?

### Based on Real Source Code

- **Not a clone** — Ported from actual TypeScript implementation
- **Architectural fidelity** — Maintains proven design patterns
- **Improvements** — Better error handling, more tests, cleaner code

### Python Native

- **Type hints** — Full type annotations
- **Modern Python** — Uses 3.10+ features
- **Idiomatic** — Clean, Pythonic code

### User Focused

- **3-step setup** — Clone, configure, run
- **Interactive config** — `clawd login` guides you
- **Rich REPL** — Tab completion, syntax highlighting
- **Session persistence** — Never lose your work

***

## 📦 Project Structure

```text
Clawd-Code/
├── src/
│   ├── cli.py           # CLI entry
│   ├── providers/       # LLM providers
│   ├── repl/            # Interactive REPL
│   ├── skills/          # SKILL.md loading and creation
│   └── tool_system/     # Tool registry, loop, validation
├── tests/               # Core test suite
├── .clawd/
│   └── skills/          # Project-local custom skills
└── FEATURE_LIST.md      # Current feature status
```

***


## 🤝 Contributing

**We welcome contributions!**

```bash
# Quick dev setup
pip install -e .[dev]
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

***

## 📖 Documentation

- **[SETUP_GUIDE.md](docs/guide/SETUP_GUIDE.md)** — Detailed installation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Development guide
- **[TESTING.md](docs/guide/TESTING.md)** — Testing guide
- **[CHANGELOG.md](CHANGELOG.md)** — Version history

***

## ⚡ Performance

- **Startup**: < 1 second
- **Memory**: < 50MB
- **Response**: Turn-based assistant output with Rich markdown rendering

***

## 🔒 Security

✅ **Basic Local Safety Practices**

- No sensitive data in Git
- API keys obfuscated in config
- `.env` files ignored
- Safe for local development workflows

***

## 📄 License

MIT License — See [LICENSE](LICENSE)

***

## 🙏 Acknowledgments

- Based on Claude Code TypeScript source
- Independent educational project
- Not affiliated with Anthropic

***

<div align="center">

### 🌟 Show Your Support

If you find this useful, please **star** ⭐ the repo!

**Made with ❤️ by Clawd Code Team**

[⬆ Back to Top](#-clawd-codex)

</div>

***

***

# 中文版

<div align="center">

[English](#-clawd-codex) | **中文** | [Français](docs/i18n/README_FR.md) | [Русский](docs/i18n/README_RU.md) | [हिन्दी](docs/i18n/README_HI.md) | [العربية](docs/i18n/README_AR.md) | [Português](docs/i18n/README_PT.md)

# 🚀 Claude Code Python

**基于真实 Claude Code 源码的完整 Python 重实现**

*从 TypeScript 源码 → 用 Python 重建 ❤️*

***

[![GitHub stars](https://img.shields.io/github/stars/GPT-AGI/Clawd-Code?style=for-the-badge&logo=github&color=yellow)](https://github.com/GPT-AGI/Clawd-Code/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GPT-AGI/Clawd-Code?style=for-the-badge&logo=github&color=blue)](https://github.com/GPT-AGI/Clawd-Code/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

**🔥 活跃开发中 • 每周更新新功能 🔥**

## FLEXIBLE SKILL SYSTEMS

**基于 Markdown 的斜杠技能系统，支持参数替换、工具限制，以及项目级 / 用户级技能加载。**

</div>

***

## 🎯 为什么是 Clawd Code？

**Clawd Code** 是一个面向真实使用的 **Claude Code Python 重构版**：它基于**真实 TypeScript 架构**移植而来，并且交付的是一个**可运行的 CLI Agent**，而不只是源码镜像。

- **真实 Agent Runtime** — 具备工具调用循环、流式 REPL、会话历史与多轮执行能力
- **高保真移植** — 尽可能保留 Claude Code 的原始架构，同时做符合 Python 风格的实现
- **适合继续开发** — 代码可读、测试完善，并支持基于 Markdown 的技能扩展

<div align="center">

**Token Streaming + Tool-Aware Agent Loop**

![流式 Agent 演示](assets/clawd-stream.gif)

**可编程 Skill Runtime 与工具沙箱**

![Skills（斜杠命令）](assets/clawd-code-skill.png)

**Instant Web Fetch for External Context**

![网页获取](assets/claude-code-webfetch.png)

**真实的 CLI • 真实的使用 • 真实的社区**

</div>

**这是一个真正可跑的 Claude Code 风格 Python 终端工作流：能流式回答、调工具、抓外部上下文，并通过 skills 扩展行为。**

**🚀 立即试用！Fork 它、修改它、让它成为你的！欢迎提交 Pull Request！**

***

## ⭐ Star 历史

<a href="https://www.star-history.com/?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=GPT-AGI%2FClawd-Code&type=date&legend=top-left" />
 </picture>
</a>

## ✨ 特性

### Streaming Agent Experience

```text
>>> /stream on
>>> 解释 tests/test_agent_loop.py
[流式回答中...]
• Read (tests/test_agent_loop.py) running...
  ↳ lines 1-180
>>> /render-last
```

- 直接回答支持真实 API 流式输出，带工具的 agent loop 也具备更完整的流式体验
- 内置 `/stream` 开关用于实时输出，`/render-last` 可按需把上一条回答重新渲染为 Markdown
- 专门为终端演示优化：一边看回答流出，一边看到工具调用，并保留稳定回退路径

### 可编程 Skill Runtime

```md
---
description: 用类比 + 图示解释代码
allowed-tools:
  - Read
  - Grep
  - Glob
arguments: [path]
---

请解释 $path 的实现：先给一个类比，再画一个结构示意图。
```

- 基于 `SKILL.md` 的 Markdown 斜杠命令
- 支持项目级技能、用户级技能、命名参数替换与工具限制

### 多提供商支持

```python
providers = ["Anthropic Claude", "OpenAI GPT", "Zhipu GLM"]  # + 易于扩展
```

### 交互式 REPL

```text
>>> 你好！
Assistant: 嗨！我是 Clawd Codex，一个 Python 重实现...

>>> /help         # 显示命令
>>> /             # 显示命令与技能
>>> /save         # 保存会话
>>> /multiline    # 多行输入模式
>>> Tab           # 自动补全
>>> /explain-code qsort.py   # 运行一个技能
```

### 完整的 CLI

```bash
clawd              # 启动 REPL
clawd login        # 配置 API
clawd --version    # 检查版本
clawd config       # 查看设置
clawd config --use glm5      # 切换到 Z.ai GLM-5.2
clawd config --use qwen3.5   # 切换到腾讯 TI-ONE Qwen 3.5
```

***

## 📊 状态

| 组件    | 状态     | 数量     |
| ----- | ------ | ------ |
| REPL 命令 | ✅ 完成   | 6+ 内置命令 |
| 工具系统 | ✅ 完成   | 30+ 工具 |
| 自动化测试 | ✅ 已覆盖  | Skills、providers、REPL、tools、context |
| 文档    | ✅ 完成   | 10+ 文档 |

### 核心系统

| 系统 | 状态 | 描述 |
|------|------|------|
| CLI 入口 | ✅ | `clawd`、`login`、`config`、`--version` |
| 交互式 REPL | ✅ | 丰富的交互输出、历史记录、Tab 补全、多行输入 |
| 多提供商支持 | ✅ | 支持 Anthropic、OpenAI、GLM、Qwen/TI-ONE |
| 会话持久化 | ✅ | 本地保存/加载会话 |
| Agent Loop | ✅ | 工具调用循环实现 |
| Skill 系统 | ✅ | 基于 SKILL.md 的 /skill 技能：参数替换 + 工具限制 |
| 上下文构建 | 🟡 | 已接入 workspace、git、CLAUDE.md 的基础上下文注入，仍需补强项目级理解 |
| 权限系统 | 🟡 | 框架已存在，需要集成 |

### 工具系统（已实现 30+ 工具）

| 类别 | 工具 | 状态 |
|------|------|------|
| 文件操作 | Read, Write, Edit, Glob, Grep | ✅ 完成 |
| 系统 | Bash 执行 | ✅ 完成 |
| 网络 | WebFetch, WebSearch | ✅ 完成 |
| 交互 | AskUserQuestion, SendMessage | ✅ 完成 |
| 任务管理 | TodoWrite, TaskManager, TaskStop | ✅ 完成 |
| Agent 工具 | Agent, Brief, Team | ✅ 完成 |
| 配置 | Config, PlanMode, Cron | ✅ 完成 |
| MCP | MCP 工具和资源 | ✅ 完成 |
| 其他 | LSP, Worktree, Skill（SKILL.md）, ToolSearch | ✅ 完成 |

### 路线图进度

- ✅ **阶段 0**：可安装、可运行的 CLI
- ✅ **阶段 1**：Claude Code 核心 MVP 体验
- ✅ **阶段 2**：真实工具调用闭环
- 🟡 **阶段 3**：上下文、权限、恢复能力（进行中）
- ⏳ **阶段 4**：MCP、插件、扩展性
- ⏳ **阶段 5**：Python 原生差异化特性

**详细功能状态和 PR 指南请查看 [FEATURE_LIST.md](FEATURE_LIST.md)。**

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/GPT-AGI/Clawd-Code.git
cd Clawd-Code

# 创建虚拟环境（推荐使用 uv）
uv venv --python 3.11
source .venv/bin/activate

# 安装
uv pip install -r requirements.txt
```

### 配置

#### 方式 1：交互式（推荐）

```bash
python -m src.cli login
```

这个流程会：

1. 让你选择 provider：anthropic / openai / glm
2. 让你输入该 provider 的 API key
3. 可选：保存自定义 base URL
4. 可选：保存默认 model
5. 将该 provider 设为默认

两个评测模型可以直接切换，不需要重复录入已有密钥：

```bash
clawd config --use glm5
clawd config --use qwen3.5
```

Qwen profile 使用 TI-ONE 服务组 ID `ms-mnhdj86z` 作为模型 ID，Base URL
以 `/ms-mnhdj86z/v1` 结尾。请通过 `clawd login` 或 `QWEN_API_KEY` 配置服务
AuthToken；密钥只保存在用户配置中，不会写入仓库。

配置文件会保存在 `~/.clawd/config.json`。示例结构：

```json
{
  "default_provider": "glm",
  "providers": {
    "anthropic": {
      "api_key": "base64-encoded-key",
      "base_url": "https://api.anthropic.com",
      "default_model": "claude-sonnet-4-20250514"
    },
    "openai": {
      "api_key": "base64-encoded-key",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4"
    },
    "glm": {
      "api_key": "base64-encoded-key",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "default_model": "glm-4.5"
    }
  }
}
```

### 运行

```bash
python -m src.cli          # 启动 REPL
python -m src.cli --help   # 显示帮助
clawd run -C ./project --prompt-file TASK.md  # 非交互执行单次任务
clawd trace ./project      # 在本地查看 teammate 运行轨迹
clawd peer run --repo ./project --prompt-file TASK.md --peers 2 --communication p2p --workspace-mode worktree
```

`clawd run` 也支持直接传入 prompt 或从 stdin 读取。本地文件操作范围限制在指定
workspace 内，执行进度输出到 stderr，最终回答输出到 stdout，便于脚本和 CI 使用。
Provider 凭据既可来自 `clawd login`，也可使用 `ANTHROPIC_AUTH_TOKEN`、
`ANTHROPIC_BASE_URL`、`OPENAI_API_KEY` 等标准环境变量。
Peer-native 模式会创建没有 lead 和 owned task graph 的平等持久 session。五种实验
condition、scripted smoke、输出 schema 与真实模型命令见
`teammate-evals/peer-collaboration/README.md`。

**就这样！** 3 步开始与 AI 对话。

***

## 💡 使用

### REPL 命令

| 命令           | 描述      |
| ------------ | ------- |
| `/`          | 显示命令与技能 |
| `/help`      | 显示所有命令  |
| `/save`      | 保存会话    |
| `/load <id>` | 加载会话    |
| `/multiline` | 切换多行模式  |
| `/clear`     | 清空历史    |
| `/exit`      | 退出 REPL |

### Teammate 工作流

Lead 可以创建持久化 teammate、分配带依赖的任务，并限制并行度执行。

Lead 会先判断协作是否值得；完全不创建 team 也是正确结果。需要分工时，Lead 根据
任务自行决定任意角色、模型、工具权限、workspace、依赖和并行度，不要求固定的
planner/coder/reviewer 流水线。Teammate 可通过 `SendMessage` 直接相互通信，并用
`ReadMessages` 获取 peer 回复，因此通信拓扑由实际工作自然产生。

```text
TeamCreate -> TeammateCreate -> TaskCreate -> TeamRun
```

仓库生成任务建议在 `TeamCreate` 开启 `quality_gates`，使用严格链路：

```text
TeamCreate -> TeamConfigure -> 至少两个 TeammateCreate/TaskCreate -> TeamRun -> TeamVerify
```

严格模式要求至少两个可立即并行的 owner、明确且不重叠的 `ownedFiles`、每个任务的
`acceptanceChecks`，以及显式的接口提供/依赖关系。跨任务接口必须加入 `blockedBy`，
相关 owner 之间至少有一次直接消息确认。Worker 全部结束后 Team 会停在
`verification_required`；只有 `TeamVerify` 在新虚拟环境中完成安装、import smoke 和
integration 三段验证，Team 才能进入 `completed`。

创建类工具会返回结构化的 `next_required_actions`；创建 teammate 并不等于启动它。
只有 worker 已拥有任务且 Lead 调用 `TeamRun` 后才会执行。如果 Lead 在 active team
尚未收敛时尝试结束，agent loop 会返回生命周期警告，要求先运行、恢复或显式删除 team。

`TeamRun` 支持 `max_workers`、`max_batches`、`max_retries`、`lease_timeout_s`、
`timeout_s`、`token_budget` 和 `turn_budget`。`TeamResume` 用于恢复过期 lease 或失败/取消的团队，
`TaskRetry` 显式重试单个任务，`TeamCancel` 执行协作式取消。所有状态迁移、模型调用、
工具调用和消息交接都会持久化，可通过 `clawd trace` 查看。
设置 `max_batches` 可在执行限定批次后把控制权交还 Lead，供其检查进度、追加或重派
任务、调整依赖、停止或恢复 worker，然后继续运行。

只有 Lead 可以调用 `TeammateStop` 和 `TeammateResume`。`TeammateStop` 只停止一个
worker，不会取消整个团队；未完成任务可选择默认的 `task_policy: "requeue"`，或使用
`task_policy: "cancel"`。当前停止会在模型/工具调用的下一个安全边界生效，不会伪装成
能够强杀线程中的 HTTP 或 Bash 调用。

人类操作者也可以直接控制同一份持久化状态，无需额外消耗 Lead 模型回合。
完整生命周期与进程级 force-stop 迁移设计见
`docs/guide/TEAMMATE_WORKER_LIFECYCLE.md`：

```bash
clawd team list -C ./project
clawd team status -C ./project
clawd team stop coder --task-policy requeue -C ./project
clawd team resume-worker coder -C ./project
clawd team reassign implementation replacement-coder -C ./project
clawd team cancel --reason "operator request" -C ./project
clawd team resume --provider anthropic --model glm-5.2 -C ./project
```

在 `TeammateCreate` 中设置 `workspace_mode: "worktree"` 可启用 Git 隔离；配合
`auto_integrate: true`，成功改动会在隔离 worktree 中提交并 cherry-pick 回 lead 仓库。
需要检查新整合改动的下游 reviewer 应使用 shared workspace。稳定性评测位于
`teammate-evals/runtime-resilience/`。

`teammate-evals/solo-vs-team/` 还提供五场景真实模型 benchmark，在保持业务任务
一致的前提下比较单 agent 与 Lead 自主决策的 teammate 工作流，并记录验收质量、耗时、token、
模型/工具调用次数和协作证据。

### Skills（技能 / 斜杠命令）教程

技能是存放在 `.clawd/skills` 下的 Markdown 斜杠命令。每个技能对应一个目录，并且文件名固定为 `SKILL.md`。

**1）创建项目技能**

创建：

```text
<project-root>/.clawd/skills/<skill-name>/SKILL.md
```

示例：

```md
---
description: 用类比 + 图示解释代码
when_to_use: 当用户问“这段代码怎么工作？”时使用
allowed-tools:
  - Read
  - Grep
  - Glob
arguments: [path]
---

请解释 $path 的实现：先给一个类比，再画一个结构示意图。
```

**2）在 REPL 中使用**

```text
❯ /
❯ /<skill-name> <args>
```

示例：

```text
❯ /explain-code qsort.py
```

**补充说明**

- 用户级技能：`~/.clawd/skills/<skill-name>/SKILL.md`
- 工具限制：`allowed-tools` 用于限制技能允许调用的工具集合
- 参数替换：支持 `$ARGUMENTS`、`$0`、`$1`、以及命名参数（例如 `$path`，来自 `arguments`）
- 占位符写法：请使用 `$path`，不要写成 `${path}`


***

## 🎓 为什么选择 Clawd Codex？

### 基于真实源码

- **不是克隆** — 从真实的 TypeScript 实现移植而来
- **架构保真** — 保持经过验证的设计模式
- **持续改进** — 更好的错误处理、更多测试、更清晰的代码

### 原生 Python

- **类型提示** — 完整的类型注解
- **现代 Python** — 使用 3.10+ 特性
- **符合习惯** — 干净的 Python 风格代码

### 以用户为中心

- **3 步设置** — 克隆、配置、运行
- **交互式配置** — `clawd login` 引导你完成设置
- **丰富的 REPL** — Tab 补全、语法高亮
- **会话持久化** — 永不丢失你的工作

***

## 📦 项目结构

```text
Clawd-Code/
├── src/
│   ├── cli.py           # CLI 入口
│   ├── providers/       # LLM 提供商
│   ├── repl/            # 交互式 REPL
│   ├── skills/          # SKILL.md 加载与创建
│   └── tool_system/     # 工具注册、循环与校验
├── tests/               # 核心测试套件
├── .clawd/
│   └── skills/          # 项目级自定义技能
└── FEATURE_LIST.md      # 当前功能状态
```

***

## 🤝 贡献

**我们欢迎贡献！**

```bash
# 快速开发设置
pip install -e .[dev]
python -m pytest tests/ -v
```

查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

***

## 📖 文档

- **[SETUP_GUIDE.md](docs/guide/SETUP_GUIDE.md)** — 详细安装说明
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — 开发指南
- **[TESTING.md](docs/guide/TESTING.md)** — 测试指南
- **[CHANGELOG.md](CHANGELOG.md)** — 版本历史

***

## ⚡ 性能

- **启动时间**：< 1 秒
- **内存占用**：< 50MB
- **响应**：回合式输出，支持 Rich Markdown 渲染

***

## 🔒 安全

✅ **基础本地安全实践**

- Git 中无敏感数据
- API 密钥在配置中做了基础混淆
- `.env` 文件被忽略
- 适合本地开发工作流

***

## 📄 许可证

MIT 许可证 — 查看 [LICENSE](LICENSE)

***

## 🙏 致谢

- 基于 Claude Code TypeScript 源码
- 独立的教育项目
- 未隶属于 Anthropic

***

<div align="center">

### 🌟 支持我们

如果你觉得这个项目有用，请给个 **star** ⭐！

**用 ❤️ 制作 by Clawd Code 团队**

[⬆ 回到顶部](#中文版)

</div>
