from __future__ import annotations

from pathlib import Path

from .loader import load_tools_from_dir
from .registry import ToolRegistry
from .tools import (
    AskUserQuestionTool,
    BashTool,
    BriefTool,
    ConfigTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    LSPTool,
    ListMcpResourcesTool,
    MCPTool,
    NotebookEditTool,
    PowerShellTool,
    REPLTool,
    ReadMcpResourceTool,
    RemoteTriggerTool,
    SendMessageTool,
    SendUserMessageTool,
    SkillTool,
    SleepTool,
    StructuredOutputTool,
    TeamCreateTool,
    TeamCancelTool,
    TeamDeleteTool,
    TeamIntegrateTool,
    TeamResumeTool,
    TeammateCreateTool,
    TeammateResumeTool,
    TeammateStopTool,
    TeamRunTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskRetryTool,
    TaskStopTool,
    TaskUpdateTool,
    TestingPermissionTool,
    TodoWriteTool,
    WebFetchTool,
    WebSearchTool,
)
from .tools.agent import AgentTool
from .tools.tool_search import ToolSearchTool


def build_default_registry(*, include_user_tools: bool = True) -> ToolRegistry:
    registry = ToolRegistry(
        tools=[
            SendUserMessageTool(),
            BashTool(),
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            GlobTool(),
            GrepTool(),
            WebFetchTool(),
            WebSearchTool(),
            SleepTool(),
            TaskStopTool(),
            ConfigTool(),
            MCPTool(),
            ListMcpResourcesTool(),
            ReadMcpResourceTool(),
            LSPTool(),
            SkillTool(),
            BriefTool(),
            AskUserQuestionTool(),
            TodoWriteTool(),
            TaskCreateTool(),
            TaskGetTool(),
            TaskListTool(),
            TaskUpdateTool(),
            TaskOutputTool(),
            TaskRetryTool(),
            TeamCreateTool(),
            TeammateCreateTool(),
            TeamRunTool(),
            TeamResumeTool(),
            TeamCancelTool(),
            TeammateStopTool(),
            TeammateResumeTool(),
            TeamIntegrateTool(),
            TeamDeleteTool(),
            EnterPlanModeTool(),
            ExitPlanModeTool(),
            EnterWorktreeTool(),
            ExitWorktreeTool(),
            CronCreateTool(),
            CronListTool(),
            CronDeleteTool(),
            SendMessageTool(),
            StructuredOutputTool(),
            RemoteTriggerTool(),
            PowerShellTool(),
            NotebookEditTool(),
            REPLTool(),
            TestingPermissionTool(),
        ]
    )
    registry.register(AgentTool(registry))
    registry.register(ToolSearchTool(registry))

    if include_user_tools:
        user_dir = Path.home() / ".clawd" / "tools"
        for tool in load_tools_from_dir(user_dir):
            registry.register(tool)

    return registry
