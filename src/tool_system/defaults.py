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
    ReadMessagesTool,
    REPLTool,
    ReadMcpResourceTool,
    RemoteTriggerTool,
    SendMessageTool,
    SendUserMessageTool,
    SkillTool,
    SleepTool,
    StructuredOutputTool,
    TeamAbortTool,
    TeamCreateTool,
    TeamConfigureTool,
    TeamCancelTool,
    TeamDeleteTool,
    TeamIntegrateTool,
    TeamPlanTool,
    TeamReplanTool,
    TeamResumeTool,
    TeammateCreateTool,
    TeammateResumeTool,
    TeammateStopTool,
    TeamRunTool,
    TeamVerifyTool,
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
from .remote_tools import (
    RemoteBashTool,
    RemoteFileEditTool,
    RemoteFileReadTool,
    RemoteFileWriteTool,
    RemoteGlobTool,
    RemoteGrepTool,
)


def build_default_registry(
    *,
    include_user_tools: bool = True,
    workspace_backend: object | None = None,
    include_team_tools: bool = True,
) -> ToolRegistry:
    workspace_tools = (
        [
            RemoteBashTool(),
            RemoteFileReadTool(),
            RemoteFileWriteTool(),
            RemoteFileEditTool(),
            RemoteGlobTool(),
            RemoteGrepTool(),
        ]
        if workspace_backend is not None
        else [
            BashTool(),
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            GlobTool(),
            GrepTool(),
        ]
    )
    collaboration_tools = (
        [
            TaskStopTool(),
            TaskCreateTool(),
            TaskGetTool(),
            TaskListTool(),
            TaskUpdateTool(),
            TaskOutputTool(),
            TaskRetryTool(),
            TeamCreateTool(),
            TeamConfigureTool(),
            TeamPlanTool(),
            TeammateCreateTool(),
            TeamRunTool(),
            TeamVerifyTool(),
            TeamReplanTool(),
            TeamResumeTool(),
            TeamCancelTool(),
            TeamAbortTool(),
            TeammateStopTool(),
            TeammateResumeTool(),
            TeamIntegrateTool(),
            TeamDeleteTool(),
            SendMessageTool(),
            ReadMessagesTool(),
            RemoteTriggerTool(),
        ]
        if include_team_tools
        else []
    )
    registry = ToolRegistry(
        tools=[
            SendUserMessageTool(),
            *workspace_tools,
            WebFetchTool(),
            WebSearchTool(),
            SleepTool(),
            ConfigTool(),
            MCPTool(),
            ListMcpResourcesTool(),
            ReadMcpResourceTool(),
            *([] if workspace_backend is not None else [LSPTool()]),
            SkillTool(),
            BriefTool(),
            AskUserQuestionTool(),
            TodoWriteTool(),
            *collaboration_tools,
            EnterPlanModeTool(),
            ExitPlanModeTool(),
            *(
                []
                if workspace_backend is not None
                else [EnterWorktreeTool(), ExitWorktreeTool()]
            ),
            CronCreateTool(),
            CronListTool(),
            CronDeleteTool(),
            StructuredOutputTool(),
            *(
                []
                if workspace_backend is not None
                else [PowerShellTool(), NotebookEditTool(), REPLTool()]
            ),
            TestingPermissionTool(),
        ]
    )
    if include_team_tools:
        registry.register(AgentTool(registry))
    registry.register(ToolSearchTool(registry))

    if include_user_tools:
        user_dir = Path.home() / ".clawd" / "tools"
        for tool in load_tools_from_dir(user_dir):
            registry.register(tool)

    return registry
