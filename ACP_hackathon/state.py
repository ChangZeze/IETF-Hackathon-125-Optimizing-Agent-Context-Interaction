"""
State Type Definitions Module (ACP Structured Context Version)

Defines structured context schemas aligned with IETF draft-chang-agent-context-interaction-01.

Schemas:
- TaskContext: Global task state maintained exclusively by the Master Agent.
  Tracks task identity, goal progress, and overall lifecycle.
- AgentContext: Per-agent execution state, strictly isolated per subtask.
  Contains todoItems, completion states, and key information abstracts.
- ACPState: Runtime state holding all active contexts and results.

Reference: Sections 5.1 and 5.2 of draft-chang-agent-context-interaction-01
"""

from typing import TypedDict, Dict, List, Any, Set


class GoalStatusEntry(TypedDict):
    """A single goal's tracking entry within TaskContext.GoalStatus"""
    Goal: str       # Goal identifier (e.g. "G1")
    Status: str     # "pending" | "in_progress" | "completed" | "failed"


class TaskContext(TypedDict, total=False):
    """
    TaskContext schema per Section 5.1 of the IETF draft.

    Created and maintained exclusively by the Master Agent throughout
    the lifecycle of a complex task. Provides a persistent and
    machine-interpretable representation of task progress.
    """
    TaskID: str
    UserQuery: str
    TaskName: str
    TaskDescription: str
    GoalStatus: List[GoalStatusEntry]
    OverallStatus: str      # "pending" | "in_progress" | "completed" | "failed"
    StartTime: str          # Optional per spec
    EndTime: str            # Optional per spec


class TodoItem(TypedDict):
    """A single actionable step assigned to an agent."""
    itemId: str
    description: str


class ItemStateUpdate(TypedDict):
    """Completion state for a single todoItem. 0 = not completed, 1 = completed."""
    itemId: str
    state: int  # 0 or 1


class KeyInformationEntry(TypedDict):
    """Concise output abstract for a completed todoItem."""
    itemId: str
    outputabstract: str


class AgentContext(TypedDict, total=False):
    """
    AgentContext schema per Section 5.2 of the IETF draft.

    Represents execution state specific to an individual agent.
    Strictly isolated between subtasks -- only the AgentContext for the
    target agent is delivered during invocation. Upon completion, the
    agent returns exclusively its own updated AgentContext.
    """
    AgentID: str
    AgentName: str
    SubTaskID: str
    SubTaskName: str
    Dependencies: List[str]
    ContextURI: str                         # Reference to external/long-term context
    todoItems: List['TodoItem']
    ItemstateUpdates: List['ItemStateUpdate']
    KeyInformation: List['KeyInformationEntry']
    LastUpdated: str                         # Optional per spec
    full_output: str                         # Internal: stores raw detailed output


class ACPState(TypedDict, total=False):
    """Runtime state for the ACP multi-agent execution."""
    task_results: Dict[str, str]            # goal_id -> full_output text
    agent_contexts: Dict[str, Any]          # goal_id -> AgentContext dict
    current_agent: str
    completed_goals: Set[str]
    running_goals: Set[str]