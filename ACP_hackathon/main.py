"""
Main orchestration module (ACP Structured Context Version)

Implements the Master Agent orchestration per IETF draft-chang-agent-context-interaction-01.

Key ACP features implemented:
1. TaskContext: Global task state maintained exclusively by the Master Agent (Section 5.1)
2. AgentContext: Per-agent isolated context with todoItems/KeyInformation (Section 5.2)
3. Context Isolation: Agents receive only KeyInformation summaries from deps, not full text (Section 4.2)
4. Parallel Scheduling: Independent agents (sub1/sub5/sub6) run concurrently (AsynchronousContextInteraction)
5. Two-Phase Evaluation: Code check + LLM quality check (Section 5.2)
6. Dependency-based DAG scheduler with thread-safe state management

Differences from baseline PlainText version:
- Baseline sends ALL previous raw outputs to every agent (~3000+ tokens per dep)
- ACP sends only KeyInformation summaries (~100-200 tokens per dep)
- Baseline evaluates full output text; ACP uses structured two-phase evaluation
- Baseline runs sequentially; ACP runs independent goals in parallel
"""

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from agents import run_sub_agent, _extract_json_object
from config import client
from evaluator import evaluate_by_master
from metrics import add_usage, reset_run_stats, get_run_stats, finish_run_stats

ENABLE_EVAL = True
COMPANIES = ["NIO", "Li Auto", "XPeng", "BYD"]

GOALS: List[Dict[str, Any]] = [
    {
        "goal_id": "G1",
        "agent": "sub1",
        "task_description": "Collect key financial data of NIO, Li Auto, XPeng, BYD (revenue scale, gross margin, delivery volume, R&D investment, cash reserves), no analysis, structured output.",
        "dependencies": [],
    },
    {
        "goal_id": "G2",
        "agent": "sub2",
        "task_description": "Based on sub1 data, analyze profitability and gross margin structure of four companies, about 200 words per company.",
        "dependencies": ["G1"],
    },
    {
        "goal_id": "G3",
        "agent": "sub3",
        "task_description": "Based on sub1 data, analyze cost control and operational efficiency of four companies, about 200 words per company.",
        "dependencies": ["G1"],
    },
    {
        "goal_id": "G4",
        "agent": "sub4",
        "task_description": "Synthesize sub2 and sub3 outputs, form key conclusions of four companies, about 300 words per company.",
        "dependencies": ["G2", "G3"],
    },
    {
        "goal_id": "G5",
        "agent": "sub5",
        "task_description": "Collect broker ratings, stock price performance, expert strategic evaluations from 2024Q4 to January 2026, summarize by company structure, about 300 words per company.",
        "dependencies": [],
    },
    {
        "goal_id": "G6",
        "agent": "sub6",
        "task_description": "Collect new energy vehicle policy related news from 2024Q4 to January 2026, and summarize policy environment changes and industry impacts.",
        "dependencies": [],
    },
    {
        "goal_id": "G7",
        "agent": "sub7",
        "task_description": "Synthesize sub1, sub5, sub6, output risk assessment of four companies, about 200 words per company.",
        "dependencies": ["G1", "G5", "G6"],
    },
    {
        "goal_id": "G8",
        "agent": "sub8",
        "task_description": "Synthesize sub4 and sub7, provide investment recommendations and investment rankings of four companies, total words not less than 400.",
        "dependencies": ["G4", "G7"],
    },
]

# Build goal_id -> agent_name mapping for logging
GOAL_AGENT_MAP = {g["goal_id"]: g["agent"] for g in GOALS}


def build_task_context() -> Dict[str, Any]:
    """
    Build TaskContext per Section 5.1 of the IETF draft.

    The TaskContext is a structured state object created and maintained
    exclusively by the Master Agent throughout the task lifecycle.
    """
    return {
        "TaskID": "task_001",
        "UserQuery": "Complete a comparative research report on NEV companies for investment decisions",
        "TaskName": "NEV Investment Analysis",
        "TaskDescription": (
            "Analyze NIO, Li Auto, XPeng, BYD covering financial performance, "
            "market ratings, policy environment, risk assessment, and investment ranking"
        ),
        "GoalStatus": [
            {"Goal": goal["goal_id"], "Status": "pending"}
            for goal in GOALS
        ],
        "OverallStatus": "in_progress",
        "StartTime": datetime.now().isoformat(),
    }


def build_agent_context(
    goal: Dict[str, Any],
    state: Dict[str, Any],
    task_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Master Agent generates AgentContext for an Invoked Agent via LLM.

    Per Section 5.3 of the draft, the Master Agent generates an isolated
    AgentContext containing only task-relevant information. Dependency context
    is delivered as KeyInformation summaries, not full text.
    """
    goal_id = goal["goal_id"]
    agent_name = goal["agent"]
    task_desc = goal["task_description"]
    dependencies = goal["dependencies"]

    # Collect KeyInformation from completed dependency goals (context isolation)
    dep_info: List[Dict[str, Any]] = []
    for dep_id in dependencies:
        dep_ctx = state.get("agent_contexts", {}).get(dep_id, {})
        key_info = dep_ctx.get("KeyInformation", [])
        if key_info:
            dep_info.append({
                "goal": dep_id,
                "agent": GOAL_AGENT_MAP.get(dep_id, dep_id),
                "key_findings": key_info,
            })

    prompt = f"""You are the Master Agent's task decomposition module. Generate an AgentContext for the sub-agent.

Current Goal:
- Goal ID: {goal_id}
- Agent: {agent_name}
- Task: {task_desc}
- Dependencies: {json.dumps(dependencies)}

Dependency Key Findings (summarized, not full output):
{json.dumps(dep_info, ensure_ascii=False, indent=2) if dep_info else "None (no dependencies or independent task)"}

Generate AgentContext as strict JSON:
{{
    "AgentID": "{agent_name}",
    "AgentName": "{agent_name}",
    "SubTaskID": "{goal_id}",
    "SubTaskName": "<brief task name>",
    "Dependencies": {json.dumps(dependencies)},
    "Context/ContextURI": "<summary of relevant dependency context>",
    "todoItems": [
        {{"itemId": "item1", "description": "<specific actionable task>"}},
        {{"itemId": "item2", "description": "<specific actionable task>"}}
    ],
    "ItemstateUpdates": [],
    "KeyInformation": [],
    "LastUpdated": ""
}}

Requirements:
1. Break down the task into 2-4 specific, concrete todoItems
2. Each todoItem must be actionable by the sub-agent
3. Context/ContextURI should summarize key points from dependencies (NOT full text)
4. Output ONLY the JSON object"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    add_usage(getattr(response, "usage", None), agent_type="master")

    raw = (response.choices[0].message.content or "").strip()
    agent_context = _extract_json_object(raw)

    if not agent_context:
        print(f"[warning] Failed to parse AgentContext for {goal_id}, using fallback")
        # Fallback: construct a minimal valid AgentContext
        agent_context = {
            "AgentID": agent_name,
            "AgentName": agent_name,
            "SubTaskID": goal_id,
            "SubTaskName": task_desc[:50],
            "Dependencies": dependencies,
            "Context/ContextURI": "",
            "todoItems": [
                {"itemId": "item1", "description": task_desc},
            ],
            "ItemstateUpdates": [],
            "KeyInformation": [],
            "LastUpdated": "",
        }

    return agent_context


# --- Global shared state (thread-safe access via state_lock) ---

state: Dict[str, Any] = {
    "task_results": {},         # goal_id -> full_output text
    "agent_contexts": {},       # goal_id -> AgentContext dict
    "current_agent": "",
    "completed_goals": set(),
    "running_goals": set(),
}
state_lock = threading.Lock()


def execute_goal(goal: Dict[str, Any], task_context: Dict[str, Any]) -> None:
    """
    Execute a single goal with retry logic (thread-safe).

    This is the core execution loop per Section 5.3 of the IETF draft:
    1. Master generates AgentContext
    2. Send AgentContext to Invoked Agent (only todoItems + dep summaries)
    3. Receive updated AgentContext with ItemstateUpdates + KeyInformation
    4. Two-phase evaluation
    5. Update TaskContext on completion
    """
    goal_id = goal["goal_id"]
    agent_name = goal["agent"]

    print(f"\n{'='*60}")
    print(f"[master] Executing {goal_id} ({agent_name})")
    print(f"{'='*60}")

    # Read state snapshot for AgentContext generation
    with state_lock:
        state_snapshot = {
            "task_results": dict(state["task_results"]),
            "agent_contexts": dict(state["agent_contexts"]),
        }

    # Generate AgentContext (uses state_snapshot for dep info, no lock needed)
    agent_context = build_agent_context(goal, state_snapshot, task_context)

    print(f"[master] AgentContext for {goal_id}:")
    print(f"  SubTaskName: {agent_context.get('SubTaskName', '')}")
    print(f"  todoItems: {[t.get('itemId') for t in agent_context.get('todoItems', [])]}")
    print(f"  Dependencies: {agent_context.get('Dependencies', [])}")

    # Execute with retry loop
    retry_count = 0
    feedback = ""

    while True:
        # Run sub-agent (reads shared state for dep context, but writes are thread-safe below)
        result = run_sub_agent(state, agent_context, feedback)
        updated_context = result["agent_context"]
        full_output = result.get("full_output", "")

        if not ENABLE_EVAL:
            # Skip evaluation, store results immediately
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["task_results"][goal_id] = full_output
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)
            break

        # Two-phase evaluation
        eval_result = evaluate_by_master(updated_context, retry_count)
        decision = eval_result["decision"]

        print(f"[master-eval] {goal_id} ({agent_name}): decision={decision}, retry_count={retry_count}")

        if decision in {"pass", "force_pass"}:
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["task_results"][goal_id] = full_output
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)

                # Update GoalStatus in TaskContext
                for gs in task_context["GoalStatus"]:
                    if gs["Goal"] == goal_id:
                        gs["Status"] = "completed"
                        break
            print(f"[master] {goal_id} ({agent_name}) -> {decision}")
            break

        elif decision == "retry":
            retry_count = eval_result["retry_count"]
            feedback = eval_result.get("feedback", "")
            print(f"[master] {goal_id} ({agent_name}) -> retry #{retry_count}: {feedback[:100]}")

        else:
            # Unexpected decision, treat as force_pass
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["task_results"][goal_id] = full_output
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)
            break


# ========================== MAIN EXECUTION ==========================

if __name__ == "__main__" or True:  # Always run (matches baseline behavior)

    task_context = build_task_context()
    print(f"[run] ENABLE_EVAL={ENABLE_EVAL}")
    print(f"[run] TaskID={task_context['TaskID']}, Goals={len(GOALS)}")
    print(f"[run] Execution DAG:")
    print(f"  Level 0 (parallel): G1, G5, G6  (independent)")
    print(f"  Level 1 (parallel): G2, G3      (depend on G1)")
    print(f"  Level 2 (parallel): G4, G7      (depend on G2+G3, G1+G5+G6)")
    print(f"  Level 3:            G8           (depend on G4+G7)")
    reset_run_stats()

    # Dynamic parallel DAG scheduler
    active_threads: Dict[str, threading.Thread] = {}

    while len(state["completed_goals"]) < len(GOALS):
        # Identify ready goals (deps satisfied, not running/completed)
        with state_lock:
            ready_goals: List[Dict[str, Any]] = []
            for goal in GOALS:
                goal_id = goal["goal_id"]
                if goal_id in state["completed_goals"]:
                    continue
                if goal_id in state["running_goals"]:
                    continue
                deps_satisfied = all(
                    dep in state["completed_goals"]
                    for dep in goal["dependencies"]
                )
                if deps_satisfied:
                    ready_goals.append(goal)
                    state["running_goals"].add(goal_id)

        # Launch ready goals as threads (AsynchronousContextInteraction for independents)
        for goal in ready_goals:
            goal_id = goal["goal_id"]
            agent_name = goal["agent"]
            print(f"[scheduler] Launching {goal_id} ({agent_name})")
            thread = threading.Thread(
                target=execute_goal,
                args=(goal, task_context),
                name=f"goal-{goal_id}",
            )
            thread.start()
            active_threads[goal_id] = thread

        # Clean up completed threads
        completed_thread_ids: List[str] = []
        for gid, thread in list(active_threads.items()):
            if not thread.is_alive():
                thread.join()
                completed_thread_ids.append(gid)
        for gid in completed_thread_ids:
            del active_threads[gid]

        # Deadlock detection
        if not active_threads and not ready_goals:
            if len(state["completed_goals"]) < len(GOALS):
                missing = set(g["goal_id"] for g in GOALS) - state["completed_goals"]
                print(f"[error] Deadlock detected! Missing goals: {missing}")
            break

        time.sleep(0.1)

    # Wait for stragglers
    for thread in active_threads.values():
        thread.join()

    # Finalize TaskContext
    task_context["OverallStatus"] = "completed"
    task_context["EndTime"] = datetime.now().isoformat()

    finish_run_stats()
    stats = get_run_stats()

    completed = len(state["completed_goals"])
    total = len(GOALS)
    print(f"\n[master] All tasks completed: {completed}/{total}")

    # ========================== OUTPUT ==========================

    sub8_output = state["task_results"].get("G8", "")
    if sub8_output:
        print("\n===== final sub8 output =====")
        print(sub8_output)

    print("\n===== task context =====")
    print(json.dumps(task_context, ensure_ascii=False, indent=2))

    print("\n===== run metrics =====")
    print(f"prompt_tokens: {stats['prompt_tokens']}")
    print(f"completion_tokens: {stats['completion_tokens']}")
    print(f"total_tokens: {stats['total_tokens']}")
    print(f"  master_tokens: {stats['master_total_tokens']}")
    print(f"  subagent_tokens: {stats['subagent_total_tokens']}")
    elapsed = stats.get("elapsed_seconds")
    print(f"elapsed_seconds: {elapsed:.2f}" if isinstance(elapsed, float) else "elapsed_seconds: N/A")

    # Save report (format matches generate_dashboard.py expectations)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{timestamp}.txt"
    elapsed_text = f"{elapsed:.2f}" if isinstance(elapsed, float) else "N/A"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== final sub8 output =====\n")
        f.write(sub8_output if sub8_output else "(empty)\n")
        f.write("\n===== task context =====\n")
        f.write(json.dumps(task_context, ensure_ascii=False, indent=2) + "\n")
        f.write("\n===== run metrics =====\n")
        f.write(f"prompt_tokens: {stats['prompt_tokens']}\n")
        f.write(f"completion_tokens: {stats['completion_tokens']}\n")
        f.write(f"total_tokens: {stats['total_tokens']}\n")
        f.write(f"master_total_tokens: {stats['master_total_tokens']}\n")
        f.write(f"subagent_total_tokens: {stats['subagent_total_tokens']}\n")
        f.write(f"elapsed_seconds: {elapsed_text}\n")

    print(f"saved_report: {output_file}")