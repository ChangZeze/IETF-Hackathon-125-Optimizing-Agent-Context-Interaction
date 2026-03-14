
import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from agents import run_sub_agent
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


def build_task_context() -> Dict[str, Any]:
    """Build TaskContext - global task context"""
    return {
        "TaskID": "task_001",
        "UserQuery": "Complete a comparative research report on NEV companies for investment decisions",
        "TaskName": "NEV Investment Analysis",
        "TaskDescription": "Analyze NIO, Li Auto, XPeng, BYD covering financial performance, market ratings, policy environment, risk assessment, and investment ranking",
        "GoalStatus": [
            {"Goal": goal["goal_id"], "Status": "pending"}
            for goal in GOALS
        ],
        "OverallStatus": "in_progress",
        "StartTime": datetime.now().isoformat()
    }


def build_agent_context(goal: Dict[str, Any], dep_agents: List[str], task_context: Dict[str, Any]) -> Dict[str, Any]:
    """Master Agent generates AgentContext for Invoked Agent via LLM"""

    goal_id = goal["goal_id"]
    agent_name = goal["agent"]
    task_desc = goal["task_description"]
    dependencies = goal["dependencies"]

    # Get key information from dependency tasks (not full output)
    dep_info = []
    for dep_id in dependencies:
        if dep_id in state.get("agent_contexts", {}):
            dep_ctx = state["agent_contexts"][dep_id]
            key_info = dep_ctx.get("KeyInformation", [])
            dep_info.append({"goal": dep_id, "key_info": key_info})

    prompt = f"""You are the Master Agent's task decomposition module. Generate AgentContext for the sub-agent.

TaskContext:
{json.dumps(task_context, ensure_ascii=False, indent=2)}

Current Goal:
- Goal ID: {goal_id}
- Agent: {agent_name}
- Task Description: {task_desc}
- Dependencies: {dependencies}

Dependency Key Information:
{json.dumps(dep_info, ensure_ascii=False, indent=2) if dep_info else "None"}

Generate AgentContext in strict JSON format:
{{
    "AgentID": "{agent_name}",
    "AgentName": "{agent_name}",
    "SubTaskID": "{goal_id}",
    "SubTaskName": "Brief task name",
    "Dependencies": {json.dumps(dependencies)},
    "Context/ContextURI": "Relevant context summary from dependencies",
    "todoItems": [
        {{"itemId": "item1", "description": "Specific actionable task 1"}},
        {{"itemId": "item2", "description": "Specific actionable task 2"}}
    ],
    "ItemstateUpdates": [],
    "KeyInformation": [],
    "LastUpdated": ""
}}

Requirements:
1. Break down task_description into 2-4 specific todoItems
2. Each todoItem must be concrete and actionable
3. Context/ContextURI should summarize relevant info from dependencies
4. Output ONLY the JSON object, no extra text"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    add_usage(getattr(response, "usage", None), agent_type="master")

    raw = response.choices[0].message.content.strip()

    # Use JSON extraction function from agents.py
    from agents import _extract_json_object
    agent_context = _extract_json_object(raw)

    if not agent_context:
        print(f"[warning] Failed to parse AgentContext, raw response: {raw[:200]}")
        agent_context = {}

    return agent_context







state: Dict[str, Any] = {
    "task_results": {},
    "agent_contexts": {},
    "current_agent": "",
    "completed_goals": set(),
    "running_goals": set(),  # 正在运行的任务
}
state_lock = threading.Lock()

def execute_goal(goal: Dict[str, Any], task_context: Dict[str, Any]) -> None:
    """Execute a single goal (thread-safe)"""
    goal_id = goal["goal_id"]
    agent_name = goal["agent"]

    print(f"\n[master] Executing {goal_id} ({agent_name})")

    # Generate AgentContext
    agent_context = build_agent_context(goal, goal["dependencies"], task_context)

    print(f"[debug] Generated AgentContext for {goal_id}:")
    print(f"  todoItems: {agent_context.get('todoItems', [])}")
    print(f"  SubTaskName: {agent_context.get('SubTaskName', '')}")

    # Execute task (with retry)
    retry_count = 0
    feedback = ""

    while True:
        result = run_sub_agent(state, agent_context, feedback)
        updated_context = result["agent_context"]

        if not ENABLE_EVAL:
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["task_results"][goal_id] = result.get("full_output", "")
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)
            break

        eval_result = evaluate_by_master(updated_context, retry_count)
        decision = eval_result["decision"]

        print(f"[master-eval] {goal_id} decision={decision}, retry={retry_count}")

        if decision in {"pass", "force_pass"}:
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["task_results"][goal_id] = result.get("full_output", "")
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)

                for gs in task_context["GoalStatus"]:
                    if gs["Goal"] == goal_id:
                        gs["Status"] = "completed"
                        break
            break

        elif decision == "retry":
            retry_count = eval_result["retry_count"]
            feedback = eval_result.get("feedback", "")
        else:
            with state_lock:
                state["agent_contexts"][goal_id] = updated_context
                state["completed_goals"].add(goal_id)
                state["running_goals"].discard(goal_id)
            break

task_context = build_task_context()
print(f"[run] ENABLE_EVAL={ENABLE_EVAL}")
reset_run_stats()

# Dynamic parallel scheduling: continuously check and start ready tasks
active_threads: Dict[str, threading.Thread] = {}

while len(state["completed_goals"]) < len(GOALS):
    # Find all ready and not-running tasks
    with state_lock:
        ready_goals = []
        for goal in GOALS:
            goal_id = goal["goal_id"]
            if goal_id in state["completed_goals"] or goal_id in state["running_goals"]:
                continue

            deps_satisfied = all(
                dep in state["completed_goals"]
                for dep in goal["dependencies"]
            )

            if deps_satisfied:
                ready_goals.append(goal)
                state["running_goals"].add(goal_id)

    # Start new ready tasks
    for goal in ready_goals:
        goal_id = goal["goal_id"]
        print(f"[master] Starting {goal_id}")
        thread = threading.Thread(target=execute_goal, args=(goal, task_context))
        thread.start()
        active_threads[goal_id] = thread

    # Clean up completed threads
    completed_threads = []
    for goal_id, thread in list(active_threads.items()):
        if not thread.is_alive():
            thread.join()
            completed_threads.append(goal_id)

    for goal_id in completed_threads:
        del active_threads[goal_id]

    # If no active threads and no new tasks, exit
    if not active_threads and not ready_goals:
        if len(state["completed_goals"]) < len(GOALS):
            print("[error] No active threads and no ready goals, possible deadlock")
        break

    # Short sleep to avoid busy waiting
    time.sleep(0.1)

# Wait for all remaining threads to complete
for thread in active_threads.values():
    thread.join()

print(f"[master] All tasks completed: {len(state['completed_goals'])}/{len(GOALS)}")


# 完成所有任务
task_context["OverallStatus"] = "completed"
task_context["EndTime"] = datetime.now().isoformat()

finish_run_stats()
stats = get_run_stats()

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
elapsed = stats.get("elapsed_seconds")
print(f"elapsed_seconds: {elapsed:.2f}" if isinstance(elapsed, float) else "elapsed_seconds: N/A")

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
    f.write(f"elapsed_seconds: {elapsed_text}\n")

print(f"saved_report: {output_file}")