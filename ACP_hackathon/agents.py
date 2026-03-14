import json
import re
from datetime import datetime
from typing import Any, Dict, List

from config import client
from metrics import add_usage

MODEL = "deepseek-chat"

SUB_AGENT_CAPABILITIES = {
    "sub1": "Responsible for collecting latest key financial data of Chinese new energy vehicle companies, no analysis.",
    "sub2": "Responsible for profitability and gross margin structure analysis of new energy vehicle companies.",
    "sub3": "Responsible for cost control and operational efficiency analysis of new energy vehicle companies.",
    "sub4": "Responsible for integrating sub2 and sub3, output key conclusions.",
    "sub5": "Responsible for collecting broker ratings, stock price performance, expert opinions of new energy vehicle companies.",
    "sub6": "Responsible for collecting new energy vehicle policy information and summarizing impacts.",
    "sub7": "Responsible for comprehensive risk assessment based on financial, market, and policy of new energy vehicle companies.",
    "sub8": "Responsible for synthesizing conclusions and risks, providing investment recommendations and rankings.",
}


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return {}
    return {}


def _build_dependency_context(state: Dict[str, Any], dependencies: List[str]) -> str:
    if not dependencies:
        return "None"

    task_results = state.get("task_results", {})
    agent_contexts = state.get("agent_contexts", {})
    blocks = []
    for dep in dependencies:
        full_output = str(task_results.get(dep, "")).strip()
        if not full_output:
            ctx = agent_contexts.get(dep, {})
            full_output = str(ctx.get("full_output", "")).strip()
        if full_output:
            blocks.append(f"[{dep} full output]\n{full_output}")
    return "\n\n".join(blocks) if blocks else "None"





def call_llm_stream_full_output(prompt: str, agent_name: str) -> str:
    print(f"\n===== {agent_name} stream =====")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are {agent_name}, a financial analysis expert. You must complete the assigned tasks and return results in strict JSON format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    parts: List[str] = []
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            add_usage(usage, agent_type="subagent")

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta else None
        if text:
            print(text, end="", flush=True)
            parts.append(text)

    print(f"\n===== {agent_name} stream end =====\n")
    return "".join(parts).strip()

def _normalize_agent_context(parsed: Dict[str, Any], base_context: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize AgentContext returned by agent"""
    # Merge base context and parsed result
    normalized = {
        "AgentID": parsed.get("AgentID", base_context.get("AgentID", "")),
        "AgentName": parsed.get("AgentName", base_context.get("AgentName", "")),
        "SubTaskID": parsed.get("SubTaskID", base_context.get("SubTaskID", "")),
        "SubTaskName": parsed.get("SubTaskName", base_context.get("SubTaskName", "")),
        "Dependencies": parsed.get("Dependencies", base_context.get("Dependencies", [])),
        "Context/ContextURI": parsed.get("Context/ContextURI", ""),
        "todoItems": parsed.get("todoItems", base_context.get("todoItems", [])),
        "ItemstateUpdates": parsed.get("ItemstateUpdates", []),
        "KeyInformation": parsed.get("KeyInformation", []),
        "LastUpdated": parsed.get("LastUpdated", ""),
        "full_output": parsed.get("full_output", "")
    }
    return normalized

def run_sub_agent(state: Dict[str, Any], agent_context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
    agent_name = agent_context.get("AgentName", "sub_agent")
    state["current_agent"] = agent_name

    capability = SUB_AGENT_CAPABILITIES.get(agent_name, "General Analysis and Output")
    dependencies = agent_context.get("Dependencies", [])
    dep_context = _build_dependency_context(state, dependencies)
    todo_items = agent_context.get("todoItems", [])
    todo_text = "\n".join(
        [f"- {item.get('itemId', '')}: {item.get('description', '')}" for item in todo_items]
    ) or "- None"

    prompt = f"""
You are {agent_name}. Capability: {capability}

Your assigned tasks (todoItems):
{todo_text}

Context from dependent agents:
{dep_context if dep_context else "None"}

{f"Evaluation feedback - you must address these issues: {feedback}" if feedback else ""}

IMPORTANT: You must ACTUALLY COMPLETE the tasks, not just return template text!

Instructions:
1. Complete each task in todoItems thoroughly
2. For data collection tasks: gather real financial data for NIO, Li Auto, XPeng, BYD
3. For analysis tasks: provide detailed analysis based on the data
4. Extract key findings for each completed item

Output format - strict JSON:
{{
    "AgentID": "{agent_name}",
    "AgentName": "{agent_name}",
    "SubTaskID": "{agent_context.get('SubTaskID', '')}",
    "SubTaskName": "{agent_context.get('SubTaskName', '')}",
    "Dependencies": {json.dumps(dependencies)},
    "Context/ContextURI": "",
    "todoItems": {json.dumps(todo_items, ensure_ascii=False)},
    "ItemstateUpdates": [
        // For each todoItem, set state: 1 if completed, 0 if not
        {{"itemId": "item1", "state": 1}}
    ],
    "KeyInformation": [
        // For each completed item, provide a brief summary (50-100 words)
        {{"itemId": "item1", "outputabstract": "Your actual key findings here"}}
    ],
    "LastUpdated": "{datetime.now().isoformat()}",
    "full_output": "Your complete detailed analysis here (200-500 words)"
}}

Output ONLY the JSON, no markdown code blocks."""


    raw_output = call_llm_stream_full_output(prompt, agent_name)
    parsed = _extract_json_object(raw_output)
    updated_context = _normalize_agent_context(parsed, agent_context)   
    if not updated_context.get("full_output"):
        updated_context["full_output"] = raw_output

    state["task_results"][agent_name] = updated_context["full_output"]
    state["agent_contexts"][agent_name] = updated_context
    return {
        "agent_context": updated_context,
        "full_output": updated_context["full_output"],
        "raw": raw_output,
    }