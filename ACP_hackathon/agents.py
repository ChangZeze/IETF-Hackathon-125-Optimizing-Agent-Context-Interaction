"""
Sub-agent module (ACP Structured Context Version)

Implements context-isolated agent invocation per IETF draft-chang-agent-context-interaction-01.
Key differences from baseline PlainText version:
- Context isolation: agents receive only KeyInformation summaries from dependencies, NOT full text
- Structured JSON input/output using AgentContext schema
- Two-level context delivery: summaries in prompt, full data only via ContextURI reference
- Thread-safe: no direct writes to shared state (caller handles locking)
"""

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
    """Extract a JSON object from potentially messy LLM output."""
    text = raw.strip()
    if not text:
        return {}

    # Try direct parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try fenced code block
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Try outermost braces
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


def _build_dependency_context_isolated(state: Dict[str, Any], dependencies: List[str]) -> str:
    """
    Build dependency context using ONLY KeyInformation summaries (context isolation).

    Per Section 4.2 and 5.2 of the IETF draft, each invoked agent receives only
    the context strictly relevant to its assigned task. We send KeyInformation
    abstracts from dependency agents, NOT their full raw outputs.

    This is the core ACP optimization that reduces token consumption.
    """
    if not dependencies:
        return "None"

    agent_contexts = state.get("agent_contexts", {})
    blocks: List[str] = []

    for dep in dependencies:
        dep_ctx = agent_contexts.get(dep, {})
        key_info = dep_ctx.get("KeyInformation", [])

        if key_info:
            # Send only the concise output abstracts (50-100 words each)
            summaries = []
            for ki in key_info:
                item_id = ki.get("itemId", "?")
                abstract = ki.get("outputabstract", "").strip()
                if abstract:
                    summaries.append(f"  - [{item_id}]: {abstract}")
            if summaries:
                dep_agent = dep_ctx.get("AgentName", dep)
                blocks.append(f"[{dep} ({dep_agent}) - Key Findings]\n" + "\n".join(summaries))
        else:
            # Fallback: if no KeyInformation yet, use a truncated summary
            full_output = state.get("task_results", {}).get(dep, "")
            if full_output:
                # Limit to ~300 chars to avoid full-text leakage
                truncated = full_output[:300].strip()
                if len(full_output) > 300:
                    truncated += "... [truncated]"
                blocks.append(f"[{dep} - Summary]\n{truncated}")

    return "\n\n".join(blocks) if blocks else "None"


def call_llm_stream_full_output(prompt: str, agent_name: str) -> str:
    """Call LLM with streaming and collect complete output."""
    print(f"\n===== {agent_name} stream =====")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are {agent_name}, a financial analysis sub-agent. "
                    "You must complete the assigned tasks and return results in strict JSON format. "
                    "Only output the final content that meets requirements."
                ),
            },
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
    """
    Normalize AgentContext returned by agent.

    Merges the LLM-parsed output with the base context, ensuring all
    required AgentContext fields are present. Falls back to base_context
    values for any missing fields.
    """
    normalized = {
        "AgentID": parsed.get("AgentID", base_context.get("AgentID", "")),
        "AgentName": parsed.get("AgentName", base_context.get("AgentName", "")),
        "SubTaskID": parsed.get("SubTaskID", base_context.get("SubTaskID", "")),
        "SubTaskName": parsed.get("SubTaskName", base_context.get("SubTaskName", "")),
        "Dependencies": parsed.get("Dependencies", base_context.get("Dependencies", [])),
        "ContextURI": parsed.get("Context/ContextURI", parsed.get("ContextURI", "")),
        "todoItems": parsed.get("todoItems", base_context.get("todoItems", [])),
        "ItemstateUpdates": parsed.get("ItemstateUpdates", []),
        "KeyInformation": parsed.get("KeyInformation", []),
        "LastUpdated": parsed.get("LastUpdated", datetime.now().isoformat()),
        "full_output": parsed.get("full_output", ""),
    }

    # Ensure ItemstateUpdates covers all todoItems (mark missing as 0)
    reported_items = {u.get("itemId") for u in normalized["ItemstateUpdates"]}
    for todo in normalized["todoItems"]:
        if todo.get("itemId") not in reported_items:
            normalized["ItemstateUpdates"].append({"itemId": todo["itemId"], "state": 0})

    return normalized


def run_sub_agent(state: Dict[str, Any], agent_context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
    """
    Execute a sub-agent with isolated context delivery.

    Per the ACI protocol (Section 5.3), the Master Agent sends only
    task-relevant information (todoItems + KeyInformation summaries from
    deps) to the Invoked Agent. The agent returns an updated AgentContext
    with ItemstateUpdates and KeyInformation.

    NOTE: This function does NOT write to shared state under lock.
    The caller (execute_goal) is responsible for thread-safe state updates.
    """
    agent_name = agent_context.get("AgentName", "sub_agent")

    capability = SUB_AGENT_CAPABILITIES.get(agent_name, "General Analysis and Output")
    dependencies = agent_context.get("Dependencies", [])

    # CONTEXT ISOLATION: send only KeyInformation summaries, not full outputs
    dep_context = _build_dependency_context_isolated(state, dependencies)

    todo_items = agent_context.get("todoItems", [])
    todo_text = "\n".join(
        [f"- {item.get('itemId', '')}: {item.get('description', '')}" for item in todo_items]
    ) or "- None"

    prompt = f"""You are {agent_name}. Capability: {capability}

Your assigned tasks (todoItems):
{todo_text}

Key findings from dependency agents (summarized):
{dep_context}

{f"CORRECTION REQUIRED - address these issues: {feedback}" if feedback else ""}

Instructions:
1. Complete EVERY task in todoItems thoroughly with real analysis
2. For data collection: gather key financial data for NIO, Li Auto, XPeng, BYD
3. For analysis tasks: provide detailed analysis based on the dependency findings above
4. For each completed item, provide a concise key finding summary (50-100 words)
5. Put your complete detailed work in the full_output field (200-500 words)

Return strict JSON only (no markdown, no code blocks):
{{
    "AgentID": "{agent_name}",
    "AgentName": "{agent_name}",
    "SubTaskID": "{agent_context.get('SubTaskID', '')}",
    "SubTaskName": "{agent_context.get('SubTaskName', '')}",
    "Dependencies": {json.dumps(dependencies)},
    "Context/ContextURI": "",
    "todoItems": {json.dumps(todo_items, ensure_ascii=False)},
    "ItemstateUpdates": [
        {{"itemId": "<id>", "state": 1}}
    ],
    "KeyInformation": [
        {{"itemId": "<id>", "outputabstract": "Concise key finding (50-100 words)"}}
    ],
    "LastUpdated": "{datetime.now().isoformat()}",
    "full_output": "Your complete detailed analysis (200-500 words)"
}}"""

    raw_output = call_llm_stream_full_output(prompt, agent_name)
    parsed = _extract_json_object(raw_output)
    updated_context = _normalize_agent_context(parsed, agent_context)

    if not updated_context.get("full_output"):
        updated_context["full_output"] = raw_output

    # Return results without writing to shared state (caller handles thread safety)
    return {
        "agent_context": updated_context,
        "full_output": updated_context["full_output"],
        "raw": raw_output,
    }