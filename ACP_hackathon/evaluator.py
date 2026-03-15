"""
Evaluation Module (ACP Structured Context Version)

Implements the two-phase evaluation mechanism per Section 5.2 of
IETF draft-chang-agent-context-interaction-01.

Phase 1 (Code-based, zero tokens):
  Inspect ItemstateUpdates — if any item has state=0, immediately
  return "retry" without invoking the LLM. This saves token overhead
  for obvious incomplete results.

Phase 2 (LLM-based, minimal tokens):
  For items with state=1, verify that KeyInformation.outputabstract
  entries are meaningful and satisfy the subtask intent. Only the
  structured eval payload (todoItems + ItemstateUpdates + KeyInformation)
  is sent — NOT the full output text.
"""

import json
from typing import Any, Dict, List, Optional

from config import client
from metrics import add_usage

MAX_RETRY = 2


def _build_eval_payload(agent_context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the structured fields needed for evaluation."""
    return {
        "SubTaskName": agent_context.get("SubTaskName", ""),
        "todoItems": agent_context.get("todoItems", []),
        "ItemstateUpdates": agent_context.get("ItemstateUpdates", []),
        "KeyInformation": agent_context.get("KeyInformation", []),
    }


def _phase1_check_completion(eval_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Phase 1: Code-based completeness check (zero LLM tokens).

    Returns a retry result if any todoItem is incomplete (state=0 or missing),
    or None if all items pass and Phase 2 should proceed.
    """
    todo_items = eval_payload.get("todoItems", [])
    item_updates = eval_payload.get("ItemstateUpdates", [])
    key_info = eval_payload.get("KeyInformation", [])

    if not todo_items:
        return {"decision": "retry", "feedback": "No todoItems found in agent output."}

    if not item_updates:
        return {"decision": "retry", "feedback": "No ItemstateUpdates returned — agent may not have completed the task."}

    # Build lookup maps
    state_map = {u.get("itemId"): u.get("state", 0) for u in item_updates}
    info_map = {k.get("itemId"): k.get("outputabstract", "") for k in key_info}

    incomplete_items: List[str] = []
    empty_info_items: List[str] = []

    for todo in todo_items:
        item_id = todo.get("itemId", "")
        # Check completion state
        if state_map.get(item_id, 0) != 1:
            incomplete_items.append(item_id)
        # Check that completed items have KeyInformation
        elif not info_map.get(item_id, "").strip():
            empty_info_items.append(item_id)

    if incomplete_items:
        return {
            "decision": "retry",
            "feedback": f"Items not completed (state=0): {incomplete_items}. Complete all assigned todoItems.",
        }

    if empty_info_items:
        return {
            "decision": "retry",
            "feedback": f"Items missing KeyInformation summaries: {empty_info_items}. Provide outputabstract for each completed item.",
        }

    # All items state=1 with KeyInformation present — proceed to Phase 2
    return None


def _phase2_quality_check(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: LLM-based quality verification.

    Sends only the structured eval payload (NOT full output) to verify
    that KeyInformation summaries meaningfully satisfy the subtask intent.
    This is the efficient two-layer evaluation from the IETF draft.
    """
    prompt = f"""You are the Master Agent's evaluation module. Evaluate sub-agent task quality based ONLY on the structured output below.

Evaluation Input:
{json.dumps(eval_payload, ensure_ascii=False, indent=2)}

Evaluation criteria:
1. Each KeyInformation.outputabstract should contain substantive findings (not placeholder/template text)
2. The summaries should be relevant to the SubTaskName
3. Quality bar: reasonable content that downstream agents can build upon

Output strict JSON only:
{{
  "decision": "pass" or "retry",
  "feedback": "If retry, give concise actionable corrections. If pass, empty string."
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    add_usage(getattr(response, "usage", None), agent_type="master")
    raw = (response.choices[0].message.content or "").strip()

    # Parse LLM response
    decision = "retry"
    feedback = ""
    try:
        result = json.loads(raw)
        decision = result.get("decision", "retry")
        feedback = result.get("feedback", "")
    except (json.JSONDecodeError, KeyError):
        # Fallback to string matching
        lower = raw.lower()
        if '"pass"' in lower or "'pass'" in lower:
            decision = "pass"
        feedback = raw

    if decision not in {"pass", "retry", "force_pass"}:
        decision = "retry"
        if not feedback:
            feedback = raw

    return {"decision": decision, "feedback": feedback, "raw_eval": raw}


def evaluate_by_master(agent_context: Dict[str, Any], retry_count: int) -> Dict[str, Any]:
    """
    Two-phase evaluation of sub-agent output per the IETF ACI draft.

    Phase 1: Code check on ItemstateUpdates (zero tokens, instant).
    Phase 2: LLM quality check on KeyInformation (minimal tokens).

    Returns dict with: decision, feedback, retry_count, raw_eval
    """
    eval_payload = _build_eval_payload(agent_context)

    # Phase 1: Code-based completeness check (no LLM call)
    phase1_result = _phase1_check_completion(eval_payload)
    if phase1_result is not None:
        decision = phase1_result["decision"]
        feedback = phase1_result["feedback"]
        print(f"  [eval-phase1] FAIL: {feedback}")
    else:
        # Phase 2: LLM-based quality check
        phase2_result = _phase2_quality_check(eval_payload)
        decision = phase2_result["decision"]
        feedback = phase2_result["feedback"]
        print(f"  [eval-phase2] decision={decision}")

    # Apply retry limits
    next_retry = retry_count
    if decision == "retry":
        next_retry = retry_count + 1
        if next_retry >= MAX_RETRY:
            decision = "force_pass"
            print(f"  [eval] Max retries ({MAX_RETRY}) reached, force_pass")

    if decision == "pass":
        next_retry = 0

    return {
        "decision": decision,
        "feedback": feedback,
        "retry_count": next_retry,
        "raw_eval": phase1_result["feedback"] if phase1_result else feedback,
    }