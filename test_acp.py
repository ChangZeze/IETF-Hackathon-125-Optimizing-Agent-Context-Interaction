"""Offline tests for ACP_hackathon modules (no API key needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ACP_hackathon'))
os.chdir(os.path.join(os.path.dirname(__file__), 'ACP_hackathon'))

passed = 0
failed = 0

def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} - {msg}")
        failed += 1

# === state.py ===
print("\n--- state.py ---")
from state import TaskContext, AgentContext, ACPState, GoalStatusEntry, TodoItem, ItemStateUpdate, KeyInformationEntry
check("All types import", True)

# === metrics.py ===
print("\n--- metrics.py ---")
from metrics import add_usage, reset_run_stats, finish_run_stats, get_run_stats
reset_run_stats()
add_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, agent_type="master")
add_usage({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}, agent_type="subagent")
stats = get_run_stats()
check("Master tokens", stats["master_total_tokens"] == 150, f"got {stats['master_total_tokens']}")
check("Subagent tokens", stats["subagent_total_tokens"] == 300, f"got {stats['subagent_total_tokens']}")
check("Total tokens", stats["total_tokens"] == 450, f"got {stats['total_tokens']}")
check("Elapsed tracked", stats["elapsed_seconds"] is not None and stats["elapsed_seconds"] > 0)

# === evaluator.py ===
print("\n--- evaluator.py ---")
from evaluator import _phase1_check_completion, _build_eval_payload

# All complete
ctx1 = {
    "SubTaskName": "Test",
    "todoItems": [{"itemId": "i1", "description": "test"}],
    "ItemstateUpdates": [{"itemId": "i1", "state": 1}],
    "KeyInformation": [{"itemId": "i1", "outputabstract": "Some findings"}],
}
r = _phase1_check_completion(_build_eval_payload(ctx1))
check("All complete -> Phase 2", r is None)

# Incomplete item
ctx2 = {
    "SubTaskName": "Test",
    "todoItems": [{"itemId": "i1", "description": "t"}, {"itemId": "i2", "description": "t"}],
    "ItemstateUpdates": [{"itemId": "i1", "state": 1}, {"itemId": "i2", "state": 0}],
    "KeyInformation": [{"itemId": "i1", "outputabstract": "ok"}],
}
r = _phase1_check_completion(_build_eval_payload(ctx2))
check("Incomplete item -> retry", r is not None and r["decision"] == "retry")

# Missing KeyInformation
ctx3 = {
    "SubTaskName": "Test",
    "todoItems": [{"itemId": "i1", "description": "t"}],
    "ItemstateUpdates": [{"itemId": "i1", "state": 1}],
    "KeyInformation": [],
}
r = _phase1_check_completion(_build_eval_payload(ctx3))
check("Missing KeyInfo -> retry", r is not None and r["decision"] == "retry")

# Empty todoItems
ctx4 = {
    "SubTaskName": "Test",
    "todoItems": [],
    "ItemstateUpdates": [],
    "KeyInformation": [],
}
r = _phase1_check_completion(_build_eval_payload(ctx4))
check("Empty todoItems -> retry", r is not None and r["decision"] == "retry")

# No ItemstateUpdates at all
ctx5 = {
    "SubTaskName": "Test",
    "todoItems": [{"itemId": "i1", "description": "t"}],
    "ItemstateUpdates": [],
    "KeyInformation": [],
}
r = _phase1_check_completion(_build_eval_payload(ctx5))
check("No ItemstateUpdates -> retry", r is not None and r["decision"] == "retry")

# === agents.py ===
print("\n--- agents.py ---")
from agents import _extract_json_object, _normalize_agent_context, _build_dependency_context_isolated

# JSON extraction
check("Parse clean JSON", _extract_json_object('{"a": 1}') == {"a": 1})
check("Parse fenced JSON", _extract_json_object('```json\n{"b": 2}\n```') == {"b": 2})
check("Parse embedded JSON", _extract_json_object('text {"c": 3} more') == {"c": 3})
check("Empty string", _extract_json_object("") == {})

# Normalize fills missing items
base = {"todoItems": [{"itemId": "i1", "description": "a"}, {"itemId": "i2", "description": "b"}]}
parsed = {"ItemstateUpdates": [{"itemId": "i1", "state": 1}]}
norm = _normalize_agent_context(parsed, base)
ids = {u["itemId"] for u in norm["ItemstateUpdates"]}
check("Auto-fill missing ItemstateUpdates", "i2" in ids)
i2_state = next(u["state"] for u in norm["ItemstateUpdates"] if u["itemId"] == "i2")
check("Auto-filled state is 0", i2_state == 0)

# Context isolation - KEY TEST
mock_state = {
    "task_results": {"G1": "This is a very long full raw output that should NOT appear in isolated context"},
    "agent_contexts": {
        "G1": {
            "AgentName": "sub1",
            "KeyInformation": [{"itemId": "i1", "outputabstract": "NIO revenue 55B CNY, BYD 600B CNY"}],
        }
    }
}
ctx = _build_dependency_context_isolated(mock_state, ["G1"])
check("KeyInfo in isolated context", "NIO revenue 55B CNY" in ctx)
check("Full output NOT leaked", "very long full raw" not in ctx)

# No deps
ctx = _build_dependency_context_isolated(mock_state, [])
check("No deps -> None", ctx == "None")

# Dep with no KeyInformation (fallback truncation)
mock_state2 = {
    "task_results": {"G5": "A" * 500},
    "agent_contexts": {"G5": {"AgentName": "sub5", "KeyInformation": []}},
}
ctx = _build_dependency_context_isolated(mock_state2, ["G5"])
check("Fallback truncated", "[truncated]" in ctx)
check("Fallback <= 400 chars", len(ctx) < 500)

# === Summary ===
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
