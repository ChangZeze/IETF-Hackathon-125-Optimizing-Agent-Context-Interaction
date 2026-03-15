"""
Metrics Statistics Module (ACP Structured Context Version)

Responsible for tracking and statistics of performance metrics during task execution.
Main features:
- Record token usage of LLM calls (separated by master and subagent)
- Thread-safe: uses a lock since parallel agents call add_usage concurrently
- Statistics of task execution time
- Provide interfaces for querying and resetting metrics
"""

import threading
import time
from typing import Any, Dict

_stats_lock = threading.Lock()

_RUN_STATS = {
    "master_prompt_tokens": 0,
    "master_completion_tokens": 0,
    "master_total_tokens": 0,
    "subagent_prompt_tokens": 0,
    "subagent_completion_tokens": 0,
    "subagent_total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "started_at": None,
    "ended_at": None,
}


def reset_run_stats() -> None:
    with _stats_lock:
        _RUN_STATS["master_prompt_tokens"] = 0
        _RUN_STATS["master_completion_tokens"] = 0
        _RUN_STATS["master_total_tokens"] = 0
        _RUN_STATS["subagent_prompt_tokens"] = 0
        _RUN_STATS["subagent_completion_tokens"] = 0
        _RUN_STATS["subagent_total_tokens"] = 0
        _RUN_STATS["prompt_tokens"] = 0
        _RUN_STATS["completion_tokens"] = 0
        _RUN_STATS["total_tokens"] = 0
        _RUN_STATS["started_at"] = time.perf_counter()
        _RUN_STATS["ended_at"] = None


def finish_run_stats() -> None:
    with _stats_lock:
        _RUN_STATS["ended_at"] = time.perf_counter()


def add_usage(usage: Any, agent_type: str = "subagent") -> None:
    """
    Add token usage statistics

    Args:
        usage: Usage object from LLM response
        agent_type: "master" or "subagent"
    """
    if usage is None:
        return

    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)

    if prompt is None and isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
    if completion is None and isinstance(usage, dict):
        completion = usage.get("completion_tokens")
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")

    with _stats_lock:
        if prompt:
            _RUN_STATS["prompt_tokens"] += int(prompt)
            if agent_type == "master":
                _RUN_STATS["master_prompt_tokens"] += int(prompt)
            else:
                _RUN_STATS["subagent_prompt_tokens"] += int(prompt)

        if completion:
            _RUN_STATS["completion_tokens"] += int(completion)
            if agent_type == "master":
                _RUN_STATS["master_completion_tokens"] += int(completion)
            else:
                _RUN_STATS["subagent_completion_tokens"] += int(completion)

        if total:
            _RUN_STATS["total_tokens"] += int(total)
            if agent_type == "master":
                _RUN_STATS["master_total_tokens"] += int(total)
            else:
                _RUN_STATS["subagent_total_tokens"] += int(total)


def get_run_stats() -> Dict[str, Any]:
    with _stats_lock:
        started = _RUN_STATS["started_at"]
        ended = _RUN_STATS["ended_at"] if _RUN_STATS["ended_at"] is not None else time.perf_counter()
        elapsed = (ended - started) if started is not None else None

        return {
            "master_prompt_tokens": _RUN_STATS["master_prompt_tokens"],
            "master_completion_tokens": _RUN_STATS["master_completion_tokens"],
            "master_total_tokens": _RUN_STATS["master_total_tokens"],
            "subagent_prompt_tokens": _RUN_STATS["subagent_prompt_tokens"],
            "subagent_completion_tokens": _RUN_STATS["subagent_completion_tokens"],
            "subagent_total_tokens": _RUN_STATS["subagent_total_tokens"],
            "prompt_tokens": _RUN_STATS["prompt_tokens"],
            "completion_tokens": _RUN_STATS["completion_tokens"],
            "total_tokens": _RUN_STATS["total_tokens"],
            "elapsed_seconds": elapsed,
        }