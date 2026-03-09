"""
指标统计模块

负责跟踪和统计任务执行过程中的性能指标。
主要功能：
- 记记录LLM调用的token使用量
- 统计任务执行耗时
- 提供指标查询和重置接口
"""

import time
from typing import Any, Dict

_RUN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "started_at": None,
    "ended_at": None,
}


def reset_run_stats() -> None:
    _RUN_STATS["prompt_tokens"] = 0
    _RUN_STATS["completion_tokens"] = 0
    _RUN_STATS["total_tokens"] = 0
    _RUN_STATS["started_at"] = time.perf_counter()
    _RUN_STATS["ended_at"] = None


def finish_run_stats() -> None:
    _RUN_STATS["ended_at"] = time.perf_counter()


def add_usage(usage: Any) -> None:
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

    if prompt:
        _RUN_STATS["prompt_tokens"] += int(prompt)
    if completion:
        _RUN_STATS["completion_tokens"] += int(completion)
    if total:
        _RUN_STATS["total_tokens"] += int(total)


def get_run_stats() -> Dict[str, Any]:
    started = _RUN_STATS["started_at"]
    ended = _RUN_STATS["ended_at"] if _RUN_STATS["ended_at"] is not None else time.perf_counter()
    elapsed = (ended - started) if started is not None else None

    return {
        "prompt_tokens": _RUN_STATS["prompt_tokens"],
        "completion_tokens": _RUN_STATS["completion_tokens"],
        "total_tokens": _RUN_STATS["total_tokens"],
        "elapsed_seconds": elapsed,
    }