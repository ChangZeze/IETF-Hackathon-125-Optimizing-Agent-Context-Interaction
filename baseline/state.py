"""
状态类型定义模块

定义全局任务状态相关的类型结构。
包含：
- ACPState：全局任务状态类型定义
"""

from typing import TypedDict, Dict, Any


class ACPState(TypedDict):
    task_results: Dict[str, Any]
    current_agent: str
    retry_count: int
    eval_decision: str