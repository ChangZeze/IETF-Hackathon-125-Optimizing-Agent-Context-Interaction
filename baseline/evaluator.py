"""
Evaluation module (PlainText Context version)

Responsible for evaluating and deciding on sub-agent outputs by the main agent.
Differences from the Structured Context version:
- Directly evaluates plain-text output content
- Uses simplified evaluation logic
- Uses string matching instead of structured data
"""

from config import client
from metrics import add_usage

MAX_RETRY = 2


def evaluate_by_master(agent_name: str, command: str, content: str, retry_count: int):
    prompt = f"""
You are the evaluation module of the main agent. Please carefully analyze the sub-agent output and decide whether it satisfies the instruction. The acceptance bar does not need to be too high.

Sub-agent: {agent_name}
Original instruction:
{command}

Sub-agent output:
{content[:4000]}

Please output strict JSON:
{{
  "decision": "pass" or "retry" or "force_pass",
  "feedback": "If retry is needed, provide concise and actionable revision suggestions; otherwise it can be empty."
}}

The task acceptance bar does not need to be too high.

Decision rules:
1. Output content matches requirements -> pass
2. Content completely does not match requirements -> retry
3. Maximum retries reached and still unsatisfactory -> force_pass
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    add_usage(getattr(response, "usage", None))
    raw = response.choices[0].message.content.strip()

    decision = "retry"
    feedback = raw

    lower_raw = raw.lower()
    if '"decision"' in lower_raw and "pass" in lower_raw:
        decision = "pass"
    if '"decision"' in lower_raw and "force_pass" in lower_raw:
        decision = "force_pass"
    if '"decision"' in lower_raw and '"retry"' in lower_raw:
        decision = "retry"

    next_retry = retry_count
    if decision == "retry":
        next_retry = retry_count + 1
        if next_retry >= MAX_RETRY:
            decision = "force_pass"

    if decision == "pass":
        next_retry = 0

    return {
        "decision": decision,
        "feedback": feedback,
        "retry_count": next_retry,
        "raw_eval": raw,
    }
