"""
Sub-agent module (PlainText Context version)

Responsible for managing sub-agent capability definitions, dependency context
construction, LLM calls, and result parsing.
Differences from the Structured Context version:
- Uses plain text to pass context instead of structured JSON
- Uses a simplified prompt construction method
- Does not require complex AgentContext structure parsing
"""

from config import client
from metrics import add_usage

MODEL = "deepseek-chat"

SUB_AGENT_CAPABILITIES = {
    "sub1": "Responsible for collecting key financial data of auto companies, without analysis.",
    "sub2": "Responsible for analyzing profitability and gross margin structure of auto companies.",
    "sub3": "Responsible for analyzing cost control and operational efficiency of auto companies.",
    "sub4": "Responsible for integrating sub2 and sub3 outputs and producing key conclusions.",
    "sub5": "Responsible for collecting broker ratings, stock performance, and expert opinions of auto companies.",
    "sub6": "Responsible for collecting NEV policy information and summarizing impacts.",
    "sub7": "Responsible for integrating financial, market, and policy information to output risk assessments.",
    "sub8": "Responsible for integrating conclusions and risks, then providing investment advice and ranking.",
}

SUB_AGENT_INPUT_HINTS = {
    "sub1": [],
    "sub2": ["sub1"],
    "sub3": ["sub1"],
    "sub4": ["sub2", "sub3"],
    "sub5": [],
    "sub6": [],
    "sub7": ["sub1", "sub5", "sub6"],
    "sub8": ["sub4", "sub7"],
}


def call_llm(prompt: str, agent_name: str):
    print(f"\n===== {agent_name} token stream =====")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a financial analysis expert who strictly follows output requirements."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            add_usage(usage)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta else None
        if text:
            print(text, end="", flush=True)
            parts.append(text)

    print(f"\n===== {agent_name} stream end =====\n")
    return "".join(parts)


def _build_context_block(state, agent_name: str) -> str:  #用于存储该agent任务执行相关的上下文内容。
    deps = SUB_AGENT_INPUT_HINTS.get(agent_name, [])
    if not deps:
        return ""

    parts = []
    task_results = state.get("task_results", {})
    for dep in deps:
        value = task_results.get(dep, "")
        if value:
            parts.append(f"[{dep} output]\n{value}")

    return "\n\n".join(parts)


def run_sub_agent(state, agent_name: str, command: str):
    state["current_agent"] = agent_name

    capability = SUB_AGENT_CAPABILITIES.get(agent_name, "General analysis and output")
    context_block = _build_context_block(state, agent_name)

    prompt = f"""
You are {agent_name}.
Capability boundary: {capability}

Main-agent instruction:
{command}

Relevant context (if any):
{context_block if context_block else "None"}

Execution requirements:

1. Only complete tasks within this sub-agent's responsibility scope.
2. Keep output structure clear; use short headings when necessary.
3. Do not explain which agent you are, and do not output JSON text.

"""
    


    result = call_llm(prompt, agent_name)
    state["task_results"][agent_name] = result
    return state


    '''
    When generating output content, consider the following points (internalize them in your reasoning), but do not directly output these points.

1. **Task Focus and Parsing**:
   - Carefully parse the main-agent instruction to ensure accurate understanding of the task objective and requirements.
   - Strictly follow the instruction and only complete tasks within this sub-agent's responsibility scope to avoid crossing capability boundaries.
   - If the instruction is ambiguous or unclear, explicitly point it out and request further clarification from the main agent.

2. **Information Processing and Analysis**:
   - Verify completeness, accuracy, and relevance of input data to ensure a reliable basis for analysis.
   - Analyze relevant context deeply and perform logical reasoning and judgment based on current instruction requirements.
   - Distinguish facts, assumptions, and inferences during processing, and mark them clearly.

3. **Output Structuring and Formatting**:
   - Output should be clearly structured and layered; use section headings/subheadings when needed.
   - Use accurate and concise language to keep content easy to understand, avoiding redundancy and ambiguity.
   - Use punctuation, capitalization, and indentation appropriately to improve readability.

4. **Logical Reasoning and Decision-making**:
   - For complex problems, perform systematic logical reasoning to ensure reasonable and accurate conclusions.
   - If a decision is required, list all feasible options, analyze pros/cons, and choose the best option based on analysis.

5. **Error Handling and Exceptional Cases**:
   - If you encounter unprocessable issues or exceptions, clearly state the specific problem and obstacle.
   - Provide possible solutions or next-step actions to ensure task continuity and completeness.

6. **Privacy and Security**:
   - Strictly follow privacy and security protocols to avoid sensitive information leakage.
   - When handling privacy-related or sensitive information, explicitly mark it and limit its propagation scope.
   '''
